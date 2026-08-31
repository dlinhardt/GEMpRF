import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cupy as cp

# gem
from gem.space.PRFSpace import PRFSpace

from gem.fitting.hpc_coefficient_matrix import CoefficientMatrix
from gem.utils.hpc_cupy_utils import HpcUtils as Utils
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

# debugging purpose
import nibabel as nib
import os

class RefineFit:
    padded_arr_2d_location_inv_M = None
    padded_multi_dim_points_neighbours_flat_indices = None

    @classmethod
    def _prepare_padded_arrays(cls, arr_2d_location_inv_M_cpu, prf_space):
        """Prepare padded arrays for arr_2d_location_inv_M and multi_dim_points_neighbours.

        NOTE: both stay in host memory. The padded M-inverse is
        (num_model_signals, 10, max_neighbours * 4) float64 -- several GB for a dense grid -- but only
        the ``num_y_signals`` rows of the current batch are ever read, so keeping it on the device
        wasted most of a card for a few MB of gathers per batch.
        """
        # --- 1a) arr_2d_location_inv_M: already padded by the numba kernel, so take it as it is ---
        # Wrapper_Grids2MpInv_numba fills a dense (N, 10, max_cols) buffer and hands it over in an
        # MpInvTable. Rebuilding that array here -- which is what this used to do, at ~9.4M Python
        # slices and 8 GB of fresh pages for a 942k-point grid -- was the whole of the "first file is
        # 5x slower than every later file" effect, because this runs lazily inside the first batch.
        #
        # PADDING: the kernel leaves the pad region 0.0 where this used to leave np.nan. Both are only
        # ever read through _compute_coefficients, which does nan_to_num(MpInv, nan=0.0) before the
        # einsum, so the two are equal element for element after masking.
        if hasattr(arr_2d_location_inv_M_cpu, "padded"):
            padded_arr_cpu = arr_2d_location_inv_M_cpu.padded
        else:
            # Fallback for callers that still hand over a plain ragged list (tests, and the leftover
            # reference implementations below).
            arr_2d_location_inv_M_cpu_list = list(arr_2d_location_inv_M_cpu)
            N = len(arr_2d_location_inv_M_cpu_list)
            R = arr_2d_location_inv_M_cpu_list[0].shape[0]
            cols = np.array([a.shape[1] for a in arr_2d_location_inv_M_cpu_list], dtype=int)
            max_cols = int(cols.max())
            padded_arr_cpu = np.zeros((N, R, max_cols), dtype=np.float64)
            for i, a in enumerate(arr_2d_location_inv_M_cpu_list):
                padded_arr_cpu[i, :, :a.shape[1]] = a

        # --- 1b) Prepare multi_dim_points_neighbours_flat_indices ---
        # The neighbour indices are a flat buffer plus per-point counts, so np.array_split is the
        # exact inverse of how they were concatenated and costs no copying.
        neigh_flat, lens = prf_space.get_neighbour_flat_indices_and_counts()
        max_len = int(lens.max())

        # Scatter straight into the padded result: row i gets its lens[i] values, the rest stays -1.
        padded_neigh_cpu = np.full((lens.shape[0], max_len), -1, dtype=np.int64)
        fill_mask = np.arange(max_len)[None, :] < lens[:, None]
        padded_neigh_cpu[fill_mask] = neigh_flat

        # --- Both stay on the host; only the per-batch slice is moved to the device ---
        cls.padded_arr_2d_location_inv_M = padded_arr_cpu
        cls.padded_multi_dim_points_neighbours_flat_indices = padded_neigh_cpu[:, :, None]


    @classmethod
    def prepare(cls, arr_2d_location_inv_M_cpu, prf_space):
        """Build the padded lookup tables up front, during setup.

        The lazy guards below would do this on the first refined fit, which puts a one-time cost
        inside the region the per-file timer wraps -- the first file of a run then looks dramatically
        slower than every file after it. Calling this once after the M-inverse thread joins keeps the
        guards as a fallback for callers that skip setup (the tests, mainly) without ever letting them
        fire in a real run.
        """
        if arr_2d_location_inv_M_cpu is None:
            return
        cls._prepare_padded_arrays(arr_2d_location_inv_M_cpu, prf_space)

    @classmethod
    def get_neighbour_columns(cls, prf_space: PRFSpace, best_fit_proj, arr_2d_location_inv_M_cpu):
        """Columns of the error/derivative matrices that the refinement will read.

        Returns a host (num_Y_signals, max_neighbours) int array of validated-grid column indices for
        the neighbourhood of each y-signal's winning grid point, padded with -1. The caller needs
        these before the derivative terms are computed, which is why this is split out of
        ``get_refined_fit_results``.
        """
        if cls.padded_arr_2d_location_inv_M is None or cls.padded_multi_dim_points_neighbours_flat_indices is None:
            cls._prepare_padded_arrays(arr_2d_location_inv_M_cpu, prf_space)

        best_fit_proj_cpu = cls._as_host_indices(best_fit_proj)
        all_block_flat_indices_cpu = cls.padded_multi_dim_points_neighbours_flat_indices[best_fit_proj_cpu].squeeze()
        return prf_space.get_full_2_validated_indices(all_block_flat_indices_cpu, invalid_key_value=-1).reshape(all_block_flat_indices_cpu.shape)

    @classmethod
    def _as_host_indices(cls, best_fit_proj):
        return cp.asnumpy(best_fit_proj) if isinstance(best_fit_proj, cp.ndarray) else np.asarray(best_fit_proj)

    @classmethod
    def get_refined_fit_results(cls, prf_space: PRFSpace, best_fit_proj,
                                arr_2d_location_inv_M_cpu, refine_input_vecs, neighbour_columns):
        """Solve the local quadratic for every y-signal.

        ``refine_input_vecs`` is (num_Y_signals, max_neighbours, num_params + 1), already gathered at
        ``neighbour_columns`` by GridFit -- the dense (num_Y_signals, num_model_signals, num_params+1)
        block this used to build itself was the largest single allocation in the fit.
        """
        on_gpu = isinstance(refine_input_vecs, cp.ndarray)
        pkg = (np, cp)[on_gpu]

        if cls.padded_arr_2d_location_inv_M is None or cls.padded_multi_dim_points_neighbours_flat_indices is None:
            cls._prepare_padded_arrays(arr_2d_location_inv_M_cpu, prf_space)

        # ---------- Gather MpInv for this batch (host-side fancy index, then one small upload) ----------
        best_fit_proj_cpu = cls._as_host_indices(best_fit_proj)
        all_MpInv = cls.padded_arr_2d_location_inv_M[best_fit_proj_cpu]
        if on_gpu:
            all_MpInv = cp.asarray(all_MpInv)

        validated_indices = pkg.asarray(neighbour_columns)

        # ---------- Compute coefficients: coefficients = MpInv@vec ----------
        coefficients = cls._compute_coefficients(pkg, refine_input_vecs, validated_indices, all_MpInv)

        # ---------- Build A, B, C ----------
        A, B, C = CoefficientMatrix.create_cofficients_matrices_A_B_and_C_vectorized(coefficients)

        # ---------- Solve system ----------
        # pinv rather than inv: some of the A matrices are singular
        return pkg.einsum('fij,fj->fi', pkg.linalg.pinv(2*A), -B)


    # ----------------- Helper functions -----------------
    @classmethod
    def _compute_coefficients(cls, pkg, vecs, validated_indices, MpInv):
        """Compute coefficients from the pre-gathered neighbourhood values.

        About validated_indices ...
        ...it has shape (number_of_signals, max_possible_neighbours_per_prf)
        ...it stores neighbor indices for each pRF
        ...since some pRFs have fewer neighbors, validated_indices is padded with -1 for missing entries
        ...the padded slots are masked to NaN here and then zeroed, exactly as before
        """
        vecs = pkg.where(validated_indices[..., None] == -1, pkg.nan, vecs) # Mask invalid indices

        # Flatten vecs -- (num_Y_signals, (num_params+1) * max_neighbours), neighbour-major
        flattened_vecs = vecs.reshape(vecs.shape[0], -1)
        del vecs

        # Set NaNs to 0 so that we can do matrix multiplication and the values do not affect the result
        MpInv_masked = pkg.nan_to_num(MpInv, nan=0.0)
        vecs_masked = pkg.nan_to_num(flattened_vecs, nan=0.0)
        del flattened_vecs

        # Compute coefficients
        coefficients = pkg.einsum('fvi,fi->fv', MpInv_masked, vecs_masked) # coefficients = MpInv@vec
        del MpInv_masked, vecs_masked

        return coefficients

    @classmethod
    def get_refined_fit_results_simpler_padded_arrays(cls, prf_space : PRFSpace, num_Y_signals, best_fit_proj, arr_2d_location_inv_M_cpu, e_full, de_dtheta_3darr):  
        """This is just a leftover function which contains the simpler version of padded arrays creation. It is kept here just for reference."""
        on_gpu = isinstance(e_full, cp.ndarray)
  
        pkg = (np, cp)[isinstance(e_full, cp.ndarray)]
        # ---------- Step 1: Create padded arrays ON GPU ----------
        if cls.padded_arr_2d_location_inv_M is None or cls.padded_multi_dim_points_neighbours_flat_indices is None:
            # Pre-convert all arrays to GPU once before the loop
            arr_2d_location_inv_M_list = [pkg.asarray(a) for a in arr_2d_location_inv_M_cpu] if on_gpu else arr_2d_location_inv_M_cpu
            multi_dim_points_neighbours_flat_indices_list = [pkg.asarray(a) for a in prf_space.multi_dim_points_neighbours_flat_indices] if on_gpu else prf_space.multi_dim_points_neighbours_flat_indices

            num_total_model_signals = len(arr_2d_location_inv_M_cpu)
            num_rows_arr_2d_location_inv_M_cpu = arr_2d_location_inv_M_cpu[0].shape[0]
            num_cols_arr_2d_location_inv_M_cpu = max(arr.shape[1] for arr in arr_2d_location_inv_M_cpu)
            num_rows_multi_dim_points_neighbours_flat_indices = max(
                arr.shape[0] for arr in prf_space.multi_dim_points_neighbours_flat_indices
            )
            num_cols_multi_dim_points_neighbours_flat_indices = 1  # since these are 1D arrays

            # Allocate padded arrays directly on GPU
            cls.padded_arr_2d_location_inv_M = pkg.full(
                (num_total_model_signals, num_rows_arr_2d_location_inv_M_cpu, num_cols_arr_2d_location_inv_M_cpu),
                pkg.nan, dtype=pkg.float64
            )

            cls.padded_multi_dim_points_neighbours_flat_indices = pkg.full(
                (num_total_model_signals, num_rows_multi_dim_points_neighbours_flat_indices, num_cols_multi_dim_points_neighbours_flat_indices),
                -1, dtype=pkg.int64
            )

            # Fill GPU arrays
            for i in range(num_total_model_signals):
                cls.padded_arr_2d_location_inv_M[i, :arr_2d_location_inv_M_cpu[i].shape[0], :arr_2d_location_inv_M_cpu[i].shape[1]] = arr_2d_location_inv_M_list[i]
                cls.padded_multi_dim_points_neighbours_flat_indices[i, :prf_space.multi_dim_points_neighbours_flat_indices[i].shape[0], :prf_space.multi_dim_points_neighbours_flat_indices[i].shape[1]] = multi_dim_points_neighbours_flat_indices_list[i]

        # ---------- Step 2: Gather indices ----------
        all_block_flat_indices = cls.padded_multi_dim_points_neighbours_flat_indices[best_fit_proj].squeeze()
        all_block_flat_indices_cpu = pkg.asnumpy(all_block_flat_indices) if on_gpu else all_block_flat_indices
        all_validated_block_flat_indices_cpu = prf_space.get_full_2_validated_indices(all_block_flat_indices_cpu, invalid_key_value=-1).reshape(all_block_flat_indices_cpu.shape)
        all_validated_block_flat_indices = pkg.asarray(all_validated_block_flat_indices_cpu)

        # ---------- Step 3: Gather MpInv ----------
        all_MpInv = cls.padded_arr_2d_location_inv_M[best_fit_proj]

        # ---------- Step 4: Prepare de_dtheta + e_full ----------
        de_dtheta_transposed = de_dtheta_3darr.transpose(1, 2, 0)  # -> (num_Y_signals, num_models, num_params)
        e_full_expanded = e_full[:, :, pkg.newaxis]  # (num_Y_signals, num_models, 1)

        combined = pkg.concatenate([e_full_expanded, de_dtheta_transposed], axis=2)

        # ---------- Step 5: Gather vecs ----------
        vecs = combined[
            pkg.arange(num_Y_signals)[:, None],
            all_validated_block_flat_indices.clip(min=0)
        ]
        vecs = pkg.where(all_validated_block_flat_indices[..., None] == -1, pkg.nan, vecs)

        flattened_vecs = vecs.reshape(vecs.shape[0], -1)  # (num_Y_signals, (num_params+1)*max_neighbors)

        # ---------- Step 6: Compute coefficients ----------
        MpInv_masked = pkg.nan_to_num(all_MpInv, nan=0.0)
        vecs_masked = pkg.nan_to_num(flattened_vecs, nan=0.0)

        coefficients = pkg.einsum('fvi,fi->fv', MpInv_masked, vecs_masked)

        # ---------- Step 7: Build A, B, C ----------
        # Assuming this function supports CuPy input
        A, B, C = CoefficientMatrix.create_cofficients_matrices_A_B_and_C_vectorized(coefficients)

        # ---------- Step 8: Solve system ----------
        refined_params_vecs_gpu = pkg.einsum('fij,fj->fi', pkg.linalg.pinv(2*A), -B)

        return refined_params_vecs_gpu, None    


    @classmethod
    def get_refined_fit_results_cpu_loop_based(cls, prf_space : PRFSpace, num_Y_signals, best_fit_proj_cpu, arr_2d_location_inv_M_cpu, e_full, de_dtheta_list_cpu):  
        """This is a leftover function which contains the CPU version of the refine fit. It is kept here just for reference."""    
        #NOTE: for DEBUG info, look into the old-gem-files folder
        ONLY_SIGNLE_SIGNAL = False
        if(num_Y_signals == 1):
            ONLY_SIGNLE_SIGNAL = True

        de_dtheta_list_cpu = np.array(de_dtheta_list_cpu).squeeze()
        num_params = len(de_dtheta_list_cpu)
        results = np.zeros((num_Y_signals, num_params), dtype=np.float64)

        Fex_results = []
        # perform refine search
        for y_idx in range(num_Y_signals):
            best_s_idx = best_fit_proj_cpu[y_idx]            
            block_flat_indices = prf_space.multi_dim_points_neighbours_flat_indices[best_s_idx]            

            # NOTE: in case of validating the pRF points, the number of total multi-dimensional points used to compute model signals are changed. 
            # However, the neighbours indices represent the indices for the full multi-points array. 
            # Therefore, we need to map the indices corresponding to full-array to indices corresponding to the validated points array.
            block_flat_indices = prf_space.get_full_2_validated_indices(block_flat_indices, invalid_key_value=None)

            # compute the coffeficients
            #...get the pre-computed Mp Inverse matrix (already containing information about the neighbors)
            MpInv = arr_2d_location_inv_M_cpu[best_s_idx] # NOTE: This is the correct way for the program, but for debugging, I am not using it and computing it again       
        
            #...compute the de/dx, de/dy and de/dsigma vectors# 
            if type(e_full) is cp.ndarray:
                default_gpu_id = ggm.get_instance().default_gpu_id
                with cp.cuda.Device(default_gpu_id):
                    e_vec = e_full[y_idx, block_flat_indices] # NOTE: ** 2 is required if we are taking error term as (yts)^2    
            else:
                e_vec = e_full[y_idx, block_flat_indices]  
            if len(e_vec) == 1: # i.e. no other neighbours
                vec = e_vec.squeeze()
            else:              
                vec = np.vstack((e_vec.squeeze())).squeeze()   
                non_NaN_e_row_indices = cls.get_non_nan_row_indices(vec)         
            for theta in range(num_params):
                if(ONLY_SIGNLE_SIGNAL):     
                    vec = np.vstack((vec, (de_dtheta_list_cpu[theta, block_flat_indices].T)[0]))
                else:
                    vec = np.vstack((vec, (de_dtheta_list_cpu[theta, y_idx, block_flat_indices].T)[0]))

            vec = vec[:, non_NaN_e_row_indices]
            vec = vec.T.reshape(-1)

            # in case of concatenation runs, we have `vec` as cupy array
            if type(vec) is cp.ndarray:
                with cp.cuda.Device(default_gpu_id):
                    vec = cp.asnumpy(vec)

            # compute non_nan_indices for M matrix            
            if len(non_NaN_e_row_indices) != len(block_flat_indices): # i.e. some of the indices are NaN
                non_NaN_e_row_indices = cp.asnumpy(non_NaN_e_row_indices)
                good_indices_M = []
                num_linear_equations = num_params + 1
                for i in non_NaN_e_row_indices:
                    good_indices_M.extend(range(i * num_linear_equations, (i + 1) * num_linear_equations))
                MpInv = MpInv[:, good_indices_M]
                                
            # compute the coefficients
            coefficients = MpInv@vec                        
            A, B, C = CoefficientMatrix.create_cofficients_matrices_A_B_and_C(coefficients)
            try:
                refined_params_vec = np.linalg.solve(2*A, -1 * B) # solve for X the derivative equation, 2AX + B = 0 ==> 2AX = -B                
            except:
                refined_params_vec = np.array([np.nan, np.nan, np.nan])
                
            # update results    
            results[y_idx, :] = refined_params_vec
            
            fex = cls.compute_fx(refined_params_vec, A, B, C)
            Fex_results.append(fex)

        return results, Fex_results             

    @classmethod
    def get_non_nan_row_indices(cls, input_array):
        package = None
        if type(input_array) is cp.ndarray:
            package = cp
        elif type(input_array) is np.ndarray:
            package = np
        
        if input_array.ndim == 1:
            return package.where(~package.isnan(input_array))[0]
        else:
            return package.where(~package.isnan(input_array).any(axis=1))[0]


    @classmethod
    def compute_fx(cls, refined_X, A, B, C):
        # e = X.T @ (A @ X) + B@X + C

        X = np.asarray(refined_X)

        # X = np.asarray([muX, muY, sigma])
        fex = X.T @ (A @ X) + B@X + C

        return fex    
    
    @classmethod
    def compute_fx_and_gradients(cls, refined_X, A, B, C):
        # e = X.T @ (A @ X) + B@X + C

        X = np.asarray(refined_X)

        # X = np.asarray([muX, muY, sigma])
        fex = X.T @ (A @ X) + B@X + C
        gradient_fex = 2 * (A @ X) + B

        return fex, gradient_fex

    #############################################---------------------Debug related code------------------------------######################################################
    #############################################---------------------Debug related code------------------------------######################################################
    #########----for DEBUG
    @classmethod
    def save_data_to_nifiti(cls, selected_y_signal_idx, X_3d, Y_3d, Sigma_3d, selected_signal_e, selected_signal_de_dx, selected_signal_de_dy, selected_signal_de_dsigma, A, B, C):
        dir_path = r'D:\results\gradients-test\saved-debug-data'        

        # dimensions
        nRows, nCols, nSigma = X_3d.shape

        # affine transform matrix
        dx = X_3d[0, 1, 0] - X_3d[0, 0, 0] 
        dy = Y_3d[1, 0, 0] - Y_3d[0, 0, 0] 
        ds = Sigma_3d[0, 0, 1] - Sigma_3d[0, 0, 0] 
        Ox = X_3d[0, 0, 0]
        Oy = Y_3d[0, 0, 0]
        Os = Sigma_3d[0, 0, 0]
        affine_mat = np.array([[dx, 0, 0, Ox], 
                               [0, dy, 0, Oy], 
                               [0, 0, ds, Os], 
                               [0, 0, 0 , 1]])

        # data = np.ones((32, 32, 15, 100), dtype=np.float64)
        # img = nib.Nifti1Image(data, np.eye(4))
        # img.set_data_dtype(np.dtype(np.float64))
        # nib.save(img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error.nii.gz')))

        # save error data
        # plt.figure(); plt.gca().contour(X_3d[:, :, 0] , Y_3d[:, :, 0] , coarse_error[:, :, 0])
        coarse_error = (selected_signal_e.reshape((nRows, nCols, nSigma), order='F'))
        coarse_error_img = nib.Nifti1Image(coarse_error, affine=affine_mat)
        coarse_error_img.set_data_dtype(np.dtype(np.float64))
        nib.save(coarse_error_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_coarse_error.nii.gz')))

        # gradients data
        de_dx = (selected_signal_de_dx.reshape((nRows, nCols, nSigma), order='F'))
        de_dy = (selected_signal_de_dy.reshape((nRows, nCols, nSigma), order='F'))
        de_dsigma = (selected_signal_de_dsigma.reshape((nRows, nCols, nSigma), order='F'))
        gradients = np.array([de_dx, de_dy, de_dsigma])  
        gradients  = np.moveaxis(gradients, 0, -1)
        gradients_img = nib.Nifti1Image(gradients, affine=affine_mat)
        gradients_img.set_data_dtype(np.dtype(np.float64))
        nib.save(gradients_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error_gradients.nii.gz')))

        # quad fitting error e = f(x) AND its gradient (gradients_fex = 2AX + B)
        fex_error = []
        gradients_fex = []
        fex_dx = []        
        fex_dy = []        
        fex_dsigma = []        
        for sigma  in range(nSigma):
            for row in range(nRows):
                for col in range(nCols):                                    
                            X = np.array([X_3d[row, col, sigma], Y_3d[row, col, sigma], Sigma_3d[row, col, sigma]])
                            fex, grad_fex = cls.compute_fx_and_gradients(X, A, B, C)
                            fex_error.append(fex)
                            # gradients_fex.append(grad_fex)
                            fex_dx.append(grad_fex[0])
                            fex_dy.append(grad_fex[1])
                            fex_dsigma.append(grad_fex[2])

        # save f(x)
        fex_error = (np.array(fex_error)).reshape((nRows, nCols, nSigma), order='F')                        
        fex_img = nib.Nifti1Image(fex_error, affine=affine_mat)
        fex_img.set_data_dtype(np.dtype(np.float64))
        nib.save(fex_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error_fex_weighted.nii.gz')))

        # save gradients_fex = 2AX + B    
        fex_dx = np.array([fex_dx])
        fex_dy = np.array([fex_dy])
        fex_dsigma= np.array([fex_dsigma])
        dfx_dx = (fex_dx.reshape((nRows, nCols, nSigma), order='F'))
        dfx_dy = (fex_dy.reshape((nRows, nCols, nSigma), order='F'))
        dfx_dsigma = (fex_dsigma.reshape((nRows, nCols, nSigma), order='F'))
        gradients_fex = np.array([dfx_dx, dfx_dy, dfx_dsigma])
        gradients_fex = np.moveaxis(gradients_fex, 0, -1)
        grad_fex_img = nib.Nifti1Image(gradients_fex, affine=affine_mat)
        grad_fex_img.set_data_dtype(np.dtype(np.float64))
        nib.save(grad_fex_img, os.path.join(dir_path, (f'{selected_y_signal_idx}_error_gradients_fex_weighted.nii.gz')))        

        print()

    @classmethod
    def get_all_debug_info_error_terms_after_refinement(cls, refined_matching_results, Y_signals_gpu, O_gpu, stimulus, stim_height, stim_width, x_range_cpu, y_range_cpu):        
        refined_matching_results_arr = np.array(refined_matching_results)
        muX_arr, muY_arr, sigma_arr = refined_matching_results_arr[: , 0], refined_matching_results_arr[: , 1], refined_matching_results_arr[: , 2]
        
        gaussian_cuda_module = Utils.get_raw_module('gaussian_using_arrays_kernel.cu')
        gc_kernel = gaussian_cuda_module.get_function("gc_using_args_arrays_cuda_Kernel")
        dgc_dx_using_args_arrays_cuda_Kernel = gaussian_cuda_module.get_function("dgc_dx_using_args_arrays_cuda_Kernel")
        dgc_dy_using_args_arrays_cuda_Kernel = gaussian_cuda_module.get_function("dgc_dy_using_args_arrays_cuda_Kernel")
        dgc_dsigma_using_args_arrays_cuda_Kernel = gaussian_cuda_module.get_function("dgc_dsigma_using_args_arrays_cuda_Kernel")

        # initialize result curves
        total_num_gc = len(refined_matching_results)
        result_gc_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)
        result_dgc_dx_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)
        result_dgc_dy_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)
        result_dgc_dsigma_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)

        # kernel grid
        block_dim = (32, 1, 1)
        bx = int((total_num_gc + block_dim[0] - 1) / block_dim[0])
        by = 1
        bz = 1
        grid_dim = (bx, by, bz)

        # launch kernel - gc
        gc_kernel(grid_dim, block_dim, (
        result_gc_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # launch kernel - dgc_dx
        dgc_dx_using_args_arrays_cuda_Kernel(grid_dim, block_dim, (
        result_dgc_dx_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # launch kernel - dgc_dy
        dgc_dy_using_args_arrays_cuda_Kernel(grid_dim, block_dim, (
        result_dgc_dy_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # launch kernel - dgc_dsigma
        dgc_dsigma_using_args_arrays_cuda_Kernel(grid_dim, block_dim, (
        result_dgc_dsigma_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        cp.asarray(x_range_cpu),
        cp.asarray(y_range_cpu),        
        stim_height,
        stim_width,
        total_num_gc))

        # timecourses
        stim_data = stimulus.get_flattened_columnmajor_stimulus_data_gpu()
        nRows_gaussian_curves_matrix = total_num_gc
        nCols_gaussian_curves_matrix = stim_height * stim_width  

        #.....reshape GC
        gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
        dgc_dx_rowmajor_gpu = cp.reshape(result_dgc_dx_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) 
        dgc_dy_rowmajor_gpu = cp.reshape(result_dgc_dy_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) 
        dgc_dsigma_rowmajor_gpu = cp.reshape(result_dgc_dsigma_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) 
        
        #....compute timecourses
        refined_S_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stim_data)
        refined_dS_dx_rowmajor_gpu = cp.dot(dgc_dx_rowmajor_gpu, stim_data)
        refined_dS_dy_signals_rowmajor_gpu = cp.dot(dgc_dy_rowmajor_gpu, stim_data)
        refined_dS_dsigma_signals_rowmajor_gpu = cp.dot(dgc_dsigma_rowmajor_gpu, stim_data)

        #.....orthonormalize
        S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu = cls._get_orthonormalize_refined_signals_for_debug(O_gpu, 
                                                                                                                                                                     refined_S_rowmajor_gpu, 
                                                                                                                                                                     refined_dS_dx_rowmajor_gpu, 
                                                                                                                                                                     refined_dS_dy_signals_rowmajor_gpu, 
                                                                                                                                                                     refined_dS_dsigma_signals_rowmajor_gpu)

        # S        
        e_gpu = cls._compute_refined_error_term_for_debug(O_gpu, Y_signals_gpu, S_prime_columnmajor_gpu)
   
        # dS_dx        
        de_dx_full_gpu = cls._compute_refined_derivative_error_term_cpu_for_debug(O_gpu, Y_signals_gpu, dS_prime_dx_columnmajor_gpu, e_gpu)

        # dS_dy
        de_dy_full_gpu = cls._compute_refined_derivative_error_term_cpu_for_debug(O_gpu, Y_signals_gpu, dS_prime_dy_columnmajor_gpu, e_gpu)
        
        # dS_dsigma
        de_dsigma_full_gpu = cls._compute_refined_derivative_error_term_cpu_for_debug(O_gpu, Y_signals_gpu, dS_prime_dsigma_columnmajor_gpu, e_gpu)
        
        # cpu
        e_cpu = cp.asnumpy(e_gpu)
        de_dx_full_cpu = cp.asnumpy(de_dx_full_gpu)
        de_dy_full_cpu = cp.asnumpy(de_dy_full_gpu)
        de_dsigma_full_cpu = cp.asnumpy(de_dsigma_full_gpu)

        return e_cpu, de_dx_full_cpu, de_dy_full_cpu, de_dsigma_full_cpu
    
    @classmethod
    def get_error_terms_after_refinement(cls, refined_matching_results, Y_signals_gpu, O_gpu, stimulus, stim_height, stim_width, x_range_gpu, y_range_gpu):        
        refined_matching_results_arr = np.array(refined_matching_results)
        muX_arr, muY_arr, sigma_arr = refined_matching_results_arr[: , 0], refined_matching_results_arr[: , 1], refined_matching_results_arr[: , 2]
        
        gaussian_cuda_module = Utils.get_raw_module('gaussian_using_arrays_kernel.cu')
        gc_kernel = gaussian_cuda_module.get_function("gc_using_args_arrays_cuda_Kernel")

        # initialize result curves
        total_num_gc = len(refined_matching_results)
        result_gc_curves_gpu = cp.zeros((total_num_gc * stim_width * stim_height), dtype=cp.float64)

        # kernel grid
        block_dim = (32, 1, 1)
        bx = int((total_num_gc + block_dim[0] - 1) / block_dim[0])
        by = 1
        bz = 1
        grid_dim = (bx, by, bz)

        # launch kernel - gc
        gc_kernel(grid_dim, block_dim, (
        result_gc_curves_gpu,            
        cp.asarray(muX_arr), 
        cp.asarray(muY_arr), 
        cp.asarray(sigma_arr), 
        x_range_gpu,
        y_range_gpu,        
        stim_height,
        stim_width,
        total_num_gc))

        # timecourses
        stim_data = stimulus.get_flattened_columnmajor_stimulus_data_gpu()
        nRows_gaussian_curves_matrix = total_num_gc
        nCols_gaussian_curves_matrix = stim_height * stim_width  

        #.....reshape GC
        gaussian_curves_rowmajor_gpu = cp.reshape(result_gc_curves_gpu, (nRows_gaussian_curves_matrix, nCols_gaussian_curves_matrix)) # each row contains a flat GC
        
        #....compute timecourses
        refined_S_rowmajor_gpu = cp.dot(gaussian_curves_rowmajor_gpu, stim_data)

        #.....orthonormalize
        S_prime_columnmajor_gpu = cls._get_orthonormalize_refined_signals(O_gpu, refined_S_rowmajor_gpu)

        # S        
        e_gpu = cls._compute_refined_error_term_for_debug(O_gpu, Y_signals_gpu, S_prime_columnmajor_gpu)
   
        # cpu
        e_cpu = cp.asnumpy(e_gpu)

        return e_cpu
    
    @classmethod
    def _get_orthonormalize_refined_signals_for_debug(cls, O_gpu, S_rowmajor_gpu, dS_dx_rowmajor_gpu, dS_dy_rowmajor_gpu, dS_dsigma_rowmajor_gpu):
        # orthogonalization + nomalization of signals/timecourses (present along the columns)
        S_star_columnmajor_gpu = cp.dot(O_gpu, S_rowmajor_gpu.T)
        S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns) 
        S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals

        dS_star_dx_columnmajor_gpu = cp.dot(O_gpu, dS_dx_rowmajor_gpu.T)
        dS_star_dy_columnmajor_gpu = cp.dot(O_gpu, dS_dy_rowmajor_gpu.T)    
        dS_star_dsigma_columnmajor_gpu = cp.dot(O_gpu, dS_dsigma_rowmajor_gpu.T)    
    
        dS_prime_dx_columnmajor_gpu = dS_star_dx_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dx_columnmajor_gpu).sum(axis=0))
        dS_prime_dy_columnmajor_gpu = dS_star_dy_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dy_columnmajor_gpu).sum(axis=0))
        dS_prime_dsigma_columnmajor_gpu = dS_star_dsigma_columnmajor_gpu * S_star_S_star_invroot_gpu -  (S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)) * ((S_star_columnmajor_gpu * dS_star_dsigma_columnmajor_gpu).sum(axis=0))

        # test_orthogonalized_tc = (cp.asnumpy(signals_columnmajor_gpu[:, 1]))        
        
        return S_prime_columnmajor_gpu, dS_prime_dx_columnmajor_gpu, dS_prime_dy_columnmajor_gpu, dS_prime_dsigma_columnmajor_gpu   

    @classmethod
    def _get_orthonormalize_refined_signals(cls, O_gpu, S_rowmajor_gpu):
        # orthogonalization + nomalization of signals/timecourses (present along the columns)
        S_star_columnmajor_gpu = cp.dot(O_gpu, S_rowmajor_gpu.T)
        S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns) 
        S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals
    
        return S_prime_columnmajor_gpu

    @classmethod
    def _compute_refined_error_term_for_debug(cls, O_gpu, Y_signals_gpu, S_prime_columnmajor_gpu):
        e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
        return e_gpu
    
    @classmethod
    def _compute_refined_derivative_error_term_cpu_for_debug(cls, O_gpu, Y_signals_gpu, dS_prime_dtheta_columnmajor_gpu, e_gpu):
        de_dtheta_gpu = (Y_signals_gpu.T @ dS_prime_dtheta_columnmajor_gpu)
        de_dtheta_cpu = cp.asnumpy(de_dtheta_gpu)
        return de_dtheta_cpu