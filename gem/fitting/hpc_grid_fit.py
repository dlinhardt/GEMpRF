# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Siddharth Mittal",
"@Desc    :   None",
"""

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

class GridFit:
    @classmethod
    def compute_error_term(cls, Y_signals_gpu, S_prime_columnmajor_gpu):
        e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
        return e_gpu

    @classmethod
    def _compute_best_projection_fit(cls, e_gpu):
        best_fit_proj_gpu = cp.nanargmax(e_gpu, axis=1)     #<<<<----find the max. element's index for the rows along their columns (that's why axis=1)
        return best_fit_proj_gpu

    @classmethod
    def _y_signals_on_device(cls, Y_signals_gpu, device_id, default_gpu_id):
        if device_id == default_gpu_id:
            return Y_signals_gpu
        # ascontiguousarray is a no-op for an already contiguous array, and CuPy cannot copy a
        # non-contiguous one across devices
        with cp.cuda.Device(Y_signals_gpu.device.id):
            contiguous = cp.ascontiguousarray(Y_signals_gpu)
        return cp.array(contiguous)

    @classmethod
    def compute_error_matrix(cls, Y_signals_gpu, S_prime_cm_gpu_batches, out=None, accumulate=False):
        """Build the (num_y_signals, num_model_signals) error matrix on the default device.

        The model signals are spread over the GPUs, so each device's chunk is multiplied where it
        already lives and only the result is brought back -- exactly as before. ``out``/``accumulate``
        let a second concatenated run add into the first run's matrix instead of allocating a second
        full copy and stacking (the stack + sum was the peak allocation of the old code path).
        """
        default_gpu_id = ggm.get_instance().default_gpu_id
        num_batches = len(S_prime_cm_gpu_batches)

        # compute total signals present across all the batches
        total_model_signals = 0
        for i in range(num_batches):
            total_model_signals = total_model_signals + S_prime_cm_gpu_batches[i].shape[1] # i.e.  number of columns: bacuse each signal is present across a single column

        with cp.cuda.Device(default_gpu_id):
            total_y_signals = Y_signals_gpu.shape[1]
            if out is None:
                out = cp.zeros((total_y_signals, total_model_signals), dtype=cp.float64)
                accumulate = False

        # process each batch individually and store the results
        column_idx = 0
        for batch_idx in range(num_batches):
            device_id = S_prime_cm_gpu_batches[batch_idx].device.id
            num_signals_current_batch = S_prime_cm_gpu_batches[batch_idx].shape[1]

            with cp.cuda.Device(device_id):
                current_device_Y_signals = cls._y_signals_on_device(Y_signals_gpu, device_id, default_gpu_id)
                chunk_e_result_gpu = cls.compute_error_term(current_device_Y_signals, S_prime_cm_gpu_batches[batch_idx])
                chunk_e_result_gpu[cp.isinf(chunk_e_result_gpu) & (chunk_e_result_gpu > 0)] = -cp.inf # replace +inf with -inf so that indices with "inf" are not seleted as best fit at the argmax() step

            # finally, get the "error" result back to the default device
            with cp.cuda.Device(default_gpu_id):
                chunk_on_default = chunk_e_result_gpu if device_id == default_gpu_id else cp.array(chunk_e_result_gpu)
                if accumulate:
                    out[:, column_idx : column_idx + num_signals_current_batch] += chunk_on_default
                else:
                    out[:, column_idx : column_idx + num_signals_current_batch] = chunk_on_default
            # NOTE: both names have to go. For a chunk that already lives on the default device they
            # are the same object, so dropping only one keeps a (num_y_signals, chunk_width) array --
            # gigabytes on a dense grid -- alive while the next chunk allocates its own copy.
            del chunk_on_default, chunk_e_result_gpu

            # Note: update index
            column_idx = column_idx + num_signals_current_batch

        return out

    @classmethod
    def get_error_terms(cls, isResultOnGPU, Y_signals_gpu, S_prime_cm_batches_gpu, out=None, accumulate=False):
        """Grid search: the error matrix and the winning grid point per y-signal.

        NOTE: the derivative error terms are deliberately NOT computed here. They are only ever read
        at the <=27 neighbours of the winner, which is not known until the argmax below has run, so
        materialising a dense (num_params, num_y_signals, num_model_signals) array was allocating
        tens of GB to serve a few MB of reads. ``accumulate_derivative_neighbour_terms()`` computes
        them afterwards, from the same per-chunk products, straight into the gathered shape.
        """
        e_gpu = cls.compute_error_matrix(Y_signals_gpu, S_prime_cm_batches_gpu, out=out, accumulate=accumulate)

        with cp.cuda.Device(ggm.get_instance().default_gpu_id):
            # NOTE: the +inf -> -inf replacement already happened per chunk inside
            # compute_error_matrix(), and the assembly cannot introduce new +inf, so no second pass
            # over the full matrix is needed here.
            best_fit_proj_cpu = cp.asnumpy(cp.nanargmax(e_gpu, axis=1))

            # GPU result
            if isResultOnGPU:
                return cp.asarray(best_fit_proj_cpu), e_gpu

            # CPU result: only the (small) index vector goes to the host. The error matrix stays on
            # the device -- callers gather the handful of columns they need from it.
            return best_fit_proj_cpu, e_gpu

    @classmethod
    def compute_matched_error_terms(cls, Y_signals_gpu, S_prime_cm_gpu_batches):
        """Per-signal projection y_i . s'_i, i.e. the diagonal of Y.T @ S' without forming it.

        Used after refinement, where every y-signal has exactly one model signal of its own, so the
        off-diagonal of the full product was computed and thrown away.
        """
        default_gpu_id = ggm.get_instance().default_gpu_id
        parts = []
        column_idx = 0
        for batch in S_prime_cm_gpu_batches:
            num_signals_current_batch = batch.shape[1]
            device_id = batch.device.id

            # NOTE: bring the model-signal chunk to the default device rather than sending a column
            # slice of Y the other way. A column slice is a strided view and CuPy refuses to copy a
            # non-contiguous array between devices; doing it this way keeps the slicing local, where
            # non-contiguity is harmless.
            if device_id == default_gpu_id:
                chunk_on_default = batch
            else:
                with cp.cuda.Device(device_id):
                    contiguous_chunk = cp.ascontiguousarray(batch)
                with cp.cuda.Device(default_gpu_id):
                    chunk_on_default = cp.array(contiguous_chunk)

            with cp.cuda.Device(default_gpu_id):
                Y_slice = Y_signals_gpu[:, column_idx : column_idx + num_signals_current_batch]
                parts.append((Y_slice * chunk_on_default).sum(axis=0))

            column_idx = column_idx + num_signals_current_batch

        with cp.cuda.Device(default_gpu_id):
            matched = cp.concatenate(parts)
            # Same sanitisation the full-matrix path applied: orthonormalize_modelled_signals() marks
            # a degenerate model signal (pRF drifted out of the aperture) by setting its whole column
            # to +inf, and the caller treats a *smaller* error as "refinement made it worse". Leaving
            # +inf here would score those as the best possible fit instead of the worst, so they would
            # never revert to their grid point.
            matched[cp.isinf(matched) & (matched > 0)] = -cp.inf
            return matched

    @classmethod
    def gather_neighbour_terms(cls, error_matrix, neighbour_columns):
        """Pick error_matrix[y, neighbour_columns[y, j]] -> (num_y_signals, num_neighbours).

        Padding entries (-1) are clipped to column 0 here and masked to NaN later by RefineFit,
        which is what the old dense-gather did as well.
        """
        with cp.cuda.Device(ggm.get_instance().default_gpu_id):
            columns_gpu = cp.asarray(neighbour_columns)
            rows = cp.arange(error_matrix.shape[0])[:, None]
            return error_matrix[rows, cp.clip(columns_gpu, 0, error_matrix.shape[1] - 1)]

    @classmethod
    def accumulate_derivative_neighbour_terms(cls, Y_signals_gpu, dS_prime_dtheta_cm_gpu_batches_list,
                                              neighbour_columns, out=None):
        """Derivative error terms, evaluated only at the neighbour columns of the winning grid point.

        For every model-signal chunk this runs the very same ``Y.T @ dS'_chunk`` product the old dense
        path ran, but keeps only the columns that some y-signal actually asks for and then drops the
        chunk. One theta is processed at a time, so at most one (num_y_signals, chunk_width) temporary
        is alive. ``out`` is (num_params, num_y_signals, num_neighbours) on the default device and is
        added to, which is how the concatenated runs are summed.
        """
        default_gpu_id = ggm.get_instance().default_gpu_id
        num_theta_params = len(dS_prime_dtheta_cm_gpu_batches_list)
        if num_theta_params == 0:
            return out

        num_y_signals, num_neighbours = neighbour_columns.shape
        neighbour_columns_cpu = cp.asnumpy(neighbour_columns) if isinstance(neighbour_columns, cp.ndarray) else np.asarray(neighbour_columns)

        with cp.cuda.Device(default_gpu_id):
            if out is None:
                out = cp.zeros((num_theta_params, num_y_signals, num_neighbours), dtype=cp.float64)

        num_chunks = len(dS_prime_dtheta_cm_gpu_batches_list[0])
        column_idx = 0
        for chunk_idx in range(num_chunks):
            reference_chunk = dS_prime_dtheta_cm_gpu_batches_list[0][chunk_idx]
            chunk_width = reference_chunk.shape[1]
            device_id = reference_chunk.device.id

            with cp.cuda.Device(device_id):
                # columns of this chunk, in chunk-local coordinates; everything else is masked off
                local_columns = cp.asarray(neighbour_columns_cpu) - column_idx
                in_chunk = (local_columns >= 0) & (local_columns < chunk_width)
                local_columns = cp.clip(local_columns, 0, chunk_width - 1)
                rows = cp.arange(num_y_signals)[:, None]
                current_device_Y_signals = cls._y_signals_on_device(Y_signals_gpu, device_id, default_gpu_id)

                for theta in range(num_theta_params):
                    # identical product to the old dense path -- same shapes, so same result
                    chunk_de_dtheta_gpu = cls.compute_error_term(current_device_Y_signals,
                                                                 dS_prime_dtheta_cm_gpu_batches_list[theta][chunk_idx])
                    gathered = cp.where(in_chunk, chunk_de_dtheta_gpu[rows, local_columns], 0.0)
                    del chunk_de_dtheta_gpu

                    with cp.cuda.Device(default_gpu_id):
                        out[theta] += gathered if device_id == default_gpu_id else cp.asarray(gathered)
                    del gathered

            column_idx = column_idx + chunk_width

        return out

    @classmethod
    def build_refine_input_vectors(cls, error_neighbour_terms, derivative_neighbour_terms, isResultOnGPU):
        """Stack into the (num_y_signals, num_neighbours, num_params + 1) block RefineFit consumes.

        The last axis is ordered [e, de/dtheta0, de/dtheta1, ...], matching the concatenate the dense
        path used to build, so the flattened layout the M-inverse expects is unchanged.
        """
        with cp.cuda.Device(ggm.get_instance().default_gpu_id):
            if derivative_neighbour_terms is None:
                vecs = error_neighbour_terms[:, :, None]
            else:
                vecs = cp.stack([error_neighbour_terms, *derivative_neighbour_terms], axis=2)

        return vecs if isResultOnGPU else cp.asnumpy(vecs)
