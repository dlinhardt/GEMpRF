# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Siddharth Mittal",
"@Desc    :   None",     
"""

import math
import os
import sys
from typing import List
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from enum import Enum

from gem.model.selected_prf_model import SelectedPRFModel
from gem.model.prf_stimulus import Stimulus
from gem.model.prf_model import PRFModel
from gem.space.PRFSpace import PRFSpace
from gem.model.prf_model import GaussianModelParams
from gem.model.prf_model import DoGModelParams
from gem.utils.hpc_cupy_utils import HpcUtils as gpu_utils
from gem.utils.gem_write_to_file import GemWriteToFile
from gem.utils.logger import Logger
from gem.utils.gem_gpu_manager import GemGpuManager as ggm

DEBUG = False

class SignalSynthesizer:
    @classmethod
    def __set_kernel_config(self, num_xThreads : int, num_yThreads : int, num_zThreads : int):
        block_dim = (512, 1, 1)
        bx = int((num_xThreads + block_dim[0] - 1) / block_dim[0])
        by = int((num_yThreads + block_dim[1] - 1) / block_dim[1])
        bz = int((num_zThreads + block_dim[2] - 1) / block_dim[2])
        grid_dim = (bx, by, bz)
        return block_dim, grid_dim

    """Compute the model curves (Gaussian curves, DoG curves, etc.) based on the selected PRF model"""
    @classmethod
    def compute_signals_on_gpu(cls,
                               model_type: SelectedPRFModel,
                               stimulus_data_selected_gpu : cp.ndarray,
                               stimulus_x_range : cp.ndarray,
                               stimulus_y_range : cp.ndarray,
                               stimulus_height : int,
                               stimulus_width : int,
                               cuda_kernel: cp.RawKernel,
                               multi_dim_points_gpu: cp.ndarray,
                               num_dimension: int) -> cp.ndarray:
        # one curve for each point
        num_model_curves = len(multi_dim_points_gpu)
        # these are just the model curves (e.g. gaussian, DoG etc.)
        result_model_curves_gpu = cp.zeros((num_model_curves * stimulus_height * stimulus_width), dtype=cp.float64)

        # we may have more than 3 dimensions, we need to flatten out data, each cuda thread responsible to compute a model curve for a single point
        # sample points data [(0, 0, 1, 2), (-9, -9, 8, 5).....]
        flattened_points_gpu = multi_dim_points_gpu.ravel()

        # call the CUDA kernel
        block_dim, grid_dim = SignalSynthesizer.__set_kernel_config(num_xThreads=num_model_curves, num_yThreads=1, num_zThreads=1)
        cuda_kernel(grid_dim, block_dim,
                    (result_model_curves_gpu,
                     flattened_points_gpu,
                     stimulus_x_range,
                     stimulus_y_range,
                     num_dimension,
                     stimulus_height,
                     stimulus_width,
                     num_model_curves))

        # each row contains a flat model curve for a single point (e.g. gaussian curve for a single point)
        model_curves_rm_gpu = result_model_curves_gpu.reshape(
            (num_model_curves, stimulus_height * stimulus_width))
        # stimulus data is column major
        signal_rowmajor_gpu = cp.dot(model_curves_rm_gpu, stimulus_data_selected_gpu)

        if (DEBUG):
            if (True):                
                # plot the model curves
                #...plot pRF
                gc = cp.asnumpy(model_curves_rm_gpu)
                center_gc_idx = int(num_model_curves // 2) #...if you want center pRF
                gc_idx = 20
                pRF = (cp.asnumpy(gc[gc_idx, :])).reshape((101, 101))
                maxEcc_dummy = 9.0 # 13.5
                fig, ax = plt.subplots()
                ax.imshow(pRF, cmap='hot', origin='lower', extent=(-maxEcc_dummy, maxEcc_dummy, -maxEcc_dummy, maxEcc_dummy))
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('pRF')
                ax.set_xticks([-maxEcc_dummy, 0, maxEcc_dummy]) # NOTE: hardcorded value for plot, maxEcc_dummy = 13.5
                ax.set_yticks([-maxEcc_dummy, 0, maxEcc_dummy])
                plt.show()

                # plot the signal
                plt.figure()
                plt.plot(cp.asnumpy(signal_rowmajor_gpu[center_gc_idx, :]))
                Logger.print_red_message(
                    message="GEM-WARNING: Turn off the debug flag.")

        return signal_rowmajor_gpu
    
    @classmethod
    def get_stimulus_data_on_selected_gpu(cls, stimulus : Stimulus, selected_device_id : int):
        # transfer stimulus data on the selected device, if the selecte device is not 0
        if(ggm.get_instance().default_gpu_id == selected_device_id):
            return stimulus.stimulus_data_gpu, stimulus.x_range_gpu, stimulus.y_range_gpu

        # NOTE: cache the copy per device. This used to re-upload the whole stimulus on every call,
        # and the refined-signal synthesis calls it twice per Y-batch, so the transferred volume grew
        # with the batch count -- hundreds of GB per hemisphere, and worse the smaller the batches.
        # The cache lives on the stimulus object so it is released with it, and each concatenated
        # task keeps its own.
        per_device_cache = getattr(stimulus, "_per_device_gpu_data", None)
        if per_device_cache is None:
            per_device_cache = {}
            stimulus._per_device_gpu_data = per_device_cache

        cached = per_device_cache.get(selected_device_id)
        if cached is None:
            with cp.cuda.Device(selected_device_id):
                cached = (cp.asarray(stimulus.stimulus_data_cpu),
                          cp.asarray(stimulus.x_range_cpu),
                          cp.asarray(stimulus.y_range_cpu))
            per_device_cache[selected_device_id] = cached

        return cached

    @classmethod
    def downsample_frame_indices(cls, stimulus : Stimulus, stim_frames : int):
        """Frame indices kept when a high-temporal-resolution stimulus is downsampled, or None.

        They depend only on the stimulus -- not on the signals, and not on which chunk a signal was
        synthesized in -- so they can be computed once up front and applied to each finished chunk.
        """
        if not stimulus.HighTemporalResolutionEnabled:
            return None

        if stim_frames < stimulus.NumFramesDownsampled: # i.e. if the number of frames in stimulus is less than the downsampled length
            Logger.print_red_message(f"Number of frames in provided stimulus ({stim_frames}) is less than the specified downsampled length ({stimulus.NumFramesDownsampled}).\
                                     \nPlease check your config file (high_temporal_resolution) and/or stimulus.", print_file_name=False)
            sys.exit(1)

        idx = np.linspace(0, stim_frames, stimulus.NumFramesDownsampled, endpoint=False, dtype=int)
        slice_time_ref_adjusted_step_size = (np.diff(idx).mean() * stimulus.SliceTimeRef).round().astype(int)
        return (idx + slice_time_ref_adjusted_step_size).astype(int)

    @classmethod
    def get_available_gpus(cls, total_model_signals, cfg):
        if( total_model_signals > 1) & (gpu_utils.get_number_of_gpus() > 1):
            # NOTE: CUDA_VISIBLE_DEVICES need not be set -- CuPy then sees every device. Reading it
            # unguarded raised KeyError on a multi-GPU box that was simply started without it.
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices:
                available_gpus = list(map(int, visible_devices.split(',')))
            else:
                available_gpus = list(range(gpu_utils.get_number_of_gpus()))
            num_gpus = len(available_gpus)
        else:
            # we don't need to use multiple GPUs for a single signal
            num_gpus = 1
            available_gpus = [0]

        return available_gpus, num_gpus



    @classmethod
    # def compute_signals_batches(cls, prf_space : PRFSpace, points_indices_mask : np.ndarray,  prf_model : PRFModel, stimulus : Stimulus, derivative_wrt : Enum, orthonormalized_model_signals : cp.ndarray = None):  # NOTE: use Enum for derivative_wrt
    def compute_signals_batches(cls, prf_multi_dim_points_cpu : np.ndarray, points_indices_mask : np.ndarray,  prf_model : PRFModel, stimulus : Stimulus, derivative_wrt : Enum, orthonormalized_model_signals : cp.ndarray = None, cfg = None):  # NOTE: use Enum for derivative_wrt
        # results batches
        result_batches = [] 

        # select the points for which the signals needs to be computed: apply the mask to the points
        if points_indices_mask is None:
            multi_dim_points_cpu = prf_multi_dim_points_cpu # prf_space.multi_dim_points_cpu
        else:
            multi_dim_points_cpu = prf_multi_dim_points_cpu[points_indices_mask] # prf_space.multi_dim_points_cpu[points_indices_mask]

        stim_width, stim_height, stim_frames = stimulus.data_shape()

        single_timecourse_length = stim_frames
        model_curve_length = stim_height * stim_width

        # Every model signal is synthesized across the stimulus' full frame count -- the whole point
        # of high_temporal_resolution is that the HRF-convolved stimulus and the model timecourses
        # are formed on the fine time grid -- and only the finished signal is thinned out to the
        # frames the scanner sampled.
        # NOTE: that thinning now happens per chunk instead of once the whole per-GPU buffer had been
        # filled. A chunk holds complete signals, so it is the same operation on the same finished
        # timecourses; what changes is that only one chunk, rather than the entire buffer, is ever
        # held at the full frame count. On a single GPU the buffer is the whole grid: 942k signals x
        # 2400 frames = 18.1 GB had to be alive to produce 2.9 GB of downsampled signals.
        downsampled_frame_indices = cls.downsample_frame_indices(stimulus, stim_frames)
        output_timecourse_length = single_timecourse_length if downsampled_frame_indices is None else len(downsampled_frame_indices)

        # # total_signals = len(prf_space.multi_dim_points_cpu)             
        total_signals = len(prf_multi_dim_points_cpu)             
        per_signal_and_model_curve_required_mem_gb = ((single_timecourse_length + model_curve_length) * 8) / (1024 ** 3) # 8 bytes per float64
        
        if (derivative_wrt.value == -1): 
            cuda_kernel = prf_model.model_curves_kernel
        else:
            cuda_kernel = prf_model.derivatives_kernels_list[derivative_wrt.value]

        available_gpus, num_gpus = SignalSynthesizer.get_available_gpus(total_signals, cfg)

        per_gpu_assigned_batch_size = np.max([1, int(total_signals / num_gpus) + 1]) # NOTE: "+1" to deal with the case where the division is fractional

        num_signals_computed = 0
        for gpu_idx in range(num_gpus):
            signal_rowmajor_batch_current_gpu = None              

            selected_gpu_id = gpu_idx #available_gpus[gpu_idx]
            with cp.cuda.Device(selected_gpu_id):  
                # get stimulus data on the selected GPU
                stimulus_data, stimulus_x_range, stimulus_y_range = SignalSynthesizer.get_stimulus_data_on_selected_gpu(stimulus = stimulus, selected_device_id = selected_gpu_id)                    
                signal_rowmajor_batch_current_gpu = cp.zeros((min(per_gpu_assigned_batch_size, total_signals - num_signals_computed), output_timecourse_length), dtype=cp.float64)
                                            
                available_bytes = gpu_utils.device_available_mem_bytes(device_id=selected_gpu_id)                                
                possible_batch_size = int((available_bytes / (1024 ** 3)) / (per_signal_and_model_curve_required_mem_gb * 1.2)) # additional "1.2" because, we will also be needing memory for the signal timecourses so, better be safe !!!

                if possible_batch_size < 1 and num_gpus > 1: # if we have only 1 GPU, then usually the unified memory is used, so we can still compute the signals
                    raise ValueError(f"Not enough GPU memory available on device-{selected_gpu_id}.\nAvailable (Gigabytes) = {available_bytes / (1024 ** 3)}, Required (Gigabytes) = {per_signal_and_model_curve_required_mem_gb * 1.2}.\nGEM-pRF cannot compute further model signals !!!")

                selected_gpu_possible_signals_chunk_size = per_gpu_assigned_batch_size if per_gpu_assigned_batch_size <= possible_batch_size else possible_batch_size
                selected_gpu_possible_signals_chunk_size = min(15000, selected_gpu_possible_signals_chunk_size) # Dirty fix, to avoid illegal memory access error, if the selected_gpu_possible_signals_chunk_size is too large
                if selected_gpu_possible_signals_chunk_size < 1 and num_gpus == 1:
                    # NOTE: the single-GPU fallback relies on unified memory, so it does not fail --
                    # it silently drops to one kernel launch per signal, which for a dense grid means
                    # the run never finishes. Say so rather than letting it look like a hang.
                    Logger.print_red_message(f"Only {available_bytes / (1024 ** 3):.2f} GB free on device-{selected_gpu_id}, which is not enough for even one signal chunk "
                                             f"({per_signal_and_model_curve_required_mem_gb * 1.2 * 1024:.1f} MB each). Falling back to one signal at a time via unified memory; "
                                             f"expect this to be unusably slow for {total_signals} signals. Free GPU memory or use more GPUs.", print_file_name=False)
                    selected_gpu_possible_signals_chunk_size = 1

                for chunk_idx in range(0, per_gpu_assigned_batch_size, selected_gpu_possible_signals_chunk_size): # (0, total_sigma, batch_size)     
                    inprocess_signal_startidx = chunk_idx
                    if num_gpus == 1:
                        inprocess_signal_endidx = min(inprocess_signal_startidx + selected_gpu_possible_signals_chunk_size, total_signals)
                        signals_in_chunk_indices = np.arange(inprocess_signal_startidx, inprocess_signal_endidx, 1)
                        full_range_signals_in_chunk_indices = signals_in_chunk_indices
                    else: # i.e. in case of multiple GPUs
                        inprocess_signal_endidx = min(inprocess_signal_startidx + selected_gpu_possible_signals_chunk_size, per_gpu_assigned_batch_size) #NOTE: change here
                        if ((gpu_idx * per_gpu_assigned_batch_size) + inprocess_signal_endidx) > total_signals:
                            inprocess_signal_endidx = (total_signals - (gpu_idx * per_gpu_assigned_batch_size))
                        signals_in_chunk_indices = np.arange(inprocess_signal_startidx, inprocess_signal_endidx, 1)
                        full_range_signals_in_chunk_indices = (gpu_idx * per_gpu_assigned_batch_size) + signals_in_chunk_indices

                    chunk_signal_rowmajor_batch_gpu = SignalSynthesizer.compute_signals_on_gpu(
                        model_type=prf_model.model_type,
                        stimulus_data_selected_gpu=stimulus_data,
                        stimulus_x_range=stimulus_x_range,
                        stimulus_y_range=stimulus_y_range,
                        stimulus_height=stim_height,
                        stimulus_width=stim_width,
                        cuda_kernel=cuda_kernel,                        
                        multi_dim_points_gpu=cp.asarray(multi_dim_points_cpu[full_range_signals_in_chunk_indices]),
                        num_dimension=prf_model.num_dimensions)
                    
                    # the signal is complete at this point: downsample the finished timecourses
                    if downsampled_frame_indices is not None:
                        chunk_signal_rowmajor_batch_gpu = chunk_signal_rowmajor_batch_gpu[:, downsampled_frame_indices] # plt.plot(cp.asnumpy(signal_rowmajor_batch_current_gpu[0])[0, :])

                    # update to batch
                    try:
                        signal_rowmajor_batch_current_gpu[signals_in_chunk_indices, :] = chunk_signal_rowmajor_batch_gpu
                    except Exception as e:
                        print(f"{str(e)}.\nTry to reduce chunk size per GPU or contact the author siddharth.mittal@meduniwien.ac.at")
                        raise e

                    # the full-resolution chunk is the transient this whole loop is sized around;
                    # drop it before the next one is allocated
                    del chunk_signal_rowmajor_batch_gpu

                result_batches.append(signal_rowmajor_batch_current_gpu)
                num_signals_computed += len(signal_rowmajor_batch_current_gpu)

        return result_batches

    @classmethod
    def orthonormalize_modelled_signals(cls, O_gpu : cp.ndarray, model_signals_rm_batches : List[cp.ndarray], dS_dtheta_rm_batches_list : List[List[cp.ndarray]], release_inputs : bool = False) -> cp.ndarray:
        """Orthonormalize the model signals and their derivatives, batch by batch.

        ``release_inputs=True`` drops each raw batch as soon as its orthonormalized counterpart
        exists, by clearing the caller's list entry. Without it both the raw and the orthonormalized
        set are fully resident at the end -- eight copies of the grid with three derivatives, which
        on a single GPU is the largest allocation of the run. Only pass it when the raw batches are
        genuinely not needed afterwards (they are, for instance, still written out in debug mode).
        """
        # initialize
        #...S_prime_batches
        S_prime_batches = []
        #...the orthonormalized derivatives signals list
        if dS_dtheta_rm_batches_list is not None:
            num_theta_params = len(dS_dtheta_rm_batches_list)
            orthonormalized_derivatives_signals_batches_list = None if num_theta_params == 0 else [None] * num_theta_params

        # orthonormalize: model signal
        for batch_idx in range(len(model_signals_rm_batches)):
            gpu_device_id = model_signals_rm_batches[batch_idx].device.id
            with cp.cuda.Device(gpu_device_id):  
                O_gpu_current_device = O_gpu if gpu_device_id == ggm.get_instance().default_gpu_id else cp.array(O_gpu)
                S_star_columnmajor_gpu = cp.dot(O_gpu_current_device, model_signals_rm_batches[batch_idx].T)
                if release_inputs:
                    model_signals_rm_batches[batch_idx] = None # the raw batch is not read again below

                # need a take care the number of signals we have in the batch
                if S_star_columnmajor_gpu.shape[1] > 1: # i.e. we have more than one signal
                    S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2) # single row vector: basically this is (s*.T @ s*) part but for all the signals, which is actually the square of a matrix and then summing up all the rows of a column (because our signals are along columns)
                else: # i.e. we have only one signal
                    S_star_S_star_invroot_gpu = ((S_star_columnmajor_gpu ** 2).sum()) ** (-1/2)
                
                S_prime_columnmajor_gpu = S_star_columnmajor_gpu * S_star_S_star_invroot_gpu # normalized, orthogonalized Signals        

                # NOTE: due to numerical instability, some signals do not normalize to 1, and have values much greater than 1. We need to set all values of these signals to infinity
                # ...find columns in S_prime_columnmajor_gpu, whose Norm is not close to 1
                norm = ((S_prime_columnmajor_gpu ** 2).sum(axis=0)) ** (-1/2)
                max_values_idx = cp.where(~cp.isclose(norm, 1.0))[0] # max_values_idx = cp.where(max_values > 1.001)[0]                                  
                S_prime_columnmajor_gpu[:, max_values_idx] = cp.full((S_prime_columnmajor_gpu.shape[0], len(max_values_idx)), cp.inf) # ...set these columns to infinity

                # append results     
                S_prime_batches.append(S_prime_columnmajor_gpu)


                # orthogonalize: derivative signals                
                for theta in range(num_theta_params):
                    dS_star_dtheta_batch_cm_gpu = cp.dot(O_gpu_current_device, (dS_dtheta_rm_batches_list[theta])[batch_idx].T)
                    if release_inputs:
                        (dS_dtheta_rm_batches_list[theta])[batch_idx] = None
                    # NOTE: this is the same expression as before,
                    #   dS'/dtheta = dS*/dtheta * n - (S* * n^3) * (S* . dS*/dtheta),
                    # only evaluated so that no (num_timepoints, num_signals) array is allocated that
                    # is not part of the answer. Written as one line it built four of them, two of
                    # which existed only to be consumed by the very next operator; those two are now
                    # folded into their predecessor in place. Each product is still formed in the
                    # same order between the same operands, so every value is rounded exactly as
                    # before -- what drops is the peak, from four such arrays to two.
                    dS_prime_dtheta_batch_cm_gpu = dS_star_dtheta_batch_cm_gpu * S_star_S_star_invroot_gpu

                    # (S* . dS*/dtheta) per column. dS*/dtheta is not read again after this, so it
                    # doubles as the scratch space for the elementwise product that gets summed away
                    # -- that product used to be a full array of its own.
                    dS_star_dtheta_batch_cm_gpu *= S_star_columnmajor_gpu
                    projection_gpu = dS_star_dtheta_batch_cm_gpu.sum(axis=0)
                    del dS_star_dtheta_batch_cm_gpu

                    correction_gpu = S_star_columnmajor_gpu * (S_star_S_star_invroot_gpu ** 3)
                    correction_gpu *= projection_gpu # in place: same operands, same order, one array instead of two
                    dS_prime_dtheta_batch_cm_gpu -= correction_gpu
                    del correction_gpu

                    if orthonormalized_derivatives_signals_batches_list[theta] is None:
                        orthonormalized_derivatives_signals_batches_list[theta] = [dS_prime_dtheta_batch_cm_gpu]
                    else:
                        (orthonormalized_derivatives_signals_batches_list[theta]).append(dS_prime_dtheta_batch_cm_gpu) # column major

        return S_prime_batches, orthonormalized_derivatives_signals_batches_list