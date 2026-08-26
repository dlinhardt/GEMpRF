# -*- coding: utf-8 -*-
"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Medical University of Vienna",
"@Desc    :   None",
        
"""
import math
import shutil
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import cupy as cp
import threading
import queue
import os
import datetime
import cProfile
import pstats
import pandas as pd
import time
import datetime

# gem
from gem.model.prf_model import PRFModel
from gem.model.prf_model import GaussianModelParams
from gem.model.prf_gaussian_model import PRFGaussianModel
from gem.space.PRFSpace import PRFSpace
from gem.fitting.hpc_grid_fit import GridFit
from gem.fitting.hpc_refine_fit import RefineFit
from gem.analysis.prf_analysis import PRFAnalysis
from gem.space.coefficient_matrix import CoefficientMatix
from gem.utils.hpc_cupy_utils import HpcUtils as gpu_utils
from gem.model.selected_prf_model import SelectedPRFModel
from gem.signals.signal_synthesizer import SignalSynthesizer
from gem.model.prf_stimulus import Stimulus
from gem.utils.logger import Logger
from gem.utils.gem_write_to_file import GemWriteToFile
from gem.analysis.prf_r2_variance_explain import R2
from gem.data.observed_data import ObservedData, DataSource
from gem.signals.orthogonalization_matrix import OrthoMatrix
from gem.data.bids_handler import GemBidsHandler
from gem.data.gem_bids_concatenation_data_info import BidsConcatenationDataInfo
from gem.data.gem_stimulus_file_info import StimulusFileInfo
from gem.tools.json_file_operations import JsonMgr
from gem.tools.result_file_writer import ResultFileWriter
from gem.utils.gem_gpu_manager import GemGpuManager as ggm
from gem.utils.logger import Logger
from gem.utils.gem_errors import TimepointMismatchError, InputFileMissingError
from gem.utils.gem_h5_file_handler import H5FileManager
from gem.signals.hrf_generator import spm_hrf_compat

class GEMpRFAnalysis:
    __selected_prf_model = SelectedPRFModel.NoneType

    # A refined fit is not trusted (and the vertex reverts to its coarse grid point) once it moves
    # more than this many grid steps away from the grid point in x, y or sigma.
    MAX_GRID_STEPS_AWAY = 10

    # Bit values for the per-vertex rejection reason stored in the h5 /grid_fallback group.
    # A vertex can trip several reasons at once, so the reasons are OR-ed into a single uint8.
    # This dict is the single source of truth: the writer emits its bit->name legend into the file.
    FALLBACK_REASON_BITS = {
        "worse_error":   1,   # refinement increased the error (individual-run path only)
        "nan_refined":   2,   # refined params came out NaN (degenerate solve)
        "x_too_far":     4,   # |refined - coarse| in x     > MAX_GRID_STEPS_AWAY * x_step
        "y_too_far":     8,   # |refined - coarse| in y     > MAX_GRID_STEPS_AWAY * y_step
        "sigma_too_far": 16,  # |refined - coarse| in sigma > MAX_GRID_STEPS_AWAY * sigma_step
        "zero_signal":   32,  # refined model timecourse all-zero -> R2 = -2 (individual-run path only)
    }

    @classmethod
    def get_hrf_curve(cls, cfg, stimulus : Stimulus):
        if cfg.optional_analysis_params['enable'] and cfg.optional_analysis_params['hrf']['use_from_file']:
            hrf_curve = H5FileManager.get_key_value(cfg.optional_analysis_params['filepath'], cfg.optional_analysis_params['hrf']['key'])
            if hrf_curve is None:
                Logger.print_red_message(f"Could not load HRF curve from file: {cfg.optional_analysis_params['filepath']} with key: {cfg.optional_analysis_params['hrf']['key']}", print_file_name=False)
                sys.exit(1)
        else:
            # get TR
            if cfg.default_hrf["TR"] is None:                
                TR = stimulus.Header['pixdim'][4]  # now we need to get TR from stimulus # assuming 4th dimension is time
                Logger.print_yellow_message(f"\nSetting HRF 't' step value to stimulus ({stimulus.StimulusTaskName}) TR: {TR:.3f} seconds.", print_file_name=False)
            else:
                TR = cfg.default_hrf["TR"]
            
            # spm "t" (start, stop, step/TR)
            t_start, t_stop = cfg.default_hrf["t"]
            hrf_t = (t_start, t_stop, TR)

            # generate HRF curve using SPM parameters
            hrf_params = (np.arange(*hrf_t),
                          cfg.default_hrf["peak_delay"], 
                          cfg.default_hrf["under_shoot_delay"], 
                          cfg.default_hrf["peak_disp"], 
                          cfg.default_hrf["under_disp"], 
                          cfg.default_hrf["peak_to_undershoot"], 
                          cfg.default_hrf["normalize"])
            hrf_curve = spm_hrf_compat(*hrf_params)
        return hrf_curve

    @classmethod
    def load_config(cls, config_filepath : str = None):        
        from gem.configs.config_manager import ConfigurationWrapper as cfg
        cfg.load_configuration(run_type=None, config_filepath=config_filepath)

        return cfg

    @classmethod
    def load_stimulus(cls, cfg, stimulus_info : StimulusFileInfo = None)-> Stimulus:
        # ...stimulus
        stim_width = int(cfg.stimulus["width"])
        stim_height = int(cfg.stimulus["height"])  
        binarize = True if cfg.stimulus["binarization"].get("@enable") == "True" else False
        binarize_threshold = float(cfg.stimulus["binarization"].get("@threshold"))
        high_temporal_resolution_info = cfg.stimulus_high_temporal_resolution if cfg.stimulus_high_temporal_resolution['enable'] else None

        stimulus = Stimulus(os.path.join(stimulus_info.stimulus_dir, 
                                         stimulus_info.stimulus_filename), 
                                         size_in_degrees=float(cfg.stimulus["visual_field"]), 
                                         stim_config = cfg.stimulus, 
                                         binarize=binarize, 
                                         binarize_threshold=binarize_threshold,
                                         high_temporal_resolution_info=high_temporal_resolution_info,
                                         stimulus_task_name=stimulus_info.stimulus_task)

        # get HRF curve
        hrf_curve = cls.get_hrf_curve(cfg, stimulus)
        GemWriteToFile.get_instance().write_array_to_h5(hrf_curve, variable_path=['hrf'], append_to_existing_variable=False)

        stimulus.compute_resample_stimulus_data((stim_height, stim_width, stimulus.org_data.shape[2])) #stimulus.org_data.shape[2]
        stimulus.compute_hrf_convolved_stimulus_data(hrf_curve=hrf_curve)
        GemWriteToFile.get_instance().write_array_to_h5(stimulus.resampled_data, variable_path=[f'stimulus', f'{stimulus_info.stimulus_task}', 'resampled_data'], append_to_existing_variable=False)
        GemWriteToFile.get_instance().write_array_to_h5(stimulus.stimulus_data_cpu, variable_path=[f'stimulus', f'{stimulus_info.stimulus_task}', 'stimulus_data_hrf_convolved'], append_to_existing_variable=False)

        return stimulus
        
    @classmethod
    def get_prf_spatial_points(cls, cfg)-> np.ndarray:
        search_space_xx = np.linspace(-float(cfg.default_spatial_grid["visual_field_radius"]), float(cfg.default_spatial_grid["visual_field_radius"]), int(cfg.default_spatial_grid["num_horizontal_prfs"])) # nCols
        search_space_yy = np.linspace(-float(cfg.default_spatial_grid["visual_field_radius"]), float(cfg.default_spatial_grid["visual_field_radius"]), int(cfg.default_spatial_grid["num_vertical_prfs"])) # nRows
        x_mesh, y_mesh = np.meshgrid(search_space_xx, search_space_yy) # NOTE: (col, row)
        spatial_points_xy = np.column_stack((y_mesh.ravel(), x_mesh.ravel())) # (col i.e. x, row i.e. y)        
        return spatial_points_xy
    
    @classmethod
    def get_additional_dimensions(cls, cfg, selected_prf_model : SelectedPRFModel):
        if selected_prf_model == SelectedPRFModel.GAUSSIAN:
            if cfg.optional_analysis_params['enable'] and cfg.optional_analysis_params['sigmas']['use_from_file']: # Get user defined custom values for sigma from H5 file
                search_space_sigma_range = H5FileManager.get_key_value(filepath=cfg.optional_analysis_params['filepath'], key = cfg.optional_analysis_params['sigmas']['key'])
                if search_space_sigma_range is None:
                    Logger.print_red_message(f"Could not load sigma range from file: {cfg.optional_analysis_params['filepath']} with key: {cfg.optional_analysis_params['sigmas']['key']}", print_file_name=False)
                    sys.exit(1)
            else: # if user hasn't specifed anything, use default values
                search_space_sigma_range = np.linspace(float(cfg.default_sigmas['min_sigma']), float(cfg.default_sigmas['max_sigma']), int(cfg.default_sigmas['num_sigmas'])) # 0.5 to 1.5

            additional_dimensions = PRFSpace.make_extra_dimensions(search_space_sigma_range)
        else:
            raise ValueError("Invalid PRF Model")

        return additional_dimensions

    @classmethod
    def execute_Grids2MpInv_NewMethod(cls, prf_space : PRFSpace, result_queue):
        # neighbour search (the kdtree.query / filter-loop split is timed inside PRFSpace)
        _t_neigh_start = time.time()
        prf_space.compute_multidim_points_neighbours()
        _num_points = len(prf_space.multi_dim_points_cpu)
        Logger.print_timing_message(f"compute_multidim_points_neighbours ({_num_points} points): {datetime.timedelta(seconds=time.time() - _t_neigh_start)}")

        # pinv loop over all grid points
        _t_pinv_start = time.time()
        arr_2d_location_inv_M = CoefficientMatix.Wrapper_Grids2MpInv_numba(prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_vf_neighbours)
        Logger.print_timing_message(f"Wrapper_Grids2MpInv_numba pinv loop ({_num_points} points): {datetime.timedelta(seconds=time.time() - _t_pinv_start)}")

        result_queue.put(arr_2d_location_inv_M)

    @classmethod
    def get_selected_prf_model(cls, cfg):
        if cfg.pRF_model_details['model'] == "2d_gaussian":
            cls.__selected_prf_model = SelectedPRFModel.GAUSSIAN
        else:
            raise ValueError("Invalid PRF Model")

        return cls.__selected_prf_model

    @classmethod    
    def compute_orthonormalized_signals(cls, O_gpu, prf_space : PRFSpace, prf_model : PRFModel, stimulus : Stimulus, cfg, stimulus_task_name : str = None):
        # model signals
        S_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE, cfg=cfg)

        subcat = f"{stimulus_task_name}" if stimulus_task_name is not None else ""

        # model derivatives signals
        dS_dtheta_batches_list = []
        if cfg.refine_fitting_enabled:
            num_theta = prf_model.num_dimensions
            for theta_idx in range(num_theta):
                dS_dtheta_batches = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=prf_space.multi_dim_points_cpu, points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams(theta_idx), cfg=cfg)
                dS_dtheta_batches_list.append(dS_dtheta_batches)
                GemWriteToFile.get_instance().write_array_to_h5(dS_dtheta_batches, variable_path=[f'model', f'{subcat}', f'model_signals_derivative_d{theta_idx}'], append_to_existing_variable=False)

        # Write debug info for the raw signals BEFORE orthonormalizing: the orthonormalization is
        # allowed to release each raw batch as it consumes it (see release_inputs), which otherwise
        # keeps the raw and the orthonormalized set alive side by side -- with three derivatives that
        # is eight copies of the whole grid, and on a single GPU they all sit on the same card. The
        # derivative batches are already written out per theta in the loop above.
        GemWriteToFile.get_instance().write_array_to_h5(S_batches, variable_path=[f'model', f'{subcat}', 'model_signals'], append_to_existing_variable=False)

        # Orthonormalized model + derivatives signals
        orthonormalized_S_cm_gpu_batches, orthonormalized_dervatives_signals_batches_list = SignalSynthesizer.orthonormalize_modelled_signals(O_gpu=O_gpu,
                                                                                                                                        model_signals_rm_batches= S_batches,
                                                                                                                                        dS_dtheta_rm_batches_list = dS_dtheta_batches_list,
                                                                                                                                        release_inputs = True)
        GemWriteToFile.get_instance().write_array_to_h5(orthonormalized_S_cm_gpu_batches, variable_path=[f'model', f'{subcat}', 'orthonormalized_model_signals'], append_to_existing_variable=False)
        if orthonormalized_dervatives_signals_batches_list is not None:
            for theta_idx in range(len(orthonormalized_dervatives_signals_batches_list)):
                GemWriteToFile.get_instance().write_array_to_h5(orthonormalized_dervatives_signals_batches_list[theta_idx], variable_path=[f'model', f'{subcat}', f'orthonormalized_model_signals_derivative_d{theta_idx}'], append_to_existing_variable=False)
        return orthonormalized_S_cm_gpu_batches, orthonormalized_dervatives_signals_batches_list    

    @classmethod
    def get_single_run_data_files_info(cls, cfg):
        # List of input measured data filepaths
        measured_data_list = None
        if cfg.bids['@enable'] == "True":
            measured_data_info_list = GemBidsHandler.get_input_filepaths(bids_config=cfg.bids, stimuli_dir_path= cfg.stimulus['directory'])
            measured_data_list = (lambda x: np.array(x)[:, 0] if x else np.array([]))(measured_data_info_list)  # extract only filepaths from the list of tuples
        else:
            measured_data_list = cfg.fixed_paths['measured_data_filepath']['filepath']
            if isinstance(measured_data_list, str):
                measured_data_list = [measured_data_list] # Ensure it is always an array
            
        # List of result filepaths
        result_filepaths_list = []
        for data_idx in range(len(measured_data_list)):
            if cfg.bids['@enable'] == "True":
                result_file = GemBidsHandler.inputpath2resultpath(cfg.bids, measured_data_info_list[data_idx], analysis_id=cfg.bids["results_anaylsis_id"]["#text"], output_format=(cfg.results or {}).get('output_format', 'hdf5'))
            else:
                file = os.path.basename(measured_data_list[data_idx])
                filename = (file.split("."))[0]
                filename = (str(str(datetime.date.today()) + '_') if cfg.results['prepend_date'] == "True" else '') + filename
                result_base_path = cfg.results['basepath']                
                custom_postfix = cfg.results['custom_filename_postfix'] if cfg.results['custom_filename_postfix'] is not None else ""
                ext = '.h5' if (cfg.results or {}).get('output_format', 'hdf5') in ('hdf5', 'h5') else '.json'
                result_file = os.path.join(result_base_path, filename.replace("bold", "estimates") + custom_postfix + ext)
            
            # append to list
            result_filepaths_list.append(result_file)
        
        return measured_data_list, result_filepaths_list

    @classmethod
    def get_concatenated_runs_data_files_info(cls, cfg):
        # List of input measured data filepaths
        if cfg.bids['@enable'] == "True":
            measured_data_info_list = GemBidsHandler.get_input_filepaths(bids_config=cfg.bids, stimuli_dir_path= cfg.stimulus['directory'])
        else:
            raise ValueError("Invalid Configuration: Concatenation runs are only supported for BIDS data")
            
        # compute result filepaths for the concatenated items
        num_specified_concatenated_items = len(cfg.bids.get("concatenated").get("concatenate_item"))
        num_found_concatenated_items = len(measured_data_info_list)
        if num_specified_concatenated_items != num_found_concatenated_items:
            raise ValueError(f"Number of specified concatenated items ({num_specified_concatenated_items}) does not match the number of found concatenated items ({num_found_concatenated_items})")

        # Making sure that each measured_data_info_list is sorted based on the filepath
        for sublist in measured_data_info_list:            
            sublist.sort(key=lambda x: x[0]) # Sort each sublist based on the filepath (sublist[i][0])

        required_concatenations_info = []
        for items_to_be_concatenated_info in zip(*measured_data_info_list):
            input_filepaths = [item[0] for item in items_to_be_concatenated_info]
            data_info_dictionaries_list = [item[1] for item in items_to_be_concatenated_info]    
            stimulus_info = [item[2] for item in items_to_be_concatenated_info]            
            # print(input_filepaths)
            # print(data_info_dictionaries_list)
            concatenation_result_info = BidsConcatenationDataInfo.compare_and_merge_data_info_dicts(data_info_dictionaries_list)
            concatenated_result_filename = GemBidsHandler.get_concatenated_result_filepath(cfg.bids, input_filepaths[0], concatenation_result_info, output_format=(cfg.results or {}).get('output_format', 'hdf5'))
            concatenation_data_info = BidsConcatenationDataInfo(input_filepaths, data_info_dictionaries_list, stimulus_info, concatenation_result_info, concatenated_result_filename)
            required_concatenations_info.append(concatenation_data_info)
        
        return required_concatenations_info

    @classmethod
    def get_refined_signals_cpu(cls, refined_prf_params_XY : np.ndarray, prf_model : PRFModel, stimulus : Stimulus, cfg):
        refined_S_batches_gpu = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=cp.asnumpy(refined_prf_params_XY), points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE, cfg=cfg)            
        
        refined_S_cpu = []
        # refined signal batches could be present on different GPUs
        for batch_idx in range(len(refined_S_batches_gpu)):
            device_id = refined_S_batches_gpu[batch_idx].device.id
            with cp.cuda.Device(device_id):
                refined_signal_batch_cpu = cp.asnumpy(refined_S_batches_gpu[batch_idx])
                refined_S_cpu.append(refined_signal_batch_cpu)
        
        refined_S_cpu = np.concatenate(refined_S_cpu, axis=0)
        # # refined_S_cpu = cp.asnumpy(cp.concatenate(refined_S_batches_gpu, axis=0))

        return refined_S_cpu

    @classmethod
    def get_valid_refined_data(cls, refined_matching_results_XY, Y_signals_gpu, O_gpu, prf_model, stimulus, coarse_e,  best_fit_proj , coarse_pRF_estimations, cfg, grid_steps):
        # refined S batches
        refined_S_batches_gpu = SignalSynthesizer.compute_signals_batches(prf_multi_dim_points_cpu=cp.asnumpy(refined_matching_results_XY), points_indices_mask=None, prf_model=prf_model, stimulus=stimulus, derivative_wrt=GaussianModelParams.NONE, cfg=cfg)

        # refined S' batches
        orthonormalized_S_cm_gpu_batches, _ = SignalSynthesizer.orthonormalize_modelled_signals(O_gpu=O_gpu,
                                                                                                model_signals_rm_batches=refined_S_batches_gpu,
                                                                                                dS_dtheta_rm_batches_list=[])
        # refined error: only the matched pair y_i . s'_i is needed, so take the per-signal projection
        # directly instead of forming the full (num_Y_signals, num_Y_signals) product and reading its
        # diagonal (that product grew quadratically with the batch size for a linear amount of data)
        refined_error_vector = cp.asnumpy(GridFit.compute_matched_error_terms(Y_signals_gpu, orthonormalized_S_cm_gpu_batches))

        # ...get the locations where the errors are getting worse (ideally (refined - coarse) should be >0)
        coarse_error_vector = cp.asnumpy(coarse_e[cp.arange(coarse_e.shape[0]), cp.asarray(best_fit_proj)])
        diff = refined_error_vector - coarse_error_vector
        worse_error_mask = (~np.isnan(diff)) & (diff < 0)  # unchanged expression / behaviour

        # zero-signal reverts are handled implicitly (an all-zero refined timecourse means
        # the pRF drifted out of the aperture -> caught by the worse-error / too-far reasons, with
        # R2's own -2 sentinel as the final guard). We therefore do NOT scan the full signal matrix
        # on the device here (that temporary was doubling GPU memory in the hot batch loop); the
        # zero count for the report is read back cheaply from the R2 result instead.
        # returns (refined_XY, stats, records); callers use the records to build the h5 side table.
        return cls.apply_grid_fallback(refined_matching_results_XY, coarse_pRF_estimations, grid_steps,
                                       worse_error_mask=worse_error_mask)

    @classmethod
    def apply_grid_fallback(cls, refined_XY, coarse_XY, grid_steps, worse_error_mask=None, zero_mask=None):
        """
        Revert a vertex completely to its coarse grid point whenever the refined fit is not
        trusted, and return ``(refined_XY, stats, records)``.

        Reasons (all evaluated on the ORIGINAL refined values, so the revert set is a strict
        superset of the historical "worse error" revert -- untriggered vertices are unchanged):
          * worse_error   -- refinement increased the error   (only when a mask is provided)
          * nan_refined   -- refined params are NaN            (degenerate solve, was R2 = -1)
          * x/y/sigma too far -- refined value moved > MAX_GRID_STEPS_AWAY grid steps from the point
          * zero_signal   -- refined model timecourse is all zeros (only when a mask is provided,
                             was R2 = -2)

        ``records`` is a small dict carrying the per-vertex rejection detail for the h5
        ``/grid_fallback`` group (the callers add the batch offset + the zero-signal reason, then
        select the flagged vertices):
          * ``reason``      -- uint8 per-point bitmask (see ``FALLBACK_REASON_BITS``)
          * ``refined_pre`` -- float32 copy of the ORIGINAL (pre-revert) refined params
        """
        bits = cls.FALLBACK_REASON_BITS
        # Force a real copy so the numpy path does not alias refined_XY (overwritten in place below).
        refined_np = np.array(cp.asnumpy(refined_XY))
        coarse_np  = cp.asnumpy(coarse_XY)
        num_points = refined_np.shape[0]

        if worse_error_mask is None:
            worse_error_mask = np.zeros(num_points, dtype=bool)
        if zero_mask is None:
            zero_mask = np.zeros(num_points, dtype=bool)

        nan_mask = np.isnan(refined_np).any(axis=1)

        delta = np.abs(refined_np - coarse_np)
        too_far = delta > (cls.MAX_GRID_STEPS_AWAY * np.asarray(grid_steps))  # grid_steps: [x_step, y_step, sigma_step]
        too_far_any = too_far.any(axis=1)

        revert_mask = worse_error_mask | nan_mask | too_far_any | zero_mask

        # np.argwhere mirrors the historical indexed-assignment (cupy/numpy compatible).
        revert_indices = np.argwhere(revert_mask)
        refined_XY[revert_indices, :] = coarse_XY[revert_indices, :]

        stats = {
            "total": int(num_points),
            "worse_error": int(worse_error_mask.sum()),
            "nan_refined": int(nan_mask.sum()),
            "x_too_far": int(too_far[:, 0].sum()),
            "y_too_far": int(too_far[:, 1].sum()),
            "sigma_too_far": int(too_far[:, 2].sum()),
            "zero_signal": int(zero_mask.sum()),
            "on_grid": int(revert_mask.sum()),
        }

        # Per-point reason bitmask (cheap boolean-OR over the batch vertex axis).
        reason = np.zeros(num_points, dtype=np.uint8)
        reason[worse_error_mask] |= bits["worse_error"]
        reason[nan_mask]         |= bits["nan_refined"]
        reason[too_far[:, 0]]    |= bits["x_too_far"]
        reason[too_far[:, 1]]    |= bits["y_too_far"]
        reason[too_far[:, 2]]    |= bits["sigma_too_far"]
        reason[zero_mask]        |= bits["zero_signal"]

        records = {"reason": reason, "refined_pre": refined_np.astype(np.float32, copy=False)}

        return refined_XY, stats, records

    @classmethod
    def _select_fallback_records(cls, records, offset, r2_batch=None):
        """Pick out the rejected vertices from one batch's ``apply_grid_fallback`` records.

        Optionally OR in the zero-signal reason where the batch R2 hit its -2 sentinel (the
        individual-run path passes ``r2_batch``; the concatenated path does not compute it), then
        offset the batch-local indices by the batch's global start.

        Returns ``(global_index int32, reason uint8, refined_params float32[K, D])`` for the
        flagged vertices only (``K`` is usually a tiny fraction of the batch).
        """
        reason = records["reason"]
        if r2_batch is not None:
            zero_local = np.flatnonzero(np.asarray(r2_batch).ravel() == -2.0)
            if zero_local.size:
                reason = reason.copy()  # don't mutate the array owned by apply_grid_fallback
                reason[zero_local] |= cls.FALLBACK_REASON_BITS["zero_signal"]
        local = np.flatnonzero(reason)
        return ((offset + local).astype(np.int32), reason[local], records["refined_pre"][local])

    @classmethod
    def _finalize_fallback_records(cls, index_parts, reason_parts, refined_parts, num_params):
        """Concatenate the per-batch record pieces into the dict the h5 writer consumes.

        The reason legend + column names travel inside the dict so the writer stays decoupled from
        this (CuPy-importing) module and the h5 file is self-describing.
        """
        legend = {
            "reason_bits": dict(cls.FALLBACK_REASON_BITS),
            "param_columns": "Centerx0,Centery0,sigmaMajor",
        }
        if index_parts:
            records = {
                "vertex_index": np.concatenate(index_parts).astype(np.int32),
                "reason": np.concatenate(reason_parts).astype(np.uint8),
                "refined_params": np.concatenate(refined_parts, axis=0).astype(np.float32),
            }
        else:
            records = {
                "vertex_index": np.empty(0, dtype=np.int32),
                "reason": np.empty(0, dtype=np.uint8),
                "refined_params": np.empty((0, num_params), dtype=np.float32),
            }
        records.update(legend)
        return records

    # ---- Y-batch sizing ---------------------------------------------------------------------
    # Fraction of the measured free VRAM a Y-batch is allowed to claim. The estimate below only
    # models the two largest arrays, so this leaves room for the refinement and R2 work that follows.
    BATCH_MEMORY_SAFETY_FRACTION = 0.6

    @classmethod
    def _resolve_batches_setting(cls, cfg):
        """``<batches>N</batches>`` -> (N, False); ``<batches auto="true">N</batches>`` -> (N, True)."""
        raw = cfg.measured_data["batches"]
        if isinstance(raw, dict):  # xmltodict wraps the text when the element carries attributes
            return int(raw.get("#text", 1)), str(raw.get("@auto", "false")).lower() == "true"
        return int(raw), False

    @classmethod
    def get_y_batch_size(cls, cfg, total_y_signals, num_model_signals):
        """Vertices per Y-batch, sized against the free VRAM actually available.

        ``<batches>`` stays the upper bound by default, so a config that already fits keeps exactly the
        batch size -- and therefore exactly the results -- it has today; the measurement can only make
        batches *smaller*, which is what turns an out-of-memory crash into a slower run. With
        ``auto="true"`` the batch size follows the measurement in both directions, which is worth it
        when the configured value is far more conservative than the hardware needs.
        """
        num_batches, auto = cls._resolve_batches_setting(cfg)
        configured_batch_size = max(1, int(total_y_signals / max(1, num_batches)))

        try:
            num_gpus = max(1, int(gpu_utils.get_number_of_gpus()))
            free_bytes = gpu_utils.device_available_mem_bytes(device_id=ggm.get_instance().default_gpu_id)
        except Exception as exc:
            Logger.print_red_message(f"Could not measure free GPU memory ({exc}); using <batches> as given.",
                                     print_file_name=False)
            return configured_batch_size

        # Per vertex on the default device: the error matrix over the whole grid (accumulated across
        # concatenated runs, so one copy regardless of how many runs there are) plus the transient
        # chunk copied back from one of the other devices.
        bytes_per_vertex = num_model_signals * 8 * (1.0 + 1.0 / num_gpus)
        affordable_batch_size = int((free_bytes * cls.BATCH_MEMORY_SAFETY_FRACTION) / max(1.0, bytes_per_vertex))
        affordable_batch_size = max(1, min(affordable_batch_size, total_y_signals))

        batch_size = affordable_batch_size if auto else min(configured_batch_size, affordable_batch_size)

        if batch_size != configured_batch_size:
            direction = "raised" if batch_size > configured_batch_size else "reduced"
            Logger.print_green_message(
                f"Y-batch size {direction} from {configured_batch_size} to {batch_size} vertices "
                f"({free_bytes / 1024 ** 3:.1f} GB free on GPU {ggm.get_instance().default_gpu_id}, "
                f"{num_model_signals} model signals).", print_file_name=False)

        return batch_size

    @classmethod
    def get_pRF_estimations(cls, cfg, O_gpu, prf_space, prf_model, stimulus, prf_analysis, arr_2d_location_inv_M_cpu, measured_data_filepath):
        valid_refined_prf_points_XY = None
        r2_results = None

        # per-dimension grid spacing (cached on prf_space) and per-run fallback counters
        grid_steps = prf_space.get_grid_steps()
        grid_fallback_stats = {key: 0 for key in ("total", "worse_error", "nan_refined",
                                                  "x_too_far", "y_too_far", "sigma_too_far",
                                                  "zero_signal", "on_grid")}
        # per-vertex rejection detail for the h5 /grid_fallback group (only flagged vertices)
        fb_index_parts, fb_reason_parts, fb_refined_parts = [], [], []

         # y-signals
        y_data = ObservedData(data_source=DataSource.measured_data)
        Y_signals_cpu = y_data.get_y_signals(measured_data_filepath)

        # process bathches
        Y_signals_cpu = Y_signals_cpu[:, None] if Y_signals_cpu.ndim == 1 else Y_signals_cpu # in case only one signal is present

        # fail this analysis if number of timepoints in y-signals and stimulus do not match
        stimulus_num_frames = (stimulus.NumFrames, stimulus.NumFramesDownsampled)[stimulus.HighTemporalResolutionEnabled]
        if Y_signals_cpu.shape[0] != stimulus_num_frames:
            raise TimepointMismatchError(f"Number of timepoints in measured fMRI data ({Y_signals_cpu.shape[0]}) and stimulus ({stimulus_num_frames}) do not match for file: {measured_data_filepath}")

        total_y_signals = Y_signals_cpu.shape[1]
        batch_size = cls.get_y_batch_size(cfg, total_y_signals, len(prf_space.multi_dim_points_cpu))
        for current_batch_idx in range(0, total_y_signals, batch_size):
            Y_signals_batch_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=(Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size],))
            Y_signals_batch_cpu = Y_signals_cpu[:, current_batch_idx: current_batch_idx + batch_size]

            # error: grid search over every model signal. The derivative error terms are NOT computed
            # here -- they are only read at the neighbourhood of the winner, which is not known until
            # the argmax below has run (see GridFit).
            isResultOnGPU = ((cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled) | (not cfg.refine_fitting_enabled))
            best_fit_proj, e = GridFit.get_error_terms(isResultOnGPU=isResultOnGPU,
                                                       Y_signals_gpu=Y_signals_batch_gpu,
                                                       S_prime_cm_batches_gpu=prf_analysis.orthonormalized_S_batches)

            # Logger.print_green_message(f"error computed for batch {current_batch_idx} - {current_batch_idx + min(batch_size, total_y_signals-current_batch_idx) }...", print_file_name=False)

            # NOTE: RefineFit produces results in (X, Y) format
            # perform refine search, the obtained refined results will be in the (X, Y) format
            if cfg.refine_fitting_enabled:
                num_Y_signals = Y_signals_batch_cpu.shape[1]
                # evaluate e and de/dtheta straight into the (num_Y_signals, max_neighbours) shape the
                # refinement reads, instead of over the whole grid
                neighbour_columns = RefineFit.get_neighbour_columns(prf_space, best_fit_proj, arr_2d_location_inv_M_cpu)
                e_neighbour_terms = GridFit.gather_neighbour_terms(e, neighbour_columns)
                de_neighbour_terms = GridFit.accumulate_derivative_neighbour_terms(Y_signals_batch_gpu,
                                                                                   prf_analysis.orthonormalized_dS_dtheta_batches_list,
                                                                                   neighbour_columns)
                refine_input_vecs = GridFit.build_refine_input_vectors(e_neighbour_terms, de_neighbour_terms, isResultOnGPU)
                del e_neighbour_terms, de_neighbour_terms

                refined_matching_results_XY, Fex_results = RefineFit.get_refined_fit_results(prf_space,
                                                                                             num_Y_signals,
                                                                                             best_fit_proj,
                                                                                             arr_2d_location_inv_M_cpu,
                                                                                             refine_input_vecs,
                                                                                             neighbour_columns)
                del refine_input_vecs
        
            # NOTE: The coarse_estimation values are in XY format (i.e. (col, row) format)
            coarse_pRF_estimations = (prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_gpu)[isResultOnGPU][best_fit_proj]
            
            # validate if the refined pRF estimations are really improving the error value, and for the pRF points where error is getting worse, keep the coarse pRF estimations
            batch_fallback_records = None
            if cfg.refine_fitting_enabled:
                valid_refined_prf_points_XY_batch, batch_fallback_stats, batch_fallback_records = GEMpRFAnalysis.get_valid_refined_data(refined_matching_results_XY,
                                                                                          Y_signals_gpu=Y_signals_batch_gpu,
                                                                                          O_gpu=O_gpu,
                                                                                          prf_model=prf_model,
                                                                                          stimulus=stimulus,
                                                                                          coarse_e=e,
                                                                                          best_fit_proj=best_fit_proj,
                                                                                          coarse_pRF_estimations=coarse_pRF_estimations,
                                                                                          cfg=cfg,
                                                                                          grid_steps=grid_steps)
                for key in grid_fallback_stats:
                    grid_fallback_stats[key] += batch_fallback_stats[key]
            else:
                valid_refined_prf_points_XY_batch = coarse_pRF_estimations
                grid_fallback_stats["total"] += int(coarse_pRF_estimations.shape[0])

            # compute timecourses for refined pRF estimated params
            valid_refined_S_cpu_batch = GEMpRFAnalysis.get_refined_signals_cpu(valid_refined_prf_points_XY_batch, prf_model, stimulus, cfg)

            # compute Variance Explained
            r2_results_batch = R2.get_r2_num_den_method_with_epsilon_as_yTs(Y_signals_batch_gpu, O_gpu, valid_refined_prf_points_XY_batch, valid_refined_S_cpu_batch).reshape(-1, 1)

            # zero-signal vertices are flagged by R2 with the -2 sentinel (see prf_r2_variance_explain);
            # count them for the report without any extra signal scan.
            if cfg.refine_fitting_enabled:
                grid_fallback_stats["zero_signal"] += int(np.count_nonzero(r2_results_batch == -2.0))
                # per-vertex rejection detail: fold in the zero-signal reason (R2 == -2), offset the
                # batch-local indices to global vertex indices, keep only the flagged vertices.
                idx, rsn, ref = cls._select_fallback_records(batch_fallback_records, current_batch_idx, r2_batch=r2_results_batch)
                if idx.size:
                    fb_index_parts.append(idx)
                    fb_reason_parts.append(rsn)
                    fb_refined_parts.append(ref)

            # concatenate the batch results
            # NOTE: the refined timecourses are deliberately NOT accumulated. They are only needed to
            # compute this batch's R2 (just above); the one caller that wanted them across batches is
            # commented out in individual_run(), so growing a (num_vertices, num_frames) array by
            # repeated np.concatenate was several hundred MB of pure waste per file.
            if current_batch_idx == 0:
                valid_refined_prf_points_XY = valid_refined_prf_points_XY_batch
                r2_results = r2_results_batch
            else:
                valid_refined_prf_points_XY = np.concatenate((valid_refined_prf_points_XY, valid_refined_prf_points_XY_batch), axis = 0)                    
                r2_results = np.concatenate((r2_results, r2_results_batch), axis = 0)            

            # NOTE: release this batch's GPU arrays before the next iteration allocates its own. The
            # error matrix is (batch_size, num_model_signals) float64 -- 6.6 GiB at batch_size=1137 on
            # a 785k-point grid -- and `e` stayed bound across the loop boundary, so the next batch's
            # matrix was built while the previous one was still resident. That is a third full copy on
            # top of the two get_y_batch_size() budgets for, and it is what made <batches auto="true">
            # run out of memory: the sizing was right, the accounting was not.
            del e, best_fit_proj, Y_signals_batch_gpu

        grid_fallback_records = cls._finalize_fallback_records(fb_index_parts, fb_reason_parts, fb_refined_parts,
                                                               num_params=int(np.asarray(grid_steps).shape[0]))
        return valid_refined_prf_points_XY, r2_results, grid_fallback_stats, grid_fallback_records

    ##########################################################---------------------------------RUN---------------------------------################################################
    @classmethod
    def _filter_existing_results(cls, cfg, items, get_result_path, report):
        """Drop the items whose result already exists (overwrite_mode == "skip").

        The skipped items are collected in the report instead of being printed one by one.
        Returns the items that still have to be analysed.
        """
        if getattr(cfg, "overwrite_mode", "false") != "skip":
            return items

        kept = []
        for item in items:
            result_filepath = get_result_path(item)
            if ResultFileWriter.result_exists(result_filepath):
                report.add_skipped(result_filepath)
            else:
                kept.append(item)

        if report.num_skipped:
            Logger.print_green_message(f"Skipping {report.num_skipped} analysis(es) - results already exist (see run report for the list).", print_file_name=False)

        return kept

    @classmethod
    def concatenated_run(cls, cfg, prf_model, prf_space, report):
        # cfg = GEMpRFAnalysis.load_config(config_filepath=config_filepath) # load default config
        default_gpu_id = ggm.get_instance().default_gpu_id
        refinefit_on_gpu = cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled
        
        # data info
        required_concatenations_info = cls.get_concatenated_runs_data_files_info(cfg)

        # skip concatenations whose result already exists (in either hdf5 or json form)
        required_concatenations_info = cls._filter_existing_results(cfg, required_concatenations_info, lambda info: info.concatenation_result_filepath, report)
        if report.num_skipped and len(required_concatenations_info) == 0:
            Logger.print_green_message("All results already exist. Nothing to do.", print_file_name=False)
            return

        if len(required_concatenations_info) == 0:
            Logger.print_red_message("No data files found. Please check the specified paths in your XML configuration file. Aborting now...", print_file_name=False)
            return

        # NOTE: ----------------- COMMON VARIABLES
        arr_2d_location_inv_M_cpu = None

        # M-Matrix
        result_queue = queue.Queue()
        MpInv_thread = threading.Thread(target=cls.execute_Grids2MpInv_NewMethod, args=(prf_space, result_queue))
        MpInv_thread.start()
        mpinv_thread_start_time = datetime.datetime.now()

        # dictionary to hold all the stimulus-task specific data
        task_specific_data_dict = {}
        class TaskSpecificData:
            def __init__(self, stimulus, O_gpu, prf_analysis):
                self.stimulus = stimulus
                self.O_gpu = O_gpu
                self.prf_analysis = prf_analysis

        # NOTE: Compute TASK-SPECIFIC data, such as Stimulus, O_gpu, and Prediction signals (and their derivatives) for each stimulus
        for concatenate_block_info in required_concatenations_info:
            # NOTE: ----------------- STIMULUS SPECIFIC VARIABLES: load all the required stimulus for each participating input data in the concatenation
            for single_stimulus_info in concatenate_block_info.all_stimuli_info:
                stimulus_task_name = single_stimulus_info.stimulus_task 
                if stimulus_task_name not in task_specific_data_dict: 
                    task_specific_stimulus = GEMpRFAnalysis.load_stimulus(cfg, single_stimulus_info)
                    #...get Orthogonalization matrix
                    # NOTE: use the correct stimulus as the number of frames could be different!!!!!!!!!!!!!!
                    ortho_matrix_dim = task_specific_stimulus.NumFrames if (not task_specific_stimulus.HighTemporalResolutionEnabled) else task_specific_stimulus.NumFramesDownsampled
                    ortho_matrix = OrthoMatrix(nDCT=cfg.nDCT, num_frame_stimulus=ortho_matrix_dim) 
                    O_gpu = ortho_matrix.get_orthogonalization_matrix()
                    GemWriteToFile.get_instance().write_array_to_h5(O_gpu, variable_path=[f'model', f'{stimulus_task_name}', 'orthogonalization_matrix'], append_to_existing_variable=False)

                    prf_analysis = PRFAnalysis(prf_space=prf_space, stimulus=task_specific_stimulus) # to hold all the information about this analysis run,  # NOTE: PRFAnalysis class will be helpful for the concatenation runs, where you can store the results with different stimulus in corresponding objects (i.e. prf_analysis)                              
                    prf_analysis.orthonormalized_S_batches, prf_analysis.orthonormalized_dS_dtheta_batches_list = cls.compute_orthonormalized_signals(O_gpu=O_gpu, 
                                                                                                                                                prf_space= prf_space, 
                                                                                                                                                prf_model= prf_model, 
                                                                                                                                                stimulus= task_specific_stimulus,
                                                                                                                                                cfg = cfg,
                                                                                                                                                stimulus_task_name=stimulus_task_name) 

                    # add to dictionary
                    task_specific_data = TaskSpecificData(task_specific_stimulus, O_gpu, prf_analysis)
                    task_specific_data_dict[stimulus_task_name] = task_specific_data
                    # task_specific_data_dict[stimulus_task] = (stimulus, O_gpu, prf_analysis)
            
        # get M-inverse matrix
        if arr_2d_location_inv_M_cpu is None:
            join_start_time = datetime.datetime.now()
            MpInv_thread.join()
            if not result_queue.empty():
                arr_2d_location_inv_M_cpu = result_queue.get()
            Logger.print_timing_message(f"M-inverse thread: {datetime.datetime.now() - mpinv_thread_start_time} total, of which {datetime.datetime.now() - join_start_time} spent waiting after the GPU work")

        # NOTE: Process each Concatenation Block
        class YSignalsInfo:
            def __init__(self, Y_signals_cpu, task_name):
                self.Y_signals_cpu = Y_signals_cpu
                self.task_name = task_name

        # grid spacing used by the refined-fit -> grid fallback (verbose only)
        grid_steps = prf_space.get_grid_steps()
        if cfg.refine_fitting_enabled:
            Logger.print_timing_message(f"Grid steps: x={grid_steps[0]:.3f} deg, y={grid_steps[1]:.3f} deg, sigma={grid_steps[2]:.3f} deg (fallback threshold = {GEMpRFAnalysis.MAX_GRID_STEPS_AWAY}x)")

        counter = 0
        for concatenate_block_info in required_concatenations_info:
            counter += 1
            start_time = time.time()
            try:
                block_fallback_stats = cls._process_concatenation_block(cfg, prf_model, prf_space, concatenate_block_info, task_specific_data_dict,
                                                 arr_2d_location_inv_M_cpu, refinefit_on_gpu, default_gpu_id, YSignalsInfo,
                                                 counter, len(required_concatenations_info), start_time)
            except Exception as exc:
                Logger.print_red_message(f"Analysis FAILED for {concatenate_block_info.concatenation_result_filepath}: {exc}", print_file_name=False)
                report.add_failed(concatenate_block_info.concatenation_result_filepath, exc)
                continue

            iteration_time = time.time() - start_time
            report.add_completed(concatenate_block_info.concatenation_result_filepath, iteration_time)
            if cfg.refine_fitting_enabled:
                report.add_grid_fallback(concatenate_block_info.concatenation_result_filepath, block_fallback_stats)
            print(f"Time taken for this analysis: {iteration_time}\n")

        print ("All files processed...")

    @classmethod
    def _process_concatenation_block(cls, cfg, prf_model, prf_space, concatenate_block_info, task_specific_data_dict,
                                     arr_2d_location_inv_M_cpu, refinefit_on_gpu, default_gpu_id, YSignalsInfo,
                                     counter, num_concatenation_blocks, start_time):
        """Run a single concatenation block. Raises on failure; the caller records it in the run report.

        Returns ``(block_fallback_stats, block_fallback_records)``: the per-block grid-fallback stats
        dict (how many vertices reverted to the grid point and why) and the per-vertex rejection
        detail for the h5 /grid_fallback group. NOTE: the concatenated path deliberately does not
        recompute the per-task error for validation (too costly across all stimuli), so the
        parameter-based reasons (nan / too-far) are applied here before the model timecourses are
        synthesised; the worse-error and zero-signal reasons are not evaluated on this path
        (reported as 0).
        """
        json_data = None
        grid_steps = prf_space.get_grid_steps()
        block_fallback_stats = {key: 0 for key in ("total", "worse_error", "nan_refined",
                                                   "x_too_far", "y_too_far", "sigma_too_far",
                                                   "zero_signal", "on_grid")}
        # per-vertex rejection detail for the h5 /grid_fallback group (only flagged vertices)
        fb_index_parts, fb_reason_parts, fb_refined_parts = [], [], []
        # Collect Y-Signals
        arr_Y_signals_cpu = []
        num_concatenation_items = len(concatenate_block_info.filepaths_to_be_concatenated)
        for concat_item_idx in range(num_concatenation_items):
            input_data_filepath = concatenate_block_info.filepaths_to_be_concatenated[concat_item_idx]
            task_name = concatenate_block_info.input_data_info_to_be_concatenated[concat_item_idx].get("task")
            if not os.path.exists(input_data_filepath):
                raise InputFileMissingError(f"Input source file does not exist: {input_data_filepath}")

            Logger.print_green_message(f"Processing-{counter}/{num_concatenation_blocks} data file: {input_data_filepath}", print_file_name=False)
            measured_data_filepath = input_data_filepath

            # y-signals
            y_data = ObservedData(data_source=DataSource.measured_data)
            Y_signals_cpu = y_data.get_y_signals(measured_data_filepath)
            y_signals_info = YSignalsInfo(Y_signals_cpu, task_name)
            arr_Y_signals_cpu.append(y_signals_info)
            # arr_Y_signals_cpu.append((Y_signals_cpu, task_name))

        ###################
        # process Y-BATCHES
        ###################               
        # json_data = None   
        total_y_signals = arr_Y_signals_cpu[0].Y_signals_cpu.shape[1]
        batch_size = cls.get_y_batch_size(cfg, total_y_signals, len(prf_space.multi_dim_points_cpu))
        for current_batch_idx in range(0, total_y_signals, batch_size):    
            # go through all datasets and compute error terms for each run
            # arr_e_cpu = None #cp.empty((num_runs, batch_size, num_signals)) #[]
            # ...the runs are summed straight into one error matrix; the old code kept every run's
            # matrix alive, stacked them (a full extra copy) and only then summed
            concatenated_e = None
            Y_signals_batch_gpu_list = []
            for concat_item_idx in range(num_concatenation_items):                                
                # current Y-BATCH, for current dataset
                Y_signals_batch_gpu = ggm.get_instance().execute_cupy_func_on_default(cp.asarray, cupy_func_args=((arr_Y_signals_cpu[concat_item_idx].Y_signals_cpu)[:, current_batch_idx: current_batch_idx + batch_size],))                                        
                Y_signals_batch_gpu_list.append(Y_signals_batch_gpu)            
                Y_signals_batch_cpu = (arr_Y_signals_cpu[concat_item_idx].Y_signals_cpu)[:, current_batch_idx: current_batch_idx + batch_size]
                num_Y_signals_in_batch = Y_signals_batch_cpu.shape[1] # this is just the number of Y-signals in the current batch, it is independent of the task-name
                current_data_task = arr_Y_signals_cpu[concat_item_idx].task_name
                concatenated_e = GridFit.compute_error_matrix(Y_signals_gpu=Y_signals_batch_gpu,
                                                              S_prime_cm_gpu_batches=task_specific_data_dict[current_data_task].prf_analysis.orthonormalized_S_batches,
                                                              out=concatenated_e,
                                                              accumulate=concatenated_e is not None)

            # current Y-BATCH concatenated best fit.
            # NOTE: the index array has to live on the same side as multi_dim_points_[cpu|gpu] below,
            # so it follows isResultOnGPU -- not refinefit_on_gpu, which disagrees with it whenever
            # refine fitting is switched off.
            isResultOnGPU = ((cfg.is_refinefit_on_gpu & cfg.refine_fitting_enabled) | (not cfg.refine_fitting_enabled))
            with cp.cuda.Device(default_gpu_id):
                best_fit_proj = cp.nanargmax(concatenated_e, axis=1)
            if not isResultOnGPU:
                best_fit_proj = cp.asnumpy(best_fit_proj)

            #  current Y-BATCH refine fit
            refined_matching_results_XY = None
            coarse_pRF_estimations = None
            if cfg.refine_fitting_enabled:
                # gather e at the neighbourhood of each winner, then release the full matrix before
                # the derivative terms are evaluated into the same small shape
                neighbour_columns = RefineFit.get_neighbour_columns(prf_space, best_fit_proj, arr_2d_location_inv_M_cpu)
                e_neighbour_terms = GridFit.gather_neighbour_terms(concatenated_e, neighbour_columns)
                del concatenated_e

                de_neighbour_terms = None
                for concat_item_idx in range(num_concatenation_items):
                    current_data_task = arr_Y_signals_cpu[concat_item_idx].task_name
                    de_neighbour_terms = GridFit.accumulate_derivative_neighbour_terms(
                        Y_signals_gpu=Y_signals_batch_gpu_list[concat_item_idx],
                        dS_prime_dtheta_cm_gpu_batches_list=task_specific_data_dict[current_data_task].prf_analysis.orthonormalized_dS_dtheta_batches_list,
                        neighbour_columns=neighbour_columns,
                        out=de_neighbour_terms)

                refine_input_vecs = GridFit.build_refine_input_vectors(e_neighbour_terms, de_neighbour_terms, refinefit_on_gpu)
                del e_neighbour_terms, de_neighbour_terms

                refined_matching_results_XY, _ = RefineFit.get_refined_fit_results(prf_space=prf_space,
                                                                                num_Y_signals=num_Y_signals_in_batch,
                                                                                best_fit_proj=best_fit_proj,
                                                                                arr_2d_location_inv_M_cpu=arr_2d_location_inv_M_cpu,
                                                                                refine_input_vecs=refine_input_vecs,
                                                                                neighbour_columns=neighbour_columns)
                del refine_input_vecs
                # parameter-based fallback (nan / too-far) applied before signal synthesis,
                # so reverted vertices get the correct grid-point timecourse below.
                coarse_pRF_estimations = (prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_gpu)[isResultOnGPU][best_fit_proj]
                refined_matching_results_XY, batch_fallback_stats, batch_fallback_records = cls.apply_grid_fallback(refined_matching_results_XY, coarse_pRF_estimations, grid_steps)
                for key in block_fallback_stats:
                    block_fallback_stats[key] += batch_fallback_stats[key]
                # per-vertex rejection detail (nan / too-far only on this path; no zero-signal R2 here)
                idx, rsn, ref = cls._select_fallback_records(batch_fallback_records, current_batch_idx)
                if idx.size:
                    fb_index_parts.append(idx)
                    fb_reason_parts.append(rsn)
                    fb_refined_parts.append(ref)
            else:
                coarse_pRF_estimations = (prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_gpu)[isResultOnGPU][best_fit_proj]
                block_fallback_stats["total"] += int(coarse_pRF_estimations.shape[0])
                                        
            # Refined-result validation is skipped here: it would require recomputing with every task's stimulus, which is expensive.
            # # coarse_pRF_estimations = prf_space.multi_dim_points_cpu[best_fit_proj_cpu]
            # # valid_refined_prf_points_XY_batch = GEMpRFAnalysis.get_valid_refined_data(refined_matching_results_XY, 
            # #                                                                           Y_signals_gpu=Y_signals_batch_gpu, # NOTE: here we would need to pass the list of signals for all concatenated_item
            # #                                                                           O_gpu=O_gpu, 
            # #                                                                           prf_model=prf_model, 
            # #                                                                           stimulus=stimulus,  # NOTE: here we would need to pass the list of stimuli required for each concatenated_item
            # #                                                                           coarse_e_cpu=e_cpu, 
            # #                                                                           best_fit_proj_cpu=best_fit_proj_cpu, 
            # #                                                                           coarse_pRF_estimations=coarse_pRF_estimations)

            # Final results 
            valid_refined_prf_points_XY_batch = (coarse_pRF_estimations, refined_matching_results_XY)[cfg.refine_fitting_enabled]

            valid_refined_S_cpu_batch_list = []
            for concat_item_idx in range(num_concatenation_items):
                stimulus_task_name = arr_Y_signals_cpu[concat_item_idx].task_name
                task_specific_stimulus = task_specific_data_dict[stimulus_task_name].stimulus
                valid_refined_S_cpu_batch = GEMpRFAnalysis.get_refined_signals_cpu(valid_refined_prf_points_XY_batch, prf_model, task_specific_stimulus, cfg)
                valid_refined_S_cpu_batch_list.append(valid_refined_S_cpu_batch)

            # current Y-BATCH compute concatenated R2        
            numerators_gpu = cp.empty((num_concatenation_items, num_Y_signals_in_batch))
            denominators_gpu = cp.empty((num_concatenation_items, num_Y_signals_in_batch))
            # ...compute the numerator and denominator terms for each run's dataset individually using the above computed Refinement results. 
            # ...The O_gpu signals depend on the Stimulus so, send the correct one !!!
            for concat_item_idx in range(num_concatenation_items):
                stimulus_task_name = arr_Y_signals_cpu[concat_item_idx].task_name
                task_specific_stimulus = task_specific_data_dict[stimulus_task_name].stimulus
                task_specific_O_gpu = task_specific_data_dict[stimulus_task_name].O_gpu
                num_gpu, den_gpu = R2.get_r2_numerator_denominator_terms(Y_signals_batch_gpu_list[concat_item_idx], 
                                                                         task_specific_O_gpu, 
                                                                         valid_refined_prf_points_XY_batch, 
                                                                         valid_refined_S_cpu_batch_list[concat_item_idx])
                numerators_gpu[concat_item_idx] = num_gpu
                denominators_gpu[concat_item_idx] = den_gpu

            ## ...compute overall r2 for current Y-BATCH
            r2_numerator_term = cp.sum(numerators_gpu, axis=0)
            r2_inverse_term = (cp.sum(denominators_gpu, axis=0)) ** (-1)
            r2_result_batch = cp.where(r2_numerator_term>0, 1 - r2_numerator_term * r2_inverse_term, r2_numerator_term) 
            batch_json_data = R2.format_in_json_format(r2_result_batch, valid_refined_prf_points_XY_batch, None, refined_signals_present=False)
            if json_data is None:
                json_data = batch_json_data
            else:
                json_data += batch_json_data

            # print ("Refined fitting done...")

        block_fallback_records = cls._finalize_fallback_records(fb_index_parts, fb_reason_parts, fb_refined_parts,
                                                                num_params=int(np.asarray(grid_steps).shape[0]))

        # Write the full results of the current concatenation block to file
        ResultFileWriter.write(
            filepath=concatenate_block_info.concatenation_result_filepath,
            data=json_data,
            cfg=cfg,
            input_filepaths=concatenate_block_info.filepaths_to_be_concatenated,
            stimulus_filepath=cfg.stimulus.get('directory', cfg.stimulus.get('filepath', '')),
            run_type='concatenated',
            duration_sec=time.time() - start_time,
            grid_steps=grid_steps,
            grid_fallback_records=block_fallback_records,
        )

        Logger.print_green_message(f"Results written to file: {concatenate_block_info.concatenation_result_filepath}", print_file_name=False)

        return block_fallback_stats

    @classmethod
    def individual_run(cls, cfg, prf_model, prf_space, report):
        # time
        start_time = time.time()

        # data info
        measured_data_list, result_filepaths_list = cls.get_single_run_data_files_info(cfg)

        # skip inputs whose result already exists (in either hdf5 or json form)
        kept_pairs = cls._filter_existing_results(cfg, list(zip(measured_data_list, result_filepaths_list)), lambda pair: pair[1], report)
        measured_data_list = [pair[0] for pair in kept_pairs]
        result_filepaths_list = [pair[1] for pair in kept_pairs]
        if report.num_skipped and len(measured_data_list) == 0:
            Logger.print_green_message("All results already exist. Nothing to do.", print_file_name=False)
            return

        if len(measured_data_list) == 0:
            Logger.print_red_message("No data files found. Please check the specified paths in your XML configuration file. Aborting now...", print_file_name=False)
            return

        GemWriteToFile.get_instance().write_array_to_h5(np.array(measured_data_list), variable_path=['input_data', 'measured_data_list'], append_to_existing_variable=False)

        # stimulus
        if cfg.bids['@enable'] == "True":
            stimulus_info = GemBidsHandler.get_stimulus_info(stimulus_dir = cfg.stimulus['directory'], stimulus_name = cfg.bids['individual']['task'])
        else:
            stimulus_info = GemBidsHandler.get_non_bids_stimulus_info(cfg)
        stimulus = GEMpRFAnalysis.load_stimulus(cfg, stimulus_info)

        # M-Matrix
        if cfg.refine_fitting_enabled:
            result_queue = queue.Queue()
            MpInv_thread = threading.Thread(target=cls.execute_Grids2MpInv_NewMethod, args=(prf_space, result_queue))
            MpInv_thread.start()
            mpinv_thread_start_time = datetime.datetime.now()

        #...get Orthogonalization matrix
        ortho_matrix_dim = stimulus.NumFrames if (not stimulus.HighTemporalResolutionEnabled) else stimulus.NumFramesDownsampled
        ortho_matrix = OrthoMatrix(nDCT=cfg.nDCT, num_frame_stimulus=ortho_matrix_dim)
        O_gpu = ortho_matrix.get_orthogonalization_matrix() # (cp.eye(stim_frames)  - cp.dot(R_gpu, R_gpu.T))
        GemWriteToFile.get_instance().write_array_to_h5(O_gpu, variable_path=['model', 'orthogonalization_matrix'], append_to_existing_variable=False)


        #...compute Model Signals
        prf_analysis = PRFAnalysis(prf_space=prf_space, stimulus=stimulus) # to hold all the information about this analysis run,  # NOTE: PRFAnalysis class will be helpful for the concatenation runs, where you can store the results with different stimulus in corresponding objects (i.e. prf_analysis)                              
        prf_analysis.orthonormalized_S_batches, prf_analysis.orthonormalized_dS_dtheta_batches_list = cls.compute_orthonormalized_signals(O_gpu=O_gpu, 
                                                                                                                                    prf_space= prf_space, 
                                                                                                                                    prf_model= prf_model, 
                                                                                                                                    stimulus= stimulus,
                                                                                                                                    cfg = cfg)  
        Logger.print_green_message("model signals computed...", print_file_name=False)

        #...get M-inverse matrix
        arr_2d_location_inv_M_cpu = None
        if cfg.refine_fitting_enabled:
            inv_mat_join_start_time = datetime.datetime.now()
            MpInv_thread.join()
            if not result_queue.empty():
                arr_2d_location_inv_M_cpu = result_queue.get()
            # the thread runs concurrently with the GPU work above, so the join only measures
            # what is left of it; the thread's own wall time is the number to compare across runs.
            Logger.print_green_message(f"Time taken to compute M-inverse matrix: {datetime.datetime.now() - mpinv_thread_start_time} (of which {datetime.datetime.now() - inv_mat_join_start_time} waiting after the GPU work)\n", print_file_name=False)

        # # end time
        # end_time = time.time()
        # iteration_time = end_time - start_time     
        # print(iteration_time)  
        # iteration_times = []       
        
        # grid spacing used by the refined-fit -> grid fallback (verbose only)
        grid_steps = prf_space.get_grid_steps()
        if cfg.refine_fitting_enabled:
            Logger.print_timing_message(f"Grid steps: x={grid_steps[0]:.3f} deg, y={grid_steps[1]:.3f} deg, sigma={grid_steps[2]:.3f} deg (fallback threshold = {GEMpRFAnalysis.MAX_GRID_STEPS_AWAY}x)")

        # pRF Estimations
        file_processed_counter = 1
        data_src = []
        # data_idx = 0
        # for i in range(10):
        for data_idx in range(len(measured_data_list)):
            measured_data_filepath = measured_data_list[data_idx]

            # check if input file exists
            if not os.path.exists(measured_data_filepath):
                exc = InputFileMissingError(f"Input source file does not exist: {measured_data_filepath}")
                Logger.print_red_message(str(exc), print_file_name=False)
                report.add_failed(measured_data_filepath, exc)
                file_processed_counter += 1
                continue

            start_time = time.time()
            Logger.print_green_message(f"Processing file ({file_processed_counter}/{len(measured_data_list)}): {measured_data_filepath}", print_file_name=False)
            file_processed_counter += 1

            try:
                valid_refined_prf_points_XY, r2_results, grid_fallback_stats, grid_fallback_records = GEMpRFAnalysis.get_pRF_estimations(cfg, O_gpu, prf_space, prf_model, stimulus, prf_analysis, arr_2d_location_inv_M_cpu, measured_data_filepath)
                # profiler.disable()
                # stats = pstats.Stats(profiler, stream=profile_stream)
                # stats.strip_dirs().sort_stats("cumulative").print_stats(20)  # Top 20 most time-consuming calls
                # print(profile_stream.getvalue())


                # format results to JSON
                # NOTE: to print the refined signals in the JSON file, re-add the per-batch
                # accumulation of valid_refined_S_cpu in get_pRF_estimations() and pass it here.
                json_data = R2.format_in_json_format( r2_results, valid_refined_prf_points_XY, None, refined_signals_present=False)

                # write results to file
                ResultFileWriter.write(
                    filepath=result_filepaths_list[data_idx],
                    data=json_data,
                    cfg=cfg,
                    input_filepaths=[measured_data_filepath],
                    stimulus_filepath=os.path.join(stimulus_info.stimulus_dir, stimulus_info.stimulus_filename),
                    run_type='individual',
                    duration_sec=time.time() - start_time,
                    grid_steps=grid_steps,
                    grid_fallback_records=grid_fallback_records,
                )
            except Exception as exc:
                Logger.print_red_message(f"Analysis FAILED for {measured_data_filepath}: {exc}", print_file_name=False)
                report.add_failed(measured_data_filepath, exc)
                continue

            # information
            Logger.print_green_message(f"Results written to file: {result_filepaths_list[data_idx]}", print_file_name=False)
            data_src.append(measured_data_filepath)

            # end time
            end_time = time.time()
            iteration_time = end_time - start_time
            report.add_completed(measured_data_filepath, iteration_time)
            if cfg.refine_fitting_enabled:
                report.add_grid_fallback(measured_data_filepath, grid_fallback_stats)
            # iteration_times.append(iteration_time)
            print(f"Time taken for this analysis: {iteration_time}\n")

            # write the time taken for each iteration
            # csv_filepath = r"D:\results\gem-paper-simulated-data\analysis\05\BIDS\derivatives\time_records\v2_iteration_times_151x151x16.csv"
            # csv_filepath = r"/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/05/BIDS/derivatives/time_records/v2_iteration_times_151x151x16--RefinefitScipy.csv"
            # df = pd.DataFrame({'DataSrc': data_src, 'Time (seconds)': iteration_times})
            # df.to_csv(csv_filepath, index=False)
        print ("All files processed...")

      

    @classmethod
    def run(cls, cfg, prf_model, prf_space, report):
        # Run the analysis (Concatenation or Individual Run)
        if cfg.bids['@enable'] == "True" and cfg.bids['@run_type'].lower() == "concatenated":
            GEMpRFAnalysis.concatenated_run(cfg, prf_model, prf_space, report)
        else:
            GEMpRFAnalysis.individual_run(cfg, prf_model, prf_space, report)

        return 0


# # ################################################---------------------------------MAIN---------------------------------################################################
# # # run the main function
# # if __name__ == "__main__":    
# #     start_time = datetime.datetime.now()

# #     print ("Running the GEM pRF Analysis...")
# #     # from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
# #     # GEMpRFAnalysis.run()    
# #     # cProfile.run('GEMpRFAnalysis.run()', sort='cumulative')

# #     # Run the profiling
# #     # profiler = cProfile.Profile()
# #     # profiler.enable()

# #     config_filepath = os.path.join(os.path.dirname(__file__), '..', 'configs', 'analysis_configs', 'analysis_config.xml')

# #     # config_filepath = r'D:\code\sid-git\fmri\gem\configs\default_config\new_concatenationDummyTest_config.xml'
# #     # GEMpRFAnalysis.concatenated_run(config_filepath)
# #     print("Starting GEM analysis...")
# #     GEMpRFAnalysis.run(config_filepath)
# #     # profiler.disable()

# #     # print time taken
# #     print(f"Complete Time taken: {datetime.datetime.now() - start_time}")

# #     # # Specify the file name to save the profiling results
# #     # output_file = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/tests/gem-paper-simulated-data/analysis/05/BIDS/derivatives/prfanalyze-gem/analysis-05/sub-100000/ses-0n0/profiling_results.txt'

# #     # # Dump the profiling statistics to the specified file
# #     # with open(output_file, 'w') as f:
# #     #     stats = pstats.Stats(profiler, stream=f)
# #     #     stats.sort_stats('cumulative')
# #     #     stats.print_stats()