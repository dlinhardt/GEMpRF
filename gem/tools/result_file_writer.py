import os
import numpy as np
import h5py

from gem.tools.json_file_operations import JsonMgr


class ResultFileWriter:

    @staticmethod
    def result_exists(filepath):
        """True if a result already exists at this path in EITHER hdf5 or json form."""
        base = os.path.splitext(filepath)[0]
        return os.path.exists(base + '.h5') or os.path.exists(base + '.json')

    @classmethod
    def write(cls, filepath, data, cfg, input_filepaths, stimulus_filepath, run_type, duration_sec, grid_steps=None):
        fmt = (cfg.results or {}).get('output_format', 'hdf5')
        base = os.path.splitext(filepath)[0]
        if fmt == 'json':
            cls.write_json(base + '.json', data)
        else:  # 'hdf5' or 'h5'
            cls.write_h5(base + '.h5', data, cfg, input_filepaths, stimulus_filepath, run_type, duration_sec, grid_steps=grid_steps)

    @classmethod
    def write_json(cls, filepath, data):
        JsonMgr.write_to_file(filepath, data)

    @classmethod
    def write_h5(cls, filepath, data, cfg, input_filepaths, stimulus_filepath, run_type, duration_sec, grid_steps=None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # unpack list-of-dicts into arrays
        centerx  = np.array([d['Centerx0']   for d in data], dtype=np.float32)
        centery  = np.array([d['Centery0']    for d in data], dtype=np.float32)
        theta    = np.array([d['Theta']       for d in data], dtype=np.float32)
        sigma_mj = np.array([d['sigmaMajor']  for d in data], dtype=np.float32)
        sigma_mn = np.array([d['sigmaMinor']  for d in data], dtype=np.float32)
        r2       = np.array([d['R2']          for d in data], dtype=np.float32)
        modelpred_list = [d.get('modelpred') for d in data]
        has_modelpred = modelpred_list[0] is not None

        str_dt = h5py.special_dtype(vlen=str)

        with h5py.File(filepath, 'w') as f:

            # --- parameters ---
            pg = f.create_group('parameters')
            pg.create_dataset('Centerx0',  data=centerx)
            pg.create_dataset('Centery0',  data=centery)
            pg.create_dataset('Theta',     data=theta)
            pg.create_dataset('sigmaMajor',data=sigma_mj)
            pg.create_dataset('sigmaMinor',data=sigma_mn)
            pg.create_dataset('R2',        data=r2)
            if has_modelpred:
                pg.create_dataset('modelpred', data=np.array(modelpred_list, dtype=np.float32))

            # --- metadata/analysis ---
            ag = f.create_group('metadata/analysis')
            ag.create_dataset('model',                   data=str(cfg.pRF_model_details.get('model', '')))
            ag.create_dataset('refine_fitting_enabled',  data=bool(cfg.refine_fitting_enabled))
            ag.create_dataset('nDCT',                    data=int(cfg.nDCT))
            ag.create_dataset('write_debug_info',        data=bool(cfg.write_debug_info))

            # --- metadata/search_space ---
            sg = f.create_group('metadata/search_space')
            sg.create_dataset('visual_field_radius', data=float(cfg.default_spatial_grid['visual_field_radius']))
            sg.create_dataset('num_horizontal_prfs', data=int(cfg.default_spatial_grid['num_horizontal_prfs']))
            sg.create_dataset('num_vertical_prfs',   data=int(cfg.default_spatial_grid['num_vertical_prfs']))
            sg.create_dataset('num_sigmas',          data=int(cfg.default_sigmas['num_sigmas']))
            sg.create_dataset('min_sigma',           data=float(cfg.default_sigmas['min_sigma']))
            sg.create_dataset('max_sigma',           data=float(cfg.default_sigmas['max_sigma']))
            # per-dimension grid spacing (degrees); the refined-fit -> grid fallback uses 2x these
            if grid_steps is not None:
                sg.create_dataset('x_grid_step',     data=float(grid_steps[0]))
                sg.create_dataset('y_grid_step',     data=float(grid_steps[1]))
                sg.create_dataset('sigma_grid_step', data=float(grid_steps[2]))

            # --- metadata/stimulus ---
            stg = f.create_group('metadata/stimulus')
            stg.create_dataset('visual_field',            data=float(cfg.stimulus.get('visual_field', 0)))
            stg.create_dataset('width',                   data=int(cfg.stimulus.get('width', 0)))
            stg.create_dataset('height',                  data=int(cfg.stimulus.get('height', 0)))
            bin_node = cfg.stimulus.get('binarization', {})
            stg.create_dataset('binarization_enabled',    data=(bin_node.get('@enable', 'False').lower() == 'true'))
            stg.create_dataset('binarization_threshold',  data=float(bin_node.get('@threshold', 0)))

            # --- metadata/hrf ---
            hg = f.create_group('metadata/hrf')
            hrf = cfg.default_hrf
            hg.create_dataset('TR',                  data=float(hrf['TR']) if hrf['TR'] is not None else -1.0)
            t = hrf.get('t', (0.0, 45.0))
            hg.create_dataset('t_start',             data=float(t[0]))
            hg.create_dataset('t_stop',              data=float(t[1]))
            hg.create_dataset('peak_delay',          data=float(hrf['peak_delay']))
            hg.create_dataset('under_shoot_delay',   data=float(hrf['under_shoot_delay']))
            hg.create_dataset('peak_disp',           data=float(hrf['peak_disp']))
            hg.create_dataset('under_disp',          data=float(hrf['under_disp']))
            hg.create_dataset('peak_to_undershoot',  data=float(hrf['peak_to_undershoot']))
            hg.create_dataset('normalize',           data=bool(hrf['normalize']))

            # --- metadata/input_files ---
            ig = f.create_group('metadata/input_files')
            fps = [str(p) for p in input_filepaths]
            ig.create_dataset('measured_data',    data=np.array(fps, dtype=str_dt))
            ig.create_dataset('stimulus_filepath', data=str(stimulus_filepath))

            # --- metadata/run_info ---
            rg = f.create_group('metadata/run_info')
            rg.create_dataset('run_type',             data=str(run_type))
            rg.create_dataset('analysis_duration_sec', data=float(duration_sec))
