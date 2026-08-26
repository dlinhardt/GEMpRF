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
    def write(cls, filepath, data, cfg, input_filepaths, stimulus_filepath, run_type, duration_sec, grid_steps=None, grid_fallback_records=None):
        fmt = (cfg.results or {}).get('output_format', 'hdf5')
        base = os.path.splitext(filepath)[0]
        if fmt == 'json':
            cls.write_json(base + '.json', data)
        else:  # 'hdf5' or 'h5'
            cls.write_h5(base + '.h5', data, cfg, input_filepaths, stimulus_filepath, run_type, duration_sec,
                         grid_steps=grid_steps, grid_fallback_records=grid_fallback_records)

    # JSON is a human-readable dump, so it gets rounded values; full float64 repr makes it
    # unreadable and the file several times larger.
    JSON_DECIMALS = 4

    @classmethod
    def write_json(cls, filepath, data):
        """Serialise the estimates as JSON, rounded.

        NOTE: the rounding belongs here and nowhere else. It used to live in
        JsonMgr.args2jsonEntry() -- whose name reads as JSON-only -- but write_h5() unpacks the very
        same dicts, so every HDF5 result was quantised to 1e-4 as well: pRF centres pinned to a
        0.0001 deg lattice, in the format that is meant to be the precise one.
        """
        rounded = [{key: cls._rounded_value(value) for key, value in record.items()} for record in data]
        JsonMgr.write_to_file(filepath, rounded)

    @classmethod
    def _rounded_value(cls, value):
        if isinstance(value, list): # modelpred, a whole timecourse
            return [round(float(element), cls.JSON_DECIMALS) for element in value]
        if isinstance(value, (int, float, np.floating, np.integer)):
            return round(float(value), cls.JSON_DECIMALS)
        return value # None (modelpred omitted), strings, anything added later

    @classmethod
    def write_h5(cls, filepath, data, cfg, input_filepaths, stimulus_filepath, run_type, duration_sec, grid_steps=None, grid_fallback_records=None):
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

            # --- grid_fallback (per-vertex refined-fit rejection detail) ---
            # Storage-efficient side table: only the rejected vertices are listed. The normal
            # /parameters group is untouched (reverted vertices simply hold their grid point there),
            # so this group is purely additive and ignored by readers that don't know about it.
            cls._write_grid_fallback(f, grid_fallback_records)

    @staticmethod
    def _write_grid_fallback(f, records):
        """Write the optional /grid_fallback group when there are rejected vertices to report.

        ``records`` (from GEMpRFAnalysis._finalize_fallback_records) carries the flagged vertices
        plus a self-describing reason legend, so nothing here needs to import the analysis module.
        """
        if not records:
            return
        vertex_index = np.asarray(records.get('vertex_index', []))
        if vertex_index.size == 0:
            return  # refine fitting off, or nothing was rejected -> no group at all

        str_dt = h5py.special_dtype(vlen=str)

        g = f.create_group('grid_fallback')
        g.create_dataset('vertex_index',   data=vertex_index.astype(np.int32))
        g.create_dataset('reason',         data=np.asarray(records['reason']).astype(np.uint8))
        g.create_dataset('refined_params', data=np.asarray(records['refined_params']).astype(np.float32))

        # legend so `reason` decodes without any external doc. Stored purely as datasets (visible in
        # every h5 explorer) -- NOT as group attributes, which some explorers hide.
        reason_bits = records.get('reason_bits', {})
        param_columns = str(records.get('param_columns', 'Centerx0,Centery0,sigmaMajor'))
        if reason_bits:
            ordered = sorted(reason_bits.items(), key=lambda kv: kv[1])
            legend = ', '.join(f'{int(bit)}={name}' for name, bit in ordered)
            g.create_dataset('reason_legend',      data=str(legend))
            g.create_dataset('reason_bit_names',   data=np.array([n for n, _ in ordered], dtype=str_dt))
            g.create_dataset('reason_bit_values',  data=np.array([int(b) for _, b in ordered], dtype=np.uint8))
        g.create_dataset('param_columns',          data=str(param_columns))
