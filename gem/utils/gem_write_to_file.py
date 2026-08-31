# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Siddharth Mittal",
"@Desc    :   None",     
"""

import os
import h5py
import numpy as np
import cupy as cp


class GemWriteToFile:
    _instance = None  # Singleton instance

    def __new__(cls, result_dir, debugging_enabled=False):
        if cls._instance is None:
            cls._instance = super(GemWriteToFile, cls).__new__(cls)
            cls._instance.__initialize(result_dir, debugging_enabled)
        elif cls._instance.__result_dir != result_dir or cls._instance.__debugging_enabled != debugging_enabled: # Re-initialize if parameters differ
            cls._instance.__initialize(result_dir, debugging_enabled)
        return cls._instance

    def __initialize(self, result_dir, debugging_enabled):
        self.__result_dir = result_dir
        self.__debugging_enabled = debugging_enabled
        # The first write of a run truncates the debug file instead of adding to whatever the
        # previous run left there -- see __create_dataset for why appending to it is not free.
        self.__debug_file_started = False

    @classmethod
    def get_instance(cls):
        return cls._instance

    def write_array_to_h5(self, data, variable_path, append_to_existing_variable=False):
        """
        Write a NumPy/CuPy array or list of arrays into an HDF5 file with hierarchical groups.
        If variable_path corresponds to 'model_signals' or 'derivative_model_signals_*', 
        concatenates list of arrays along axis=0 before writing.
        """
        if not self.__debugging_enabled:
            return

        filepath = os.path.join(self.__result_dir, "debug_model_data.h5")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert variable_path to a slash-separated string
        if isinstance(variable_path, (list, tuple)):
            variable_path_str = "/".join(variable_path)
        else:
            variable_path_str = variable_path

        # Concatenate list of arrays if needed
        if isinstance(data, list):
            # Keywords for special handling
            special_keys = [
                "model_signals",
                "model_signals_derivative",
                "orthonormalized_model_signals",
                "orthonormalized_model_signals_derivative",
            ]

            if any(key in variable_path_str for key in special_keys):
                # Flatten list of lists
                flat_list = []
                for item in data:
                    flat_list.extend(item if isinstance(item, list) else [item])

                concat_arrays = []
                for arr in flat_list:
                    if isinstance(arr, cp.ndarray):
                        arr = cp.asnumpy(arr)

                    # Transpose first if it's an orthonormalized variant
                    if any(key in variable_path_str for key in [
                        "orthonormalized_model_signals",
                        "orthonormalized_model_signals_derivative"
                    ]):
                        arr = arr.T

                    concat_arrays.append(arr)

                # Now concatenate safely
                data_to_write = np.concatenate(concat_arrays, axis=0)
            else:
                # Not a special variable, just convert list → numpy
                data_to_write = np.array(data)                
        else:
            # Already array-like, move to CPU if needed
            data_to_write = cp.asnumpy(data) if isinstance(data, cp.ndarray) else data


        # Handle string arrays
        if np.issubdtype(data_to_write.dtype, np.str_) or np.issubdtype(data_to_write.dtype, np.object_):
            data_to_write = np.array(data_to_write, dtype=h5py.string_dtype(encoding='utf-8'))

        # Open HDF5 file. The first write of a run truncates: every dataset this writer produces is
        # rewritten from scratch each run, so carrying the previous run's copy forward only costs
        # space. NOTE: this makes two runs writing into the same result_dir at the same time
        # mutually destructive -- they already were, since they overwrite each other's datasets.
        file_mode = "a"
        if not self.__debug_file_started:
            file_mode = "w"
            self.__debug_file_started = True

        with h5py.File(filepath, file_mode) as f:
            if variable_path_str in f:
                dset = f[variable_path_str]
                if append_to_existing_variable:
                    # Handle 1D arrays
                    if data_to_write.ndim == 1:
                        data_to_write = data_to_write.reshape(-1, 1)
                        dset_shape = (dset.shape[0], 1)
                    else:
                        dset_shape = dset.shape

                    if data_to_write.shape[1:] != dset_shape[1:]:
                        raise ValueError(
                            f"Shape mismatch: cannot append array {data_to_write.shape} "
                            f"to existing dataset {dset.shape}"
                        )
                    dset.resize((dset.shape[0] + data_to_write.shape[0]), axis=0)
                    dset[-data_to_write.shape[0]:] = data_to_write
                else:
                    del f[variable_path_str]
                    self.__create_dataset(f, variable_path_str, data_to_write, append_to_existing_variable)
            else:
                self.__create_dataset(f, variable_path_str, data_to_write, append_to_existing_variable)

    @staticmethod
    def __create_dataset(f, variable_path_str, data_to_write, resizable):
        """Create the dataset, extendable only if something is actually going to extend it.

        Passing ``maxshape=`` forces chunked storage -- ~12,000 chunks per 2.9 GB array for the
        whole-grid signals. When such a dataset was deleted and rewritten, the freed chunks were
        never picked up again: the default file space strategy does not persist free space across a
        close, and this writer opens and closes the file once per dataset. The result was that
        **every run stranded its predecessor's data in full**. Measured against the pre-change
        writer, five runs of identical datasets into one directory grew the file 5.00x, exactly
        linearly, and alternating two dataset sizes grew it 10.98x. On the server that produced a
        99.33 GB debug_model_data.h5 holding 28 GB of live datasets, its first 71.29 GB one
        contiguous unreferenced region that h5stat reported as 0 bytes of tracked free space.

        A dataset created without ``maxshape`` is contiguous, allocated as a single block, and is
        reused cleanly. Nothing currently passes append_to_existing_variable=True, so in practice
        every dataset takes the contiguous path; the resizable branch stays for that caller's sake.
        """
        if resizable:
            maxshape = (None,) + data_to_write.shape[1:] if data_to_write.ndim > 1 else (None,)
            f.create_dataset(variable_path_str, data=data_to_write, maxshape=maxshape)
        else:
            f.create_dataset(variable_path_str, data=data_to_write)