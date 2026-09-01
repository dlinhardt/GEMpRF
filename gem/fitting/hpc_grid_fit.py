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
    # Target size of the transient product that is held while one column slice is written into the
    # error matrix. The slow path used to multiply a whole model-signal chunk at once and only then
    # add it into `out`, so its peak was `out` plus a second array of the same width. On the
    # concatenated path, where `out` is passed in and accumulated into, that is what failed:
    # 791 vertices x 471,015 columns x 8 bytes = the 2,980,583,424 byte allocation cons01 died on.
    # Slicing the columns bounds the transient here instead of letting it scale with the grid.
    ERROR_MATRIX_CHUNK_BYTES = 256 * 1024 ** 2

    @classmethod
    def compute_error_term(cls, Y_signals_gpu, S_prime_columnmajor_gpu):
        e_gpu = (Y_signals_gpu.T @ S_prime_columnmajor_gpu)
        return e_gpu

    @classmethod
    def _fold_positive_infinity(cls, e_gpu):
        """Rewrite +inf error terms to -inf, in place.

        orthonormalize_modelled_signals() flags a degenerate model signal -- a pRF that drifted out
        of the aperture, leaving a zero column -- by setting its whole column to +inf. Error terms
        are maximised, so leaving +inf would make exactly those signals win every argmax and never
        revert to their grid point. Folding to -inf makes them lose instead.
        """
        e_gpu[cp.isinf(e_gpu) & (e_gpu > 0)] = -cp.inf
        return e_gpu

    @classmethod
    def _column_slice_width(cls, num_y_signals, num_columns):
        """How many model-signal columns to multiply at once, from ERROR_MATRIX_CHUNK_BYTES.

        Derived from the vertex count rather than fixed, so the transient stays the same size
        whatever the batch size is: a wider Y-batch simply gets fewer columns per slice. When the
        whole chunk already fits, this returns its full width and the loop runs exactly once, which
        is what every configuration that fits today keeps doing.
        """
        bytes_per_column = max(1, int(num_y_signals) * 8)
        width = int(cls.ERROR_MATRIX_CHUNK_BYTES // bytes_per_column)
        return max(1, min(int(num_columns), width))

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

        Each chunk is multiplied a column slice at a time (see ERROR_MATRIX_CHUNK_BYTES). Without
        that, the peak here is ``out`` plus a transient of the same width, which is what the
        concatenated path -- the only caller that passes ``out`` and accumulates -- ran out of
        memory on. Slicing the columns does not change any value: every output column is a dot
        product over the same summation index against the same operands, so the columns a call
        happens to share affect nothing but residency.
        """
        default_gpu_id = ggm.get_instance().default_gpu_id
        num_batches = len(S_prime_cm_gpu_batches)

        # each model signal occupies one column, so the grid width is the sum of the chunk widths
        total_model_signals = sum(batch.shape[1] for batch in S_prime_cm_gpu_batches)

        # NOTE: with a single model-signal chunk that already lives on the default device there is
        # nothing to assemble -- the product IS the error matrix. Allocating `out` and copying the
        # chunk into it doubled the peak for exactly the configuration that can least afford it: one
        # GPU, where that one chunk spans the whole grid and the copy is gigabytes.
        if out is None and num_batches == 1 and S_prime_cm_gpu_batches[0].device.id == default_gpu_id:
            with cp.cuda.Device(default_gpu_id):
                return cls._fold_positive_infinity(
                    cls.compute_error_term(Y_signals_gpu, S_prime_cm_gpu_batches[0]))

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

            # Multiply and store one column slice at a time. Every output column is an independent
            # dot product over the same summation index, so which columns share a call changes no
            # value -- only how much is resident at once.
            slice_width = cls._column_slice_width(total_y_signals, num_signals_current_batch)
            for slice_start in range(0, num_signals_current_batch, slice_width):
                slice_stop = min(slice_start + slice_width, num_signals_current_batch)

                with cp.cuda.Device(device_id):
                    chunk_e_result_gpu = cls._fold_positive_infinity(
                        cls.compute_error_term(current_device_Y_signals,
                                               S_prime_cm_gpu_batches[batch_idx][:, slice_start:slice_stop]))

                # finally, get the "error" result back to the default device
                with cp.cuda.Device(default_gpu_id):
                    chunk_on_default = chunk_e_result_gpu if device_id == default_gpu_id else cp.array(chunk_e_result_gpu)
                    target = out[:, column_idx + slice_start : column_idx + slice_stop]
                    if accumulate:
                        target += chunk_on_default
                    else:
                        target[...] = chunk_on_default
                # NOTE: both names have to go. For a chunk that already lives on the default device
                # they are the same object, so dropping only one keeps the transient alive while the
                # next slice allocates its own.
                del chunk_on_default, chunk_e_result_gpu

            del current_device_Y_signals

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
            # Same fold the full-matrix path applies; here it matters because the caller reads a
            # smaller error as "refinement made it worse".
            return cls._fold_positive_infinity(cp.concatenate(parts))

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
    def _chunk_column_map(cls, neighbour_columns_cpu, column_offset, chunk_width):
        """Which of this chunk's columns are wanted, and where each request lands in the narrow product.

        Returns ``(needed_columns, narrow_columns, in_chunk)`` or ``None`` when no y-signal asks for
        anything in this chunk. ``needed_columns`` is the ascending, de-duplicated list of chunk-local
        columns to actually multiply; ``narrow_columns`` maps every entry of ``neighbour_columns`` to
        its position within that narrow result (arbitrary where ``in_chunk`` is False, and masked off
        by the caller).

        Built with a boolean mark + flatnonzero rather than np.unique: same ascending-unique result,
        O(chunk_width) instead of a sort, and ~3x cheaper measured. It runs once per chunk and is
        shared by every theta.
        """
        local_columns = neighbour_columns_cpu - column_offset
        in_chunk = (local_columns >= 0) & (local_columns < chunk_width)
        if not in_chunk.any():
            return None

        wanted = np.zeros(chunk_width, dtype=bool)
        wanted[local_columns[in_chunk]] = True
        needed_columns = np.flatnonzero(wanted)

        position_of_column = np.zeros(chunk_width, dtype=np.int32)
        position_of_column[needed_columns] = np.arange(needed_columns.size, dtype=np.int32)
        # out-of-chunk requests are clipped to a valid index and then masked away, exactly as before
        narrow_columns = position_of_column[np.clip(local_columns, 0, chunk_width - 1)]

        return needed_columns, narrow_columns, in_chunk

    @classmethod
    def accumulate_derivative_neighbour_terms(cls, Y_signals_gpu, dS_prime_dtheta_cm_gpu_batches_list,
                                              neighbour_columns, out=None):
        """Derivative error terms, evaluated only at the neighbour columns of the winning grid point.

        Only the <=27 neighbours of each y-signal's winner are ever read, so only those columns of
        each model-signal chunk are multiplied: the chunk is sliced down to the columns some y-signal
        actually asks for and ``Y.T @ dS'_narrow`` is run on that. The full-width product this used to
        form was 75% of the fit's arithmetic and ~2-8% of it was read, and the (num_y_signals,
        chunk_width) temporary it needed was 5.96 GB on a single-GPU 942k-point grid -- the same size
        as the error matrix it had to sit alongside.

        ``out`` is (num_params, num_y_signals, num_neighbours) on the default device and is added to,
        which is how the concatenated runs are summed.
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

            column_map = cls._chunk_column_map(neighbour_columns_cpu, column_idx, chunk_width)
            if column_map is None:  # no winner's neighbourhood reaches into this chunk
                column_idx = column_idx + chunk_width
                continue
            needed_columns_cpu, narrow_columns_cpu, in_chunk_cpu = column_map

            with cp.cuda.Device(device_id):
                needed_columns = cp.asarray(needed_columns_cpu)
                narrow_columns = cp.asarray(narrow_columns_cpu)
                in_chunk = cp.asarray(in_chunk_cpu)
                rows = cp.arange(num_y_signals)[:, None]
                current_device_Y_signals = cls._y_signals_on_device(Y_signals_gpu, device_id, default_gpu_id)

                for theta in range(num_theta_params):
                    narrow_dS_prime = dS_prime_dtheta_cm_gpu_batches_list[theta][chunk_idx][:, needed_columns]
                    narrow_de_dtheta_gpu = cls.compute_error_term(current_device_Y_signals, narrow_dS_prime)
                    del narrow_dS_prime

                    gathered = cp.where(in_chunk, narrow_de_dtheta_gpu[rows, narrow_columns], 0.0)
                    del narrow_de_dtheta_gpu

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
