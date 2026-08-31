import numpy as np
from numba import njit, prange

# linear equations specifications - Gaussian model
NUM_UNKNOWN_COEFFICIENTS = 10
NUM_LINEAR_EQUATIONS = 4


@njit(nogil=True, cache=True, inline='always')
def _fill_M_rows(M, i, vecX):
    """Write the 4 rows (value + 3 derivatives) contributed by one neighbour into M.

    NOTE: written as explicit scalar stores rather than `M[row, :] = [...]`, because a list
    literal fails type inference inside a `parallel=True` region.
    """
    row = i * NUM_LINEAR_EQUATIONS

    # e
    M[row, 0] = vecX[0] ** 2
    M[row, 1] = vecX[1] ** 2
    M[row, 2] = vecX[2] ** 2
    M[row, 3] = 2. * vecX[0] * vecX[1]
    M[row, 4] = 2. * vecX[0] * vecX[2]
    M[row, 5] = 2. * vecX[1] * vecX[2]
    M[row, 6] = vecX[0]
    M[row, 7] = vecX[1]
    M[row, 8] = vecX[2]
    M[row, 9] = 1.

    # de_dx
    M[row + 1, 0] = 2. * vecX[0]
    M[row + 1, 3] = 2. * vecX[1]
    M[row + 1, 4] = 2. * vecX[2]
    M[row + 1, 6] = 1.

    # de_dy
    M[row + 2, 1] = 2. * vecX[1]
    M[row + 2, 3] = 2. * vecX[0]
    M[row + 2, 5] = 2. * vecX[2]
    M[row + 2, 7] = 1.

    # de_dsigma
    M[row + 3, 2] = 2. * vecX[2]
    M[row + 3, 4] = 2. * vecX[0]
    M[row + 3, 5] = 2. * vecX[1]
    M[row + 3, 8] = 1.


@njit(nogil=True, parallel=True, cache=True)
def GEM_Grids2MpInv_numba(neighbours_flat, neighbours_offsets, neighbours_counts, arr_2d_location_inv_M, num_cols):
    """Compute the Moore-Penrose pseudoinverse of the coefficient matrix M at every grid point.

    The per-point pseudoinverses are ragged (a point with fewer neighbours yields fewer columns),
    so they are written into a padded (N, 10, max_cols) buffer alongside the true column count.
    The caller keeps that buffer as-is -- it is the layout the refinement's per-batch gather wants.

    NOTE: the neighbours are passed flattened (values + per-point offsets/counts) because a
    reflected list of arrays does not type under `parallel=True`.
    """
    for multi_dim_point_idx in prange(len(neighbours_counts)):
        num_neighbours = neighbours_counts[multi_dim_point_idx]
        offset = neighbours_offsets[multi_dim_point_idx]

        M = np.zeros((num_neighbours * NUM_LINEAR_EQUATIONS, NUM_UNKNOWN_COEFFICIENTS), dtype=np.float64)
        for i in range(num_neighbours):
            _fill_M_rows(M, i, neighbours_flat[offset + i])

        Mp_inv = np.linalg.pinv(M)
        num_cols[multi_dim_point_idx] = Mp_inv.shape[1]
        arr_2d_location_inv_M[multi_dim_point_idx, :, :Mp_inv.shape[1]] = Mp_inv


class MpInvTable:
    """The per-grid-point pseudoinverses, kept in the padded (N, 10, max_cols) layout numba fills.

    The refinement gathers whole rows of this buffer per batch (``padded[best_fit_proj]``), so the
    padded layout *is* the useful one -- it is what makes the gather a single rectangular fancy-index.
    This used to be shredded into a list of N trimmed copies right after the kernel wrote it, and then
    reassembled into the identical padded array on the first refined fit: at N = 942,030 that was
    ~9.4 million Python-level slices rebuilding 8 GB that had just been thrown away.

    Indexing/iterating this object still yields the trimmed 2-D arrays the old ragged-list contract
    promised, but as **views** into the padded buffer rather than copies. Views were rejected before
    because a surviving slice would pin the whole buffer; that inverts once the buffer is the thing
    being kept.

    NOTE: the padded region is zero, where the old ragged rebuild left NaN. Both are read only through
    ``_compute_coefficients``, which runs ``nan_to_num(MpInv, nan=0.0)`` first, so the two are exactly
    equal after masking -- see the padding note in RefineFit._prepare_padded_arrays.
    """

    __slots__ = ("padded", "num_cols")

    def __init__(self, padded, num_cols):
        self.padded = padded
        self.num_cols = num_cols

    def __len__(self):
        return self.padded.shape[0]

    def __getitem__(self, index):
        return self.padded[index, :, :self.num_cols[index]]


class CoefficientMatix:
    @classmethod
    def Wrapper_Grids2MpInv_numba(cls, multi_dim_points, multi_dim_points_neighbours):
        num_points = len(multi_dim_points_neighbours)
        neighbours_counts = np.array([len(n) for n in multi_dim_points_neighbours], dtype=np.int64)

        neighbours_offsets = np.zeros(num_points, dtype=np.int64)
        np.cumsum(neighbours_counts[:-1], out=neighbours_offsets[1:])
        neighbours_flat = np.ascontiguousarray(np.concatenate(multi_dim_points_neighbours), dtype=np.float64)

        max_cols = int(neighbours_counts.max()) * NUM_LINEAR_EQUATIONS
        padded_inv_M = np.zeros((num_points, NUM_UNKNOWN_COEFFICIENTS, max_cols), dtype=np.float64)
        num_cols = np.zeros(num_points, dtype=np.int64)

        GEM_Grids2MpInv_numba(neighbours_flat, neighbours_offsets, neighbours_counts, padded_inv_M, num_cols)

        # Hand back the padded buffer itself. MpInvTable still indexes like the ragged list of 2-D
        # arrays the consumers were written against, but the refinement reads `.padded` directly
        # instead of rebuilding it.
        return MpInvTable(padded_inv_M, num_cols)
