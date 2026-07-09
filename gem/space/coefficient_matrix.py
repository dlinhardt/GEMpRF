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
    The caller slices them back into the ragged list its consumers expect.

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

        # Back to the ragged "list of 2-D arrays" contract the consumers rely on. These are copies,
        # not views: the padded buffer is dense (num_points x 10 x max_cols) and would otherwise be
        # kept alive in full by any surviving slice, well above what the ragged result needs.
        arr_2d_location_inv_M = [padded_inv_M[i, :, :num_cols[i]].copy() for i in range(num_points)]
        del padded_inv_M

        return arr_2d_location_inv_M
