"""Bit-exact identity checks for the M-inverse optimizations.

Several optimizations replaced hot code in the M-inverse stage:

  1. `get_points_neighbours_chunk_Numba` filters candidate neighbours with a boolean mask over the
     full grid instead of `x not in validated_multidim_indices` (an O(N) scan in numba).
  2. It also takes a precomputed KD-tree query. The xy neighbours depend only on a point's xy
     coordinates, so the tree is queried once per unique xy point rather than once per replication
     of that point across the extra dimensions.
  3. It processes a chunk of points per call, and the driver dispatches one joblib task per chunk
     instead of one per point.
  4. `GEM_Grids2MpInv_numba` runs the per-point `pinv` under `prange`, writing into a padded buffer
     that the wrapper slices back into the ragged list its consumers expect.

None of these may change a single bit of the output. This module pins the pre-optimization
reference implementations and asserts exact equality (`np.array_equal`, not `allclose`) against the
live code, over both mask branches (validation dropping points, and no validation at all).

Run on a machine with CuPy available (gem.space.PRFSpace imports cupy at module level):

    pytest gem/tests/test_mpinv_optimization_identity.py -v
"""
import numpy as np
import numba
import pytest
from numba import njit

from gem.space.PRFSpace import PRFSpace
from gem.space.coefficient_matrix import CoefficientMatix
from gem.model.prf_gaussian_model import PRFGaussianModel


# --------------------------------------------------------------------------------------
# Pinned reference implementations (verbatim, pre-optimization)
# --------------------------------------------------------------------------------------

@njit(nogil=True)
def _reference_Grids2MpInv(multi_dim_points, multi_dim_points_neighbours):
    """Verbatim pre-optimization GEM_Grids2MpInv_numba: serial loop, list append."""
    arr_2d_location_inv_M = []
    num_unknown_coefficients = 10
    num_linear_equations = 4

    for multi_dim_point_idx in range(len(multi_dim_points)):
        gaussian_args = multi_dim_points_neighbours[multi_dim_point_idx]
        num_neighbours = len(gaussian_args)
        M = np.zeros((num_neighbours * num_linear_equations, num_unknown_coefficients), dtype=float)
        for i in range(len(gaussian_args)):
            vecX = gaussian_args[i]
            x = np.array(
                [[vecX[0] ** 2, vecX[1] ** 2, vecX[2] ** 2, 2. * vecX[0] * vecX[1], 2. * vecX[0] * vecX[2], 2. * vecX[1] * vecX[2], vecX[0], vecX[1], vecX[2], 1],
                 [2. * vecX[0], 0, 0, 2. * vecX[1], 2. * vecX[2], 0, 1, 0, 0, 0],
                 [0, 2. * vecX[1], 0, 2. * vecX[0], 0, 2. * vecX[2], 0, 1, 0, 0],
                 [0, 0, 2. * vecX[2], 0, 2. * vecX[0], 2. * vecX[1], 0, 0, 1, 0]]
            )
            M[i * num_linear_equations: (i * num_linear_equations) + num_linear_equations, :] = x

        arr_2d_location_inv_M.append(np.linalg.pinv(M))

    return arr_2d_location_inv_M


def _reference_neighbours_for_point(multi_dim_point, kdtree, points_xy, sigma_values,
                                    validated_set, num_neighbors=9):
    """Independently recompute the neighbour flat-index list for one point, in pure numpy.

    Mirrors the neighbour kernel for the single-extra-dimension (sigma) case, applying the ORIGINAL
    `flat_idx in validated` predicate and querying the KD-tree per point (as the code did before the
    query was hoisted out). Built from the inputs rather than from the live output, so it catches a
    neighbour wrongly *dropped* as well as one wrongly kept -- which filtering the emitted list could
    not -- and, because it never consults `xy_row_ids`, a wrong xy-row lookup after the hoist.
    """
    _, neighbours_indices_xy, _ = kdtree.query([multi_dim_point[:2]], k=num_neighbors)
    neighbours_indices_xy = neighbours_indices_xy[0]

    sigma = multi_dim_point[2]
    sigma_idx = int(np.argmin(np.abs(sigma_values - sigma)))
    sigma_neighbour_indices = np.arange(max(0, sigma_idx - 1), min(len(sigma_values), sigma_idx + 2))

    n_sigma = len(sigma_values)
    kept = []
    for xy_flat_idx in neighbours_indices_xy:
        for s_idx in sigma_neighbour_indices:
            flat_idx = int(xy_flat_idx) * n_sigma + int(s_idx)
            if flat_idx in validated_set:
                kept.append(flat_idx)
    return np.array(kept, dtype=np.int64)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

def _build_prf_space(validate):
    """Small grid. radius=9.2 against x in [-8,8], y in [-9,9] drops corners => mask NOT all-True."""
    x_values = np.linspace(-8, +8, 15, dtype=np.float64)
    y_values = np.linspace(-9, +9, 15, dtype=np.float64)
    sigma_values = np.linspace(0.5, 3, 5, dtype=np.float64)
    y, x = np.meshgrid(y_values, x_values)
    points_xy = np.column_stack((y.ravel(), x.ravel()))

    prf_space = PRFSpace(points_xy, additional_dimensions=PRFSpace.make_extra_dimensions(sigma_values))
    prf_space.convert_spatial_to_multidim()

    if validate:
        prf_model = PRFGaussianModel(visual_field_radius=9.2)
        prf_space.keep_validated_sampling_points(prf_model.get_validated_sampling_points_indices)

    prf_space.compute_multidim_points_neighbours()
    return prf_space


@pytest.fixture(scope="module")
def spaces():
    return {"validated": _build_prf_space(True), "unvalidated": _build_prf_space(False)}


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_validated_case_actually_drops_points(spaces):
    """Guard: if validation dropped nothing, the mask would be all-True and the other tests hollow."""
    validated = spaces["validated"]
    unvalidated = spaces["unvalidated"]
    n_val = len(validated.multi_dim_points_cpu)
    n_full = len(unvalidated.multi_dim_points_cpu)
    assert n_val < n_full, f"validation dropped nothing ({n_val} == {n_full}); test is vacuous"


@pytest.mark.parametrize("case", ["validated", "unvalidated"])
def test_xy_row_ids_recover_each_points_own_xy(spaces, case):
    """The hoisted KD-tree query is indexed by `flat_index // prod(extra dim sizes)`.

    That must land on the row holding the point's OWN xy coordinates. A wrong mapping is already
    caught by the neighbour oracle below, but only as a diffuse "neighbours differ"; asserting the
    derivation directly localises the failure. Holds for any number of extra dimensions.
    """
    prf_space = spaces[case]
    validated = prf_space._PRFSpace__validated_multidim_indices
    product_shape_extra_dimensions = int(np.prod([len(dim) for dim in prf_space.extra_dimensions]))
    xy_row_ids = np.asarray(validated, dtype=np.int64) // product_shape_extra_dimensions

    assert np.array_equal(
        prf_space.points_xy[xy_row_ids],
        prf_space.multi_dim_points_cpu[:, :prf_space.num_spatial_dimensions],
    ), "xy_row_ids do not recover each point's own xy coordinates"


@pytest.mark.parametrize("case", ["validated", "unvalidated"])
def test_neighbour_output_alignment(spaces, case):
    """One neighbour list per validated point, aligned with multi_dim_points_cpu.

    This is the guard against conflating the validated *index array* (which selects which points
    get a list) with the *mask* (which filters candidates within a list).
    """
    prf_space = spaces[case]
    n_points = len(prf_space.multi_dim_points_cpu)
    assert len(prf_space.multi_dim_points_neighbours_flat_indices) == n_points
    assert len(prf_space.multi_dim_points_vf_neighbours) == n_points


@pytest.mark.parametrize("case", ["validated", "unvalidated"])
def test_neighbours_match_independent_reference(spaces, case):
    """The mask must admit exactly the neighbours the original predicate admitted.

    Recomputes each point's neighbour list from the inputs (not from the emitted output), so a
    neighbour wrongly dropped fails just as loudly as one wrongly kept.
    """
    prf_space = spaces[case]
    # name-mangled privates: this test deliberately reaches past the public surface
    validated_indices = prf_space._PRFSpace__validated_multidim_indices
    validated_set = set(validated_indices.tolist())
    sigma_values = prf_space.extra_dimensions[0]

    points = prf_space.multi_dim_points_cpu
    emitted_lists = prf_space.multi_dim_points_neighbours_flat_indices

    for i in range(0, len(points), 7):  # stride: full sweep is slow, this still covers edges+interior
        expected = _reference_neighbours_for_point(
            points[i], prf_space.kdtree, prf_space.points_xy, sigma_values, validated_set)
        emitted = np.ravel(emitted_lists[i])
        assert np.array_equal(emitted, expected), (
            f"point {i} (x={points[i][0]:.2f}, y={points[i][1]:.2f}, s={points[i][2]:.2f}): "
            f"emitted {emitted.tolist()} != expected {expected.tolist()}")


@pytest.mark.parametrize("case", ["validated", "unvalidated"])
def test_flat_indices_and_values_are_row_aligned(spaces, case):
    """Per point, the index list and the value list must have the same number of rows."""
    prf_space = spaces[case]
    for i, (idx, val) in enumerate(zip(prf_space.multi_dim_points_neighbours_flat_indices,
                                       prf_space.multi_dim_points_vf_neighbours)):
        assert idx.shape[0] == val.shape[0], f"point {i}: {idx.shape[0]} indices vs {val.shape[0]} values"


@pytest.mark.parametrize("case", ["validated", "unvalidated"])
def test_pinv_bit_exact_against_reference(spaces, case):
    """The prange/padded-buffer pinv must reproduce the serial reference bit for bit."""
    prf_space = spaces[case]
    points = prf_space.multi_dim_points_cpu
    neighbours = prf_space.multi_dim_points_vf_neighbours

    actual = CoefficientMatix.Wrapper_Grids2MpInv_numba(points, neighbours)

    typed_neighbours = numba.typed.List()
    for a in neighbours:
        typed_neighbours.append(np.ascontiguousarray(a))
    expected = _reference_Grids2MpInv(points, typed_neighbours)

    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.shape == e.shape, f"point {i}: shape {a.shape} != {e.shape}"
        # exact: pinv is deterministic and per-point independent, so prange must not perturb it
        assert np.array_equal(a, e), f"point {i}: max abs diff {np.abs(a - e).max()}"


@pytest.mark.parametrize("case", ["validated", "unvalidated"])
def test_pinv_results_are_independent_arrays(spaces, case):
    """Slices must be copies, not views into the dense padded buffer (which would pin ~10 GB)."""
    prf_space = spaces[case]
    result = CoefficientMatix.Wrapper_Grids2MpInv_numba(
        prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_vf_neighbours)
    assert all(a.base is None for a in result), "results are views into the padded buffer"


@pytest.mark.parametrize("case", ["validated", "unvalidated"])
def test_pinv_shape_follows_neighbour_count(spaces, case):
    """pinv(M) is (10, 4*num_neighbours); ragged edges must not be silently padded."""
    prf_space = spaces[case]
    result = CoefficientMatix.Wrapper_Grids2MpInv_numba(
        prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_vf_neighbours)
    for i, (inv_M, neigh) in enumerate(zip(result, prf_space.multi_dim_points_vf_neighbours)):
        assert inv_M.shape == (10, 4 * len(neigh)), f"point {i}: {inv_M.shape} for {len(neigh)} neighbours"
