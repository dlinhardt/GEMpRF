"""The refinement's padded lookup tables must equal what the old lazy rebuild produced.

`RefineFit._prepare_padded_arrays` used to reconstruct, on the first refined fit of a run, the exact
`(num_model_signals, 10, max_cols)` array that `Wrapper_Grids2MpInv_numba` had already built and then
thrown away. On a 942,030-point grid that rebuild walked ~9.4 million Python-level slices and faulted
in ~8 GB of fresh pages, and because it fired lazily inside the batch loop the whole cost landed on
the first file's clock -- one observed run spent 736 s on file 1 and 143 s on every file after it.

The kernel's buffer is now kept and read directly. Two things have to hold for that to be safe, and
both are pinned here:

  1. The kept buffer equals the old rebuild **element for element** after `nan_to_num`. The two pad
     differently -- the kernel with 0.0, the old rebuild with NaN -- but the only reader,
     `_compute_coefficients`, runs `nan_to_num(MpInv, nan=0.0)` before the einsum, so the padding
     value cannot reach the result.
  2. The neighbour-index table, now built by scattering a flat buffer, equals the old
     `np.concatenate` + `np.insert` construction exactly.

Run on a machine with CuPy available (gem.space.PRFSpace imports cupy at module level):

    pytest gem/tests/test_padded_lookup_tables.py -v
"""
import numpy as np
import pytest

from gem.fitting.hpc_refine_fit import RefineFit
from gem.model.prf_gaussian_model import PRFGaussianModel
from gem.space.PRFSpace import PRFSpace
from gem.space.coefficient_matrix import CoefficientMatix


# --------------------------------------------------------------------------------------
# Pinned reference implementations (verbatim, from the deleted _prepare_padded_arrays body)
# --------------------------------------------------------------------------------------

def _reference_padded_mpinv(arr_2d_location_inv_M_cpu_list):
    """Verbatim pre-change section 1a: concatenate per row, np.insert NaN at the pad positions."""
    N = len(arr_2d_location_inv_M_cpu_list)
    R = arr_2d_location_inv_M_cpu_list[0].shape[0]
    cols = np.array([a.shape[1] for a in arr_2d_location_inv_M_cpu_list], dtype=int)
    max_cols = int(cols.max())

    if (cols == max_cols).all():
        return np.stack(arr_2d_location_inv_M_cpu_list, axis=0)

    cumsum_cols = np.cumsum(cols)
    pad_lens = max_cols - cols
    where_to_pad = np.repeat(cumsum_cols, pad_lens) if pad_lens.sum() > 0 else np.array([], dtype=int)

    padded_rows = []
    for r in range(R):
        row_concat = np.concatenate([a[r] for a in arr_2d_location_inv_M_cpu_list])
        row_padded = np.insert(row_concat, where_to_pad, np.nan) if where_to_pad.size else row_concat
        padded_rows.append(row_padded.reshape(N, max_cols))
    return np.stack(padded_rows, axis=1)


def _reference_padded_neighbours(neigh_list):
    """Verbatim pre-change section 1b: concatenate, np.insert -1 at the pad positions."""
    neigh_list = [np.asarray(a).ravel() for a in neigh_list]
    N = len(neigh_list)
    lens = np.array([a.size for a in neigh_list], dtype=int)
    max_len = int(lens.max())

    if (lens == max_len).all():
        padded = np.stack(neigh_list, axis=0)
    else:
        cumsum_lens = np.cumsum(lens)
        pad_lens = max_len - lens
        where_to_pad = np.repeat(cumsum_lens, pad_lens) if pad_lens.sum() > 0 else np.array([], dtype=int)
        all_concat = np.concatenate(neigh_list)
        padded = np.insert(all_concat, where_to_pad, -1) if where_to_pad.size else all_concat
        padded = padded.reshape(N, max_len)

    return padded.astype(np.int64)[:, :, None]


# --------------------------------------------------------------------------------------
# Fixture: a small grid whose corners are dropped, so the neighbour lists really are ragged
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prf_space():
    x_values = np.linspace(-8, +8, 15, dtype=np.float64)
    y_values = np.linspace(-9, +9, 15, dtype=np.float64)
    sigma_values = np.linspace(0.5, 3, 5, dtype=np.float64)
    y, x = np.meshgrid(y_values, x_values)
    points_xy = np.column_stack((y.ravel(), x.ravel()))

    space = PRFSpace(points_xy, additional_dimensions=PRFSpace.make_extra_dimensions(sigma_values))
    space.convert_spatial_to_multidim()
    space.keep_validated_sampling_points(PRFGaussianModel(visual_field_radius=9.2).get_validated_sampling_points_indices)
    space.compute_multidim_points_neighbours()
    return space


@pytest.fixture(scope="module")
def reference_tables(prf_space):
    """Both reference tables, captured BEFORE anything drops the per-point neighbour list."""
    table = CoefficientMatix.Wrapper_Grids2MpInv_numba(
        prf_space.multi_dim_points_cpu, prf_space.multi_dim_points_vf_neighbours)
    ragged = [np.array(table[i]) for i in range(len(table))]  # real copies, trimmed to num_cols
    return {
        "mpinv_table": table,
        "ragged": ragged,
        "old_mpinv": _reference_padded_mpinv(ragged),
        "old_neighbours": _reference_padded_neighbours(list(prf_space.multi_dim_points_neighbours_flat_indices)),
        "original_neighbour_lists": [np.asarray(a).ravel().copy()
                                     for a in prf_space.multi_dim_points_neighbours_flat_indices],
    }


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_grid_is_actually_ragged(reference_tables):
    """Guard: if every point had the same neighbour count, the padding paths would never run."""
    widths = {a.shape[1] for a in reference_tables["ragged"]}
    assert len(widths) > 1, f"all pinv widths identical ({widths}); the padding tests are vacuous"


def test_kept_buffer_equals_old_rebuild_after_masking(prf_space, reference_tables):
    """The whole point of the change: the kernel's buffer IS the old rebuild, once NaN -> 0."""
    RefineFit._prepare_padded_arrays(reference_tables["mpinv_table"], prf_space)
    new = RefineFit.padded_arr_2d_location_inv_M
    old = reference_tables["old_mpinv"]

    assert new.shape == old.shape
    assert np.array_equal(np.nan_to_num(old, nan=0.0), np.nan_to_num(new, nan=0.0))


def test_only_the_padding_region_differs(reference_tables):
    """Sharpen the above: inside num_cols the two are already identical, no masking needed."""
    old = reference_tables["old_mpinv"]
    table = reference_tables["mpinv_table"]

    for i in range(len(table)):
        width = table.num_cols[i]
        assert np.array_equal(old[i, :, :width], table.padded[i, :, :width]), f"point {i} differs"
        assert np.isnan(old[i, :, width:]).all(), f"point {i}: old padding was not NaN"
        assert np.array_equal(table.padded[i, :, width:],
                              np.zeros_like(table.padded[i, :, width:])), f"point {i}: new padding is not 0"


def test_neighbour_table_matches_old_construction(prf_space, reference_tables):
    """The scatter-into-preallocated build must equal concatenate + np.insert exactly."""
    RefineFit._prepare_padded_arrays(reference_tables["mpinv_table"], prf_space)
    new = RefineFit.padded_multi_dim_points_neighbours_flat_indices
    old = reference_tables["old_neighbours"]

    assert new.shape == old.shape
    assert new.dtype == old.dtype
    assert np.array_equal(new, old)


def test_array_split_round_trips_the_neighbour_lists(prf_space, reference_tables):
    """After the flat form is built the per-point list is dropped; the property must rebuild it."""
    prf_space.get_neighbour_flat_indices_and_counts()  # drops the per-point list
    rebuilt = prf_space.multi_dim_points_neighbours_flat_indices
    original = reference_tables["original_neighbour_lists"]

    assert len(rebuilt) == len(original)
    for i, (got, want) in enumerate(zip(rebuilt, original)):
        assert np.array_equal(got, want), f"point {i}: {got} != {want}"


def test_flat_form_is_consistent_with_the_counts(prf_space):
    """The counts must partition the flat buffer exactly -- no lost or duplicated neighbours."""
    flat, counts = prf_space.get_neighbour_flat_indices_and_counts()
    assert counts.sum() == flat.size
    assert len(counts) == len(prf_space.multi_dim_points_cpu)


def test_ragged_entries_are_views_not_copies(reference_tables):
    """MpInvTable keeps the old ragged indexing contract, but without duplicating the buffer."""
    table = reference_tables["mpinv_table"]
    assert all(table[i].base is not None for i in range(len(table)))
    assert all(table[i].shape == (10, table.num_cols[i]) for i in range(len(table)))


def test_prepare_is_idempotent(prf_space, reference_tables):
    """Setup calls prepare(); the lazy guards may still fire in tests. Both must agree."""
    RefineFit._prepare_padded_arrays(reference_tables["mpinv_table"], prf_space)
    first = RefineFit.padded_arr_2d_location_inv_M.copy()
    RefineFit.prepare(reference_tables["mpinv_table"], prf_space)
    assert np.array_equal(first, RefineFit.padded_arr_2d_location_inv_M)
