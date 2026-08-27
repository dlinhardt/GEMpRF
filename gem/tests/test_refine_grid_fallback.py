"""Unit tests for the refined-fit -> grid fallback and its supporting pieces.

When a refined pRF estimate is not trusted -- it moved more than two grid steps away in x, y or
sigma, its parameters came out NaN, or (individual-run path) the refined error got worse -- the
vertex is reverted *completely* to its coarse grid point. These tests pin that behaviour and, most
importantly, the invariance property: a vertex that triggers none of the new reasons is left exactly
as the original "worse error" revert left it.

Covered here:
  * PRFSpace.get_grid_steps        -- per-dimension grid spacing (+ caching)
  * GEMpRFAnalysis.apply_grid_fallback -- the revert reasons and the reported stats
  * ... its invariance             -- new reasons off => byte-identical to the historical revert
  * RunReport.add_grid_fallback    -- the report section renders (no CuPy needed)

`gem.space.PRFSpace` and `gem.run.run_gem_prf_analysis` import CuPy at module load, so those imports
are done lazily inside the tests that need them; the RunReport test stays import-light on purpose.
"""
import numpy as np

# grid steps used throughout the fallback tests: x/y share a step, sigma is coarser.
GRID_STEPS = np.array([0.18, 0.18, 0.30])


def _as_str(value):
    """h5py returns str datasets as bytes on read; normalise to str for comparison."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _build_prf_space(x_values, y_values, sigma_values):
    """A small, fully-built PRFSpace over the given 1-D generators (CPU only)."""
    from gem.space.PRFSpace import PRFSpace

    y, x = np.meshgrid(y_values, x_values)
    points_xy = np.column_stack((y.ravel(), x.ravel()))
    prf_space = PRFSpace(points_xy, additional_dimensions=PRFSpace.make_extra_dimensions(sigma_values))
    prf_space.convert_spatial_to_multidim()
    return prf_space


def test_get_grid_steps():
    """Spacing equals the linspace step of each dimension, is cached, and is 0 for a lone value."""
    x_values = np.linspace(-8, +8, 15)      # step = 16 / 14
    y_values = np.linspace(-9, +9, 15)      # step = 18 / 14
    sigma_values = np.linspace(0.5, 3, 5)   # step = 2.5 / 4 = 0.625
    prf_space = _build_prf_space(x_values, y_values, sigma_values)

    steps = prf_space.get_grid_steps()

    # x and y occupy the first two columns (order depends on the meshgrid), sigma is last.
    assert np.allclose(sorted(steps[:2]), sorted([16.0 / 14.0, 18.0 / 14.0]))
    assert np.isclose(steps[2], 0.625)

    # cached: the same array object is returned on subsequent calls
    assert prf_space.get_grid_steps() is steps

    # a dimension with a single unique value has no meaningful spacing -> 0
    single_sigma_space = _build_prf_space(x_values, y_values, np.array([1.0]))
    assert single_sigma_space.get_grid_steps()[2] == 0.0


def test_apply_grid_fallback():
    """Every revert reason fires on its own row; the in-range row keeps its refined value."""
    from gem.run.run_gem_prf_analysis import GEMpRFAnalysis

    # "too far" is scaled off the actual threshold so the test survives future factor changes.
    far = (GEMpRFAnalysis.MAX_GRID_STEPS_AWAY + 1) * GRID_STEPS  # per-dimension over-the-line offset
    cx, cy, cs = 0.2, 0.1, 1.1
    in_range = [cx + 0.5 * GRID_STEPS[0], cy + 0.5 * GRID_STEPS[1], cs + 0.5 * GRID_STEPS[2]]

    coarse = np.array([[cx, cy, cs]] * 5)
    refined = np.array([
        in_range,                        # 0: within the threshold everywhere -> keep refined
        [cx + far[0], cy, cs],           # 1: x moved too far                 -> revert
        [np.nan] * 3,                    # 2: NaN params                      -> revert
        [cx, cy, cs + far[2]],           # 3: sigma moved too far             -> revert
        [cx + 0.1 * GRID_STEPS[0], cy, cs],  # 4: in range but flagged worse  -> revert
    ])
    worse_error_mask = np.array([False, False, False, False, True])

    out, stats, records = GEMpRFAnalysis.apply_grid_fallback(
        refined.copy(), coarse, GRID_STEPS, worse_error_mask=worse_error_mask)

    assert np.allclose(out[0], in_range)                    # kept
    assert np.allclose(out[1:], coarse[1:])                 # all others reverted to the grid point

    assert stats == {
        "total": 5,
        "worse_error": 1,
        "nan_refined": 1,
        "x_too_far": 1,
        "y_too_far": 0,
        "sigma_too_far": 1,
        "zero_signal": 0,
        "on_grid": 4,        # distinct reverted vertices (rows 1-4)
    }

    # per-vertex records: a full-length reason bitmask + the ORIGINAL (pre-revert) refined params.
    bits = GEMpRFAnalysis.FALLBACK_REASON_BITS
    reason = records["reason"]
    assert reason.dtype == np.uint8
    assert reason[0] == 0                                    # row 0 kept -> no reason bit
    assert reason[1] == bits["x_too_far"]
    assert reason[2] == bits["nan_refined"]
    assert reason[3] == bits["sigma_too_far"]
    assert reason[4] == bits["worse_error"]
    # refined_pre still holds the rejected fit (row 3 sigma moved far), NOT the reverted grid value
    assert np.isclose(records["refined_pre"][3, 2], cs + far[2])
    assert np.array_equal(np.isnan(records["refined_pre"][2]), np.array([True, True, True]))


def test_apply_grid_fallback_invariance():
    """With none of the new reasons triggered, the result is exactly the historical revert."""
    from gem.run.run_gem_prf_analysis import GEMpRFAnalysis

    coarse = np.array([[0.2, 0.1, 1.1]] * 3)
    refined = np.array([
        [0.25, 0.15, 1.20],   # in range
        [0.28, 0.05, 1.00],   # in range, flagged worse
        [0.15, 0.12, 1.25],   # in range
    ])
    worse_error_mask = np.array([False, True, False])

    # historical behaviour: revert only the worse-error rows, nothing else
    expected = refined.copy()
    idx = np.argwhere(worse_error_mask)
    expected[idx, :] = coarse[idx, :]

    out, _, _ = GEMpRFAnalysis.apply_grid_fallback(
        refined.copy(), coarse, GRID_STEPS, worse_error_mask=worse_error_mask)

    assert np.array_equal(out, expected)
    assert np.allclose(out[0], refined[0])   # untriggered rows untouched
    assert np.allclose(out[2], refined[2])


def test_select_and_finalize_fallback_records():
    """Batch-local records get the zero-signal reason merged, the batch offset added, then packed."""
    from gem.run.run_gem_prf_analysis import GEMpRFAnalysis

    bits = GEMpRFAnalysis.FALLBACK_REASON_BITS

    # one batch of 4 vertices: vertex 1 reverted (x_too_far), vertex 3 has NaN params
    reason = np.zeros(4, dtype=np.uint8)
    reason[1] = bits["x_too_far"]
    reason[3] = bits["nan_refined"]
    refined_pre = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    records = {"reason": reason.copy(), "refined_pre": refined_pre}

    # R2 == -2 on vertices 2 (new) and 3 (merges with the existing nan reason)
    r2_batch = np.array([[0.5], [0.3], [-2.0], [-2.0]])
    idx, rsn, ref = GEMpRFAnalysis._select_fallback_records(records, offset=100, r2_batch=r2_batch)

    assert list(idx) == [101, 102, 103]                                  # offset applied, vertex 0 dropped
    assert rsn[0] == bits["x_too_far"]                                   # vertex 1: unchanged
    assert rsn[1] == bits["zero_signal"]                                 # vertex 2: zero-signal only
    assert rsn[2] == (bits["nan_refined"] | bits["zero_signal"])         # vertex 3: merged
    assert np.array_equal(ref[2], refined_pre[3])                        # refined params carried through
    assert records["reason"][3] == bits["nan_refined"]                  # source array not mutated

    # without r2_batch (concatenated path): no zero-signal bit, only the pre-flagged vertices
    idx2, rsn2, _ = GEMpRFAnalysis._select_fallback_records(records, offset=0)
    assert list(idx2) == [1, 3]
    assert list(rsn2) == [bits["x_too_far"], bits["nan_refined"]]

    packed = GEMpRFAnalysis._finalize_fallback_records([idx], [rsn], [ref], num_params=3)
    assert packed["vertex_index"].dtype == np.int32 and list(packed["vertex_index"]) == [101, 102, 103]
    assert packed["reason"].dtype == np.uint8
    assert packed["refined_params"].shape == (3, 3)
    assert packed["reason_bits"] == bits and packed["param_columns"] == "Centerx0,Centery0,sigmaMajor"

    # empty run -> well-formed empty arrays with the right shapes/dtypes (no records at all)
    empty = GEMpRFAnalysis._finalize_fallback_records([], [], [], num_params=3)
    assert empty["vertex_index"].shape == (0,) and empty["refined_params"].shape == (0, 3)


def test_write_grid_fallback_group(tmp_path):
    """The /grid_fallback group round-trips (data + legend), and is absent when there is nothing."""
    import h5py
    from gem.tools.result_file_writer import ResultFileWriter

    records = {
        "vertex_index": np.array([5, 42], dtype=np.int32),
        "reason": np.array([4, 2 | 32], dtype=np.uint8),      # x_too_far ; nan+zero_signal
        "refined_params": np.array([[1.5, -2.0, 0.8], [np.nan, np.nan, np.nan]], dtype=np.float32),
        "reason_bits": {"nan_refined": 2, "x_too_far": 4, "zero_signal": 32},
        "param_columns": "Centerx0,Centery0,sigmaMajor",
    }

    path = tmp_path / "with_records.h5"
    with h5py.File(path, "w") as f:
        ResultFileWriter._write_grid_fallback(f, records)
    with h5py.File(path, "r") as f:
        g = f["grid_fallback"]
        assert list(g["vertex_index"][()]) == [5, 42]
        assert g["vertex_index"].dtype == np.int32
        assert g["reason"].dtype == np.uint8 and list(g["reason"][()]) == [4, 34]
        assert g["refined_params"].shape == (2, 3)
        # legend is stored purely as datasets (no group attributes)
        assert not dict(g.attrs)
        assert "4=x_too_far" in _as_str(g["reason_legend"][()])
        names = [_as_str(n) for n in g["reason_bit_names"][()]]
        values = list(g["reason_bit_values"][()])
        assert dict(zip(names, values)) == {"nan_refined": 2, "x_too_far": 4, "zero_signal": 32}
        assert _as_str(g["param_columns"][()]) == "Centerx0,Centery0,sigmaMajor"

    # None and empty -> no group written at all (files with refine fitting off / nothing rejected)
    for empty_records in (None, {"vertex_index": np.empty(0, dtype=np.int32)}):
        p = tmp_path / f"none_{id(empty_records)}.h5"
        with h5py.File(p, "w") as f:
            ResultFileWriter._write_grid_fallback(f, empty_records)
        with h5py.File(p, "r") as f:
            assert "grid_fallback" not in f


def test_run_report_grid_fallback_section():
    """The run report gains an 'On-grid fallbacks' section only once fallbacks are recorded."""
    from gem.utils.run_report import RunReport

    report = RunReport(run_type="individual", gemprf_version="test",
                       config_filepath="cfg.xml", result_dir="/tmp/gemprf-test")

    assert "On-grid fallbacks" not in report.render()

    report.add_grid_fallback("sub-01_estimates.h5", {
        "total": 200, "on_grid": 7, "worse_error": 3,
        "x_too_far": 2, "y_too_far": 1, "sigma_too_far": 1,
        "nan_refined": 0, "zero_signal": 5,
    })
    rendered = report.render()

    assert "On-grid fallbacks" in rendered
    assert "reverted to grid" in rendered
    assert "sub-01_estimates.h5" in rendered
    assert "zero model signal (R2=-2): 5" in rendered


def test_sigma_is_compared_as_a_magnitude():
    """A refinement that came back with negative sigma is the same pRF, and must be judged as one.

    The Gaussian is even in sigma, so +s and -s produce an identical model signal, error and R2. The
    distance check used to measure |refined - coarse| across the sign, which inflated the distance by
    2|sigma| and made the verdict depend on how large sigma happened to be -- identical models were
    reverted or kept depending on their magnitude.
    """
    from gem.run.run_gem_prf_analysis import GEMpRFAnalysis

    sigma_step = GRID_STEPS[2]
    limit = GEMpRFAnalysis.MAX_GRID_STEPS_AWAY * sigma_step
    coarse_sigma = 3.0
    # |refined| sits well inside the limit, but the signed difference is far outside it
    refined_sigma = -(coarse_sigma + 0.25 * limit)
    assert abs(refined_sigma - coarse_sigma) > limit
    assert abs(abs(refined_sigma) - coarse_sigma) < limit

    coarse = np.array([[0.2, 0.1, coarse_sigma]])
    refined = np.array([[0.2, 0.1, refined_sigma]])

    kept, stats, _ = GEMpRFAnalysis.apply_grid_fallback(refined.copy(), coarse, GRID_STEPS)

    assert stats["sigma_too_far"] == 0, "the sign must not count as distance"
    assert stats["on_grid"] == 0, "the refinement is close enough to keep"
    assert kept[0, 2] == abs(refined_sigma), "sigma must be reported as a magnitude"


def test_opposite_sigma_signs_give_the_same_verdict():
    """The core invariant: +s and -s are the same pRF, so the fallback must not tell them apart."""
    from gem.run.run_gem_prf_analysis import GEMpRFAnalysis

    coarse = np.array([[0.2, 0.1, 1.1], [0.2, 0.1, 1.1]])
    for magnitude in (0.4, 1.0, 1.3):
        positive = np.array([[0.25, 0.15, magnitude], [0.25, 0.15, magnitude]])
        negative = np.array([[0.25, 0.15, -magnitude], [0.25, 0.15, -magnitude]])

        kept_pos, stats_pos, _ = GEMpRFAnalysis.apply_grid_fallback(positive, coarse, GRID_STEPS)
        kept_neg, stats_neg, _ = GEMpRFAnalysis.apply_grid_fallback(negative, coarse, GRID_STEPS)

        np.testing.assert_array_equal(kept_pos, kept_neg)
        assert stats_pos == stats_neg


def test_no_negative_sigma_survives_the_fallback():
    """Whatever the verdict, a negative pRF size must never come out the other side."""
    from gem.run.run_gem_prf_analysis import GEMpRFAnalysis

    coarse = np.array([[0.2, 0.1, 1.1]] * 3)
    refined = np.array([[0.2, 0.1, -0.05],      # near zero, kept
                        [0.2, 0.1, -1.2],       # comparable to the grid point, kept
                        [0.2, 0.1, -40.0]])     # genuinely far, reverted

    kept, stats, _ = GEMpRFAnalysis.apply_grid_fallback(refined, coarse, GRID_STEPS)

    assert (kept[:, 2] >= 0).all()
    assert stats["sigma_too_far"] == 1
