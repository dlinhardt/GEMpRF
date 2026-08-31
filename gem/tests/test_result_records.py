"""The estimate record that both result writers unpack, and where rounding is allowed to happen.

`JsonMgr.args2estimate_record` is the single place where a fitted vertex becomes a record; the JSON writer
serialises it and `ResultFileWriter.write_h5` unpacks the same dicts into HDF5 datasets. Two things
have gone wrong at that junction and both are pinned here:

  * sigma reached the file signed. The Gaussian is even in sigma, so the refined fit is free to land
    on the negative branch and a negative pRF size is simply wrong output.
  * the record was rounded to 4 decimals. Its old name, `args2jsonEntry`, read as JSON-only, but
    the HDF5 writer consumes the same dicts, so the precise format was quantised to 1e-4 -- much
    coarser than the float32 it stores. Rounding now happens in `write_json` alone.

No CuPy involved -- these run anywhere.
"""
import json
import os

import numpy as np
import pytest

from gem.tools.json_file_operations import JsonMgr
from gem.tools.result_file_writer import ResultFileWriter


SIGNAL = np.arange(4, dtype=float)
# deliberately more decimals than the JSON writer keeps
PRECISE_X, PRECISE_Y, PRECISE_R2 = 3.14159265358979, 2.718281828459045, 0.6180339887498949
PRECISE_SIGMA = 1.4142135623730951


def _entry(sigma, muX=3.0, muY=2.0, r2=0.5):
    return JsonMgr.args2estimate_record(muX=muX, muY=muY, sigma=sigma, r2=r2, signal=SIGNAL)


@pytest.mark.parametrize("sigma", [-0.7795, -0.5469, -0.0666, -5.0])
def test_negative_sigma_is_stored_as_a_magnitude(sigma):
    """A pRF size must never be written out negative -- the sign carries no information."""
    assert _entry(sigma)["sigmaMajor"] == abs(sigma)


@pytest.mark.parametrize("sigma", [0.0, 0.5, 1.5584, 5.0])
def test_positive_sigma_is_untouched(sigma):
    """The fix must not perturb the values that were already fine."""
    assert _entry(sigma)["sigmaMajor"] == sigma


def test_sigma_signs_describe_the_same_prf():
    """+s and -s are the same model, so they must produce the same record."""
    assert _entry(-1.2345) == _entry(1.2345)


def test_the_record_keeps_full_precision():
    """The record is what HDF5 gets, so nothing may be rounded away on the way in."""
    entry = _entry(PRECISE_SIGMA, muX=PRECISE_X, muY=PRECISE_Y, r2=PRECISE_R2)

    assert entry["Centerx0"] == PRECISE_X
    assert entry["Centery0"] == PRECISE_Y
    assert entry["sigmaMajor"] == PRECISE_SIGMA
    assert entry["R2"] == PRECISE_R2


def test_negative_centres_are_kept():
    """Only sigma is sign-free: a pRF at negative x/y is an ordinary left/lower-field pRF."""
    entry = _entry(1.0, muX=-3.0, muY=-2.0)

    assert entry["Centerx0"] == -3.0
    assert entry["Centery0"] == -2.0


def test_other_fields_are_unaffected():
    """Centre, R2 and the predicted signal pass through untouched."""
    entry = _entry(-1.5)

    assert entry["Theta"] == 0
    assert entry["sigmaMinor"] == 0
    assert entry["R2"] == 0.5
    assert entry["modelpred"] == SIGNAL.tolist()


def test_json_writer_rounds(tmp_path):
    """JSON stays a readable dump, so it is the one place that rounds."""
    filepath = os.path.join(tmp_path, "estimates.json")
    ResultFileWriter.write_json(filepath, [_entry(PRECISE_SIGMA, muX=PRECISE_X, muY=PRECISE_Y, r2=PRECISE_R2)])

    with open(filepath) as handle:
        written = json.load(handle)[0]

    assert written["Centerx0"] == round(PRECISE_X, ResultFileWriter.JSON_DECIMALS)
    assert written["sigmaMajor"] == round(PRECISE_SIGMA, ResultFileWriter.JSON_DECIMALS)
    assert written["R2"] == round(PRECISE_R2, ResultFileWriter.JSON_DECIMALS)


def test_json_rounding_does_not_mutate_the_caller_records(tmp_path):
    """write_json must round a copy -- the same list is handed to the HDF5 writer in other runs."""
    record = _entry(PRECISE_SIGMA, muX=PRECISE_X)
    ResultFileWriter.write_json(os.path.join(tmp_path, "estimates.json"), [record])

    assert record["Centerx0"] == PRECISE_X
    assert record["sigmaMajor"] == PRECISE_SIGMA


def test_json_writer_handles_a_missing_modelpred(tmp_path):
    """modelpred is dropped in some runs; rounding must not trip over the None."""
    record = _entry(1.0)
    record["modelpred"] = None
    ResultFileWriter.write_json(os.path.join(tmp_path, "estimates.json"), [record])

    with open(os.path.join(tmp_path, "estimates.json")) as handle:
        assert json.load(handle)[0]["modelpred"] is None


def test_json_writer_handles_a_none_inside_the_modelpred_list(tmp_path):
    """The real shape of a dropped timecourse is [None], not None.

    build_estimate_records() writes np.array([None]).tolist() when the refined signals are not kept,
    so the container is a list and only its elements are None. Checking the container alone let
    float(None) through and killed the whole analysis -- but only on the JSON path, because write_h5
    tests modelpred_list[0] and skips the dataset entirely.
    """
    record = _entry(1.0)
    record["modelpred"] = [None]
    ResultFileWriter.write_json(os.path.join(tmp_path, "estimates.json"), [record])

    with open(os.path.join(tmp_path, "estimates.json")) as handle:
        assert json.load(handle)[0]["modelpred"] == [None]


def test_json_writer_rounds_a_real_modelpred(tmp_path):
    """A timecourse that is present still gets rounded element by element."""
    record = _entry(1.0)
    record["modelpred"] = [PRECISE_X, None, PRECISE_R2]
    ResultFileWriter.write_json(os.path.join(tmp_path, "estimates.json"), [record])

    with open(os.path.join(tmp_path, "estimates.json")) as handle:
        written = json.load(handle)[0]["modelpred"]

    assert written == [round(PRECISE_X, ResultFileWriter.JSON_DECIMALS), None,
                       round(PRECISE_R2, ResultFileWriter.JSON_DECIMALS)]


# --------------------------------------------------------------------------------------
# The HDF5 writer takes arrays; only the JSON writer sees per-vertex dicts
# --------------------------------------------------------------------------------------

def _stub_cfg():
    """The minimum ResultFileWriter.write_h5 reads off a config."""
    from types import SimpleNamespace
    return SimpleNamespace(
        results={'output_format': 'hdf5'},
        pRF_model_details={'model': '2d_gaussian'},
        refine_fitting_enabled=True,
        nDCT=3,
        write_debug_info=False,
        default_spatial_grid={'visual_field_radius': 15, 'num_horizontal_prfs': 201, 'num_vertical_prfs': 201},
        default_sigmas={'num_sigmas': 30, 'min_sigma': 0.1, 'max_sigma': 5},
        stimulus={'visual_field': 10, 'width': 301, 'height': 301,
                  'binarization': {'@enable': 'False', '@threshold': 0}},
        default_hrf={'TR': None, 't': (0.0, 45.0), 'peak_delay': 6.16, 'under_shoot_delay': 12.0,
                     'peak_disp': 1.0, 'under_disp': 1.0, 'peak_to_undershoot': 6.0, 'normalize': True},
    )


def test_h5_writer_stores_the_fit_arrays_unchanged():
    """write_h5 reads columns straight off the fit's arrays -- no dict round trip in between."""
    h5py = pytest.importorskip("h5py")
    import tempfile

    params_xy = np.array([[1.5, -2.5, 0.75], [-0.125, 3.25, 2.5], [0.0, 0.0, 0.1]])
    r2 = np.array([0.9, 0.4, -2.0])

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub-01", "estimates.h5")
        ResultFileWriter.write_h5(path, params_xy, r2, _stub_cfg(), ["/in.nii.gz"], "/stim", "individual", 12.5)
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["parameters/Centerx0"][...], params_xy[:, 0].astype(np.float32))
            np.testing.assert_array_equal(f["parameters/Centery0"][...], params_xy[:, 1].astype(np.float32))
            np.testing.assert_array_equal(f["parameters/sigmaMajor"][...], params_xy[:, 2].astype(np.float32))
            np.testing.assert_array_equal(f["parameters/R2"][...], r2.astype(np.float32))
            # isotropic Gaussian: the two placeholder fields exist so both formats describe the same thing
            assert not f["parameters/Theta"][...].any()
            assert not f["parameters/sigmaMinor"][...].any()
            assert "modelpred" not in f["parameters"]


def test_h5_writer_stores_sigma_as_a_magnitude():
    """apply_grid_fallback normalises the sign, but the writer must not depend on that having run."""
    h5py = pytest.importorskip("h5py")
    import tempfile

    params_xy = np.array([[1.0, 1.0, -0.75], [1.0, 1.0, 0.75]])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "estimates.h5")
        ResultFileWriter.write_h5(path, params_xy, np.array([0.5, 0.5]), _stub_cfg(),
                                  ["/in.nii.gz"], "/stim", "individual", 1.0)
        with h5py.File(path, "r") as f:
            sigma = f["parameters/sigmaMajor"][...]
    assert sigma[0] == sigma[1] == np.float32(0.75), sigma


def test_h5_and_json_describe_the_same_vertex():
    """The two formats diverge in representation only -- the values behind them must agree."""
    h5py = pytest.importorskip("h5py")
    import tempfile

    params_xy = np.array([[PRECISE_X, PRECISE_Y, -PRECISE_SIGMA]])
    r2 = np.array([PRECISE_R2])
    records = ResultFileWriter.build_estimate_records(params_xy, r2)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "estimates.h5")
        ResultFileWriter.write_h5(path, params_xy, r2, _stub_cfg(), ["/in.nii.gz"], "/stim", "individual", 1.0)
        with h5py.File(path, "r") as f:
            assert f["parameters/Centerx0"][0] == np.float32(records[0]["Centerx0"])
            assert f["parameters/sigmaMajor"][0] == np.float32(records[0]["sigmaMajor"])
            assert f["parameters/R2"][0] == np.float32(records[0]["R2"])


@pytest.mark.parametrize("output_format,extension", [("hdf5", ".h5"), ("json", ".json")])
def test_write_dispatches_on_the_configured_format(output_format, extension):
    """write() is the only entry point the pipeline uses; both branches must accept the fit's arrays.

    The JSON branch is where the per-vertex dicts get built -- the HDF5 branch never sees them. This
    is the seam that broke once already: write() takes host arrays, and the fit hands back CuPy ones
    whenever refinement ran on the GPU, so the conversion belongs at the call site.
    """
    import tempfile

    cfg = _stub_cfg()
    cfg.results = {'output_format': output_format}
    params_xy = np.array([[1.5, -2.5, -0.75], [0.25, 0.5, 1.25]])
    r2 = np.array([0.9, 0.4])

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "sub-01", "estimates")
        ResultFileWriter.write(filepath=base + ".h5", params_xy=params_xy, r2=r2, cfg=cfg,
                               input_filepaths=["/in.nii.gz"], stimulus_filepath="/stim",
                               run_type="individual", duration_sec=1.0)
        written = base + extension
        assert os.path.exists(written), f"{output_format} branch wrote nothing at {written}"
        assert ResultFileWriter.result_exists(written)

        if output_format == "json":
            with open(written) as handle:
                records = json.load(handle)
            assert len(records) == 2
            assert records[0]["sigmaMajor"] == 0.75  # magnitude, rounded for JSON
