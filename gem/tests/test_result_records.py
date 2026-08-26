"""The estimate record that both result writers unpack, and where rounding is allowed to happen.

`JsonMgr.args2jsonEntry` is the single place where a fitted vertex becomes a record; the JSON writer
serialises it and `ResultFileWriter.write_h5` unpacks the same dicts into HDF5 datasets. Two things
have gone wrong at that junction and both are pinned here:

  * sigma reached the file signed. The Gaussian is even in sigma, so the refined fit is free to land
    on the negative branch and a negative pRF size is simply wrong output.
  * the record was rounded to 4 decimals. The name `args2jsonEntry` suggests that only concerns
    JSON, but the HDF5 writer consumes the same dicts, so the precise format was quantised to 1e-4
    -- much coarser than the float32 it stores. Rounding now happens in `write_json` alone.

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
    return JsonMgr.args2jsonEntry(muX=muX, muY=muY, sigma=sigma, r2=r2, signal=SIGNAL)


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
