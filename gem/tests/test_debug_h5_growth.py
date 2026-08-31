"""The debug HDF5 file must not accumulate unreachable data across runs.

`debug_model_data.h5` on the server reached **99.33 GB holding 28 GB of live datasets**. Walking the
chunk B-tree of every object showed the entire live set sitting in the last 28 GB and the first
71.29 GB being one contiguous region that nothing referenced, and `h5stat` reported
`Amount of tracked free space: 0 bytes` against that hole with `Free-space persist: FALSE` -- so
HDF5 could not reuse it and simply appended past it on the next run.

Two things in `write_array_to_h5` caused it, and both are pinned here:

  1. Every dataset was created with `maxshape=`, which forces chunked storage -- ~12,000 chunks per
     2.9 GB array. Nothing ever passes `append_to_existing_variable=True`, so the extendability
     bought nothing and only fragmented the file. Datasets are contiguous now.
  2. The file was opened `"a"` for every write, so each run's datasets were deleted and rewritten on
     top of the previous run's. The first write of a run truncates now.

This file deliberately does not import `gem.utils.gem_write_to_file` at module scope: that module
imports CuPy, which is absent off the GPU machine. A minimal stub is installed instead, since none
of the behaviour under test touches CuPy.
"""
import importlib
import os
import sys
import types

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


@pytest.fixture()
def writer_module():
    """Import gem_write_to_file with a stubbed `cupy`, and undo the stub afterwards."""
    installed_stub = False
    if "cupy" not in sys.modules:
        try:
            importlib.import_module("cupy")
        except ImportError:
            stub = types.ModuleType("cupy")
            stub.ndarray = type("ndarray", (), {})   # nothing under test is ever an instance
            stub.asnumpy = np.asarray
            sys.modules["cupy"] = stub
            installed_stub = True
    try:
        module = importlib.import_module("gem.utils.gem_write_to_file")
        module.GemWriteToFile._instance = None       # singleton: start from a known state
        yield module
    finally:
        module = sys.modules.get("gem.utils.gem_write_to_file")
        if module is not None:
            module.GemWriteToFile._instance = None
        if installed_stub:
            del sys.modules["cupy"]


# The real write order, with the real dataset names, scaled down by ~1/1000.
_RUN_DATASETS = [
    (["stimulus", "prflong", "resampled_data"], 2079),
    (["stimulus", "prflong", "stimulus_data_hrf_convolved"], 2079),
    (["model", "", "model_signals_derivative_d0"], 2702),
    (["model", "", "model_signals_derivative_d1"], 2702),
    (["model", "", "model_signals_derivative_d2"], 2702),
    (["model", "", "model_signals"], 2702),
    (["model", "", "orthonormalized_model_signals"], 2702),
    (["model", "", "orthonormalized_model_signals_derivative_d0"], 2702),
]


def _one_run(writer_module, result_dir, scale=1):
    """Simulate a run: a fresh writer instance, then every dataset written once."""
    writer_module.GemWriteToFile._instance = None
    writer = writer_module.GemWriteToFile(result_dir=result_dir, debugging_enabled=True)
    for path, kilobytes in _RUN_DATASETS:
        writer.write_array_to_h5(np.zeros(kilobytes * scale * 1000 // 8), variable_path=path)
    return os.path.join(result_dir, "debug_model_data.h5")


def test_file_does_not_grow_across_runs(writer_module, tmp_path):
    """Five identical runs into one result dir must leave the file the size of one run."""
    sizes = [os.path.getsize(_one_run(writer_module, str(tmp_path))) for _ in range(5)]
    assert sizes[0] == pytest.approx(sizes[-1], rel=0.01), f"file grew across runs: {sizes}"


def test_file_does_not_grow_when_dataset_sizes_change(writer_module, tmp_path):
    """A stimulus resolution change between runs must not strand the previous run's blocks.

    This is the case that actually happened -- 101x101 and 301x301 runs into the same results dir.
    """
    sizes = []
    for scale in (1, 4, 1, 4, 1):
        sizes.append((scale, os.path.getsize(_one_run(writer_module, str(tmp_path), scale=scale))))
    small = [size for scale, size in sizes if scale == 1]
    assert small[0] == pytest.approx(small[-1], rel=0.01), f"file grew across runs: {sizes}"


def test_datasets_are_contiguous_not_chunked(writer_module, tmp_path):
    """`maxshape=` is what forced chunked storage; nothing appends, so nothing should be chunked."""
    filepath = _one_run(writer_module, str(tmp_path))
    with h5py.File(filepath, "r") as f:
        chunked = []
        f.visititems(lambda name, obj: chunked.append(name)
                     if isinstance(obj, h5py.Dataset) and obj.chunks is not None else None)
    assert chunked == [], f"still chunked, so still fragmenting: {chunked}"


def test_live_data_fills_the_file(writer_module, tmp_path):
    """The direct statement of the bug: bytes on disk must be accounted for by live datasets.

    Three runs, not one -- a single run packed the file tightly even before the fix, so checking
    after one run would pass either way. The dead space only appears from the second run onwards.
    """
    for _ in range(3):
        filepath = _one_run(writer_module, str(tmp_path))
    live = []
    with h5py.File(filepath, "r") as f:
        f.visititems(lambda name, obj: live.append(obj.size * obj.dtype.itemsize)
                     if isinstance(obj, h5py.Dataset) else None)
    on_disk = os.path.getsize(filepath)
    assert sum(live) > 0.95 * on_disk, (
        f"{sum(live)} bytes of live datasets in a {on_disk} byte file -- "
        f"{on_disk - sum(live)} bytes are unreachable")


def test_content_survives_the_truncation(writer_module, tmp_path):
    """Truncating on the first write must not lose datasets written later in the same run."""
    writer_module.GemWriteToFile._instance = None
    writer = writer_module.GemWriteToFile(result_dir=str(tmp_path), debugging_enabled=True)
    writer.write_array_to_h5(np.arange(10.0), variable_path=["model", "first"])
    writer.write_array_to_h5(np.arange(20.0), variable_path=["model", "second"])
    writer.write_array_to_h5(np.arange(30.0), variable_path=["model", "third"])

    with h5py.File(os.path.join(str(tmp_path), "debug_model_data.h5"), "r") as f:
        assert np.array_equal(f["model/first"][:], np.arange(10.0))
        assert np.array_equal(f["model/second"][:], np.arange(20.0))
        assert np.array_equal(f["model/third"][:], np.arange(30.0))


def test_rewriting_the_same_path_within_a_run_keeps_the_last_value(writer_module, tmp_path):
    """The overwrite branch (`del` then recreate) must still overwrite."""
    writer_module.GemWriteToFile._instance = None
    writer = writer_module.GemWriteToFile(result_dir=str(tmp_path), debugging_enabled=True)
    writer.write_array_to_h5(np.zeros(50), variable_path=["model", "signals"])
    writer.write_array_to_h5(np.ones(70), variable_path=["model", "signals"])

    with h5py.File(os.path.join(str(tmp_path), "debug_model_data.h5"), "r") as f:
        assert np.array_equal(f["model/signals"][:], np.ones(70))


def test_nothing_is_written_when_debugging_is_disabled(writer_module, tmp_path):
    writer_module.GemWriteToFile._instance = None
    writer = writer_module.GemWriteToFile(result_dir=str(tmp_path), debugging_enabled=False)
    writer.write_array_to_h5(np.zeros(10), variable_path=["model", "signals"])
    assert not os.path.exists(os.path.join(str(tmp_path), "debug_model_data.h5"))
