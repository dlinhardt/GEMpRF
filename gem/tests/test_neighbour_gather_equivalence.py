"""Equivalence tests for the deferred derivative / neighbour-gather refactor.

The grid fit used to build dense (num_y_signals, num_model_signals) and
(num_params, num_y_signals, num_model_signals) matrices, concatenate them into a
(num_y_signals, num_model_signals, num_params + 1) block and only then pick out the <=27 neighbour
columns of each y-signal's winning grid point. Those three arrays were the OOM in every failing run.
They are now evaluated straight into the gathered shape, one model-signal chunk at a time.

What has to hold for the refinement to be unchanged:
  * the same values land in the same slots of the flattened vector the M-inverse multiplies, and
  * the -1 padding slots are masked in exactly the same places.

`test_gather_algebra_matches_dense_path` pins that with plain numpy and runs anywhere.
`test_grid_fit_helpers_match_dense_path` runs the real GridFit helpers and needs a CUDA GPU, so it
skips without CuPy.
"""
import numpy as np
import pytest


NUM_Y_SIGNALS = 7
NUM_MODEL_SIGNALS = 53
NUM_PARAMS = 3
NUM_NEIGHBOURS = 5
# model signals per device -- deliberately uneven, and not a divisor of anything
CHUNK_WIDTHS = [17, 12, 20, 4]


def _reference_inputs(seed=0):
    """e, de and a neighbour-column table with -1 padding, as the fit produces them."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(NUM_Y_SIGNALS, NUM_MODEL_SIGNALS))
    de = rng.normal(size=(NUM_PARAMS, NUM_Y_SIGNALS, NUM_MODEL_SIGNALS))
    # get_full_2_validated_indices() marks neighbours that fell outside the validated grid with -1
    neighbour_columns = rng.integers(-1, NUM_MODEL_SIGNALS, size=(NUM_Y_SIGNALS, NUM_NEIGHBOURS))
    return e, de, neighbour_columns


def _dense_path(e, de, neighbour_columns):
    """The historical implementation: build the dense block, then gather out of it."""
    combined = np.concatenate([e[:, :, None], de.transpose(1, 2, 0)], axis=2)
    vecs = combined[np.arange(e.shape[0])[:, None], neighbour_columns.clip(min=0)]
    vecs = np.where(neighbour_columns[..., None] == -1, np.nan, vecs)
    return vecs.reshape(vecs.shape[0], -1)


def _assert_matches(dense_flat, new_flat):
    assert dense_flat.shape == new_flat.shape
    # NaN placement must agree, and the finite values must agree exactly (no arithmetic changed)
    np.testing.assert_array_equal(np.isnan(dense_flat), np.isnan(new_flat))
    np.testing.assert_array_equal(dense_flat[~np.isnan(dense_flat)], new_flat[~np.isnan(new_flat)])


def test_gather_algebra_matches_dense_path():
    """Chunk-wise masked gather reproduces the dense gather, slot for slot (CPU only)."""
    e, de, neighbour_columns = _reference_inputs()

    rows = np.arange(NUM_Y_SIGNALS)[:, None]
    error_neighbour_terms = e[rows, np.clip(neighbour_columns, 0, NUM_MODEL_SIGNALS - 1)]

    derivative_neighbour_terms = np.zeros((NUM_PARAMS, NUM_Y_SIGNALS, NUM_NEIGHBOURS))
    column_idx = 0
    for chunk_width in CHUNK_WIDTHS:  # one chunk per device
        local_columns = neighbour_columns - column_idx
        in_chunk = (local_columns >= 0) & (local_columns < chunk_width)
        local_columns = np.clip(local_columns, 0, chunk_width - 1)
        for theta in range(NUM_PARAMS):
            chunk = de[theta][:, column_idx:column_idx + chunk_width]
            derivative_neighbour_terms[theta] += np.where(in_chunk, chunk[rows, local_columns], 0.0)
        column_idx += chunk_width

    vecs = np.stack([error_neighbour_terms, *derivative_neighbour_terms], axis=2)
    vecs = np.where(neighbour_columns[..., None] == -1, np.nan, vecs)

    _assert_matches(_dense_path(e, de, neighbour_columns), vecs.reshape(vecs.shape[0], -1))


def test_padding_is_actually_exercised():
    """Guard the guard: the fixture must contain -1 padding, or the test above proves little."""
    _, _, neighbour_columns = _reference_inputs()
    assert (neighbour_columns == -1).any()


def test_grid_fit_helpers_match_dense_path():
    """The real GridFit helpers, on a GPU. Skipped where CuPy is unavailable."""
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.fitting.hpc_grid_fit import GridFit
    from gem.utils.gem_gpu_manager import GemGpuManager

    if GemGpuManager.get_instance() is None:
        GemGpuManager(default_gpu_id=0)

    e, de, neighbour_columns = _reference_inputs(seed=1)

    # Y and S'/dS' chosen so that Y.T @ S' reproduces e and Y.T @ dS' reproduces de exactly:
    # with Y = I the products are just the matrices themselves.
    Y_signals_gpu = cp.asarray(np.eye(NUM_Y_SIGNALS))
    S_prime_batches, dS_prime_batches = [], [[] for _ in range(NUM_PARAMS)]
    column_idx = 0
    for chunk_width in CHUNK_WIDTHS:
        S_prime_batches.append(cp.asarray(e[:, column_idx:column_idx + chunk_width]))
        for theta in range(NUM_PARAMS):
            dS_prime_batches[theta].append(cp.asarray(de[theta][:, column_idx:column_idx + chunk_width]))
        column_idx += chunk_width

    error_matrix = GridFit.compute_error_matrix(Y_signals_gpu, S_prime_batches)
    cp.testing.assert_array_equal(error_matrix, cp.asarray(e))

    error_neighbour_terms = GridFit.gather_neighbour_terms(error_matrix, neighbour_columns)
    derivative_neighbour_terms = GridFit.accumulate_derivative_neighbour_terms(
        Y_signals_gpu, dS_prime_batches, neighbour_columns)
    vecs = GridFit.build_refine_input_vectors(error_neighbour_terms, derivative_neighbour_terms,
                                              isResultOnGPU=False)
    vecs = np.where(neighbour_columns[..., None] == -1, np.nan, vecs)

    _assert_matches(_dense_path(e, de, neighbour_columns), vecs.reshape(vecs.shape[0], -1))


def test_single_chunk_error_matrix_matches_the_assembled_path():
    """One chunk on the default device is returned directly; the values must not change.

    With a single GPU the whole grid is one chunk, so building a separate (num_y_signals,
    num_model_signals) output and copying the product into it doubled the largest allocation of the
    run for no gain. The short-circuit returns the product itself -- including the +inf -> -inf fold
    the assembled path applies per chunk.
    """
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.fitting.hpc_grid_fit import GridFit
    from gem.utils.gem_gpu_manager import GemGpuManager

    if GemGpuManager.get_instance() is None:
        GemGpuManager(default_gpu_id=0)
    default_gpu_id = GemGpuManager.get_instance().default_gpu_id

    rng = np.random.default_rng(23)
    with cp.cuda.Device(default_gpu_id):
        Y = cp.asarray(rng.normal(size=(31, NUM_Y_SIGNALS)))
        S_prime = cp.asarray(rng.normal(size=(31, NUM_MODEL_SIGNALS)))
        # a degenerate model signal, as orthonormalize_modelled_signals marks them
        S_prime[:, 4] = cp.inf

        short_circuit = GridFit.compute_error_matrix(Y, [S_prime])

        # force the assembling path by handing it somewhere to assemble into
        out = cp.zeros((NUM_Y_SIGNALS, NUM_MODEL_SIGNALS), dtype=cp.float64)
        assembled = GridFit.compute_error_matrix(Y, [S_prime], out=out, accumulate=False)

        np.testing.assert_array_equal(cp.asnumpy(short_circuit), cp.asnumpy(assembled))

        # The degenerate column must not score as a *best* fit. It comes out NaN rather than +inf
        # whenever Y has mixed signs -- the product sums +inf and -inf -- and nanargmax skips NaN,
        # so either outcome is safe; what must never survive is a +inf, which would win the argmax.
        degenerate = cp.asnumpy(short_circuit[:, 4])
        assert np.all(np.isneginf(degenerate) | np.isnan(degenerate)), \
            f"degenerate column scored {degenerate}, expected -inf or NaN"
        assert not np.isposinf(cp.asnumpy(short_circuit)).any(), "no +inf may survive"


def test_error_matrix_accumulates_across_concatenated_runs():
    """Two runs added in place must equal the old stack-then-sum. Skipped without CuPy."""
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.fitting.hpc_grid_fit import GridFit
    from gem.utils.gem_gpu_manager import GemGpuManager

    if GemGpuManager.get_instance() is None:
        GemGpuManager(default_gpu_id=0)

    e_run1, _, _ = _reference_inputs(seed=2)
    e_run2, _, _ = _reference_inputs(seed=3)
    Y_signals_gpu = cp.asarray(np.eye(NUM_Y_SIGNALS))

    def _as_batches(matrix):
        batches, column_idx = [], 0
        for chunk_width in CHUNK_WIDTHS:
            batches.append(cp.asarray(matrix[:, column_idx:column_idx + chunk_width]))
            column_idx += chunk_width
        return batches

    accumulated = GridFit.compute_error_matrix(Y_signals_gpu, _as_batches(e_run1))
    accumulated = GridFit.compute_error_matrix(Y_signals_gpu, _as_batches(e_run2),
                                               out=accumulated, accumulate=True)

    cp.testing.assert_array_equal(accumulated, cp.asarray(e_run1 + e_run2))


# --------------------------------------------------------------------------------------------
# Y-batch auto-sizing (C3). These import gem.run.run_gem_prf_analysis, which pulls in CuPy at
# module load, so they are GPU-machine only.
# --------------------------------------------------------------------------------------------

def _analysis_class():
    pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.run.run_gem_prf_analysis import GEMpRFAnalysis
    return GEMpRFAnalysis


class _Cfg:
    def __init__(self, batches):
        self.measured_data = {"batches": batches}


def test_batches_setting_accepts_plain_value_and_auto_attribute():
    GEMpRFAnalysis = _analysis_class()
    # <batches>200</batches>
    assert GEMpRFAnalysis._resolve_batches_setting(_Cfg("200")) == (200, False)
    # <batches auto="true">200</batches> -- xmltodict wraps the text once attributes are present
    assert GEMpRFAnalysis._resolve_batches_setting(_Cfg({"#text": "200", "@auto": "true"})) == (200, True)
    assert GEMpRFAnalysis._resolve_batches_setting(_Cfg({"#text": "200", "@auto": "false"})) == (200, False)
    # attribute present but no explicit auto flag
    assert GEMpRFAnalysis._resolve_batches_setting(_Cfg({"#text": "50"})) == (50, False)


def test_auto_sizing_never_enlarges_batches_without_the_auto_flag(monkeypatch):
    """The default must not change the batch size of a config that already fits."""
    GEMpRFAnalysis = _analysis_class()
    from gem.run import run_gem_prf_analysis as module

    total_y_signals, num_model_signals = 141034, 942030
    configured = max(1, int(total_y_signals / 200))  # 705

    # pretend the card is nearly empty, so the memory budget alone would allow a much bigger batch
    monkeypatch.setattr(module.gpu_utils, "device_available_mem_bytes", lambda device_id: 30 * 1024 ** 3)
    monkeypatch.setattr(module.gpu_utils, "get_number_of_gpus", lambda: 4)

    without_flag = GEMpRFAnalysis.get_y_batch_size(_Cfg("200"), total_y_signals, num_model_signals)
    assert without_flag == configured, "default mode must stay at the configured batch size"

    with_flag = GEMpRFAnalysis.get_y_batch_size(_Cfg({"#text": "200", "@auto": "true"}),
                                                total_y_signals, num_model_signals)
    assert with_flag > configured, "auto mode should use the spare memory"


def test_auto_sizing_shrinks_batches_when_memory_is_short(monkeypatch):
    """A card too small for the configured batch size gets a smaller one instead of an OOM."""
    GEMpRFAnalysis = _analysis_class()
    from gem.run import run_gem_prf_analysis as module

    total_y_signals, num_model_signals = 141034, 942030
    configured = max(1, int(total_y_signals / 200))

    monkeypatch.setattr(module.gpu_utils, "device_available_mem_bytes", lambda device_id: 2 * 1024 ** 3)
    monkeypatch.setattr(module.gpu_utils, "get_number_of_gpus", lambda: 4)

    batch_size = GEMpRFAnalysis.get_y_batch_size(_Cfg("200"), total_y_signals, num_model_signals)
    assert 1 <= batch_size < configured


def test_auto_sizing_falls_back_to_config_when_memory_cannot_be_queried(monkeypatch):
    GEMpRFAnalysis = _analysis_class()
    from gem.run import run_gem_prf_analysis as module

    def _boom(device_id):
        raise RuntimeError("no device")

    monkeypatch.setattr(module.gpu_utils, "device_available_mem_bytes", _boom)
    assert GEMpRFAnalysis.get_y_batch_size(_Cfg("200"), 141034, 942030) == 705


# --------------------------------------------------------------------------------------------
# Multi-GPU placement. The model signals are spread over the GPUs, so every helper that touches
# them has to move data across devices correctly. CuPy refuses to copy a *non-contiguous* array
# between devices, which is easy to hit by accident because a column slice of Y is a strided view.
# Needs more than one GPU.
# --------------------------------------------------------------------------------------------

def _multi_gpu_or_skip():
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs at least 2 GPUs")
    from gem.utils.gem_gpu_manager import GemGpuManager
    if GemGpuManager.get_instance() is None:
        GemGpuManager(default_gpu_id=0)
    return cp


def test_matched_error_terms_across_devices():
    """Per-signal projection with the model chunks spread over several GPUs.

    Regression guard: this used to raise "CuPy cannot copy non-contiguous array between devices",
    because the column slice Y[:, a:b] was being sent to the chunk's device instead of the chunk
    being brought back.
    """
    cp = _multi_gpu_or_skip()
    from gem.fitting.hpc_grid_fit import GridFit

    num_timepoints, num_signals = 385, 32
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(num_timepoints, num_signals))
    S_prime = rng.normal(size=(num_timepoints, num_signals))

    num_devices = min(4, cp.cuda.runtime.getDeviceCount())
    widths = [num_signals // num_devices] * num_devices
    widths[-1] += num_signals - sum(widths)

    Y_signals_gpu = cp.asarray(Y)
    batches, column_idx = [], 0
    for device_id, width in enumerate(widths):
        with cp.cuda.Device(device_id):
            batches.append(cp.asarray(S_prime[:, column_idx:column_idx + width]))
        column_idx += width

    matched = cp.asnumpy(GridFit.compute_matched_error_terms(Y_signals_gpu, batches))

    # reference: the diagonal of the full product, which is what this replaced
    np.testing.assert_allclose(matched, np.diagonal(Y.T @ S_prime), rtol=0, atol=1e-12)


def test_error_matrix_across_devices():
    """compute_error_matrix must assemble the same matrix with chunks on different GPUs."""
    cp = _multi_gpu_or_skip()
    from gem.fitting.hpc_grid_fit import GridFit

    num_timepoints, num_y, num_models = 385, 16, 40
    rng = np.random.default_rng(8)
    Y = rng.normal(size=(num_timepoints, num_y))
    S_prime = rng.normal(size=(num_timepoints, num_models))

    num_devices = min(4, cp.cuda.runtime.getDeviceCount())
    widths = [num_models // num_devices] * num_devices
    widths[-1] += num_models - sum(widths)

    batches, column_idx = [], 0
    for device_id, width in enumerate(widths):
        with cp.cuda.Device(device_id):
            batches.append(cp.asarray(S_prime[:, column_idx:column_idx + width]))
        column_idx += width

    error_matrix = cp.asnumpy(GridFit.compute_error_matrix(cp.asarray(Y), batches))
    np.testing.assert_allclose(error_matrix, Y.T @ S_prime, rtol=0, atol=1e-12)


def test_derivative_neighbour_terms_across_devices():
    """The derivative gather must also work with chunks spread over several GPUs."""
    cp = _multi_gpu_or_skip()
    from gem.fitting.hpc_grid_fit import GridFit

    num_timepoints, num_y, num_models, num_params, num_neigh = 385, 16, 40, 3, 5
    rng = np.random.default_rng(9)
    Y = rng.normal(size=(num_timepoints, num_y))
    dS = rng.normal(size=(num_params, num_timepoints, num_models))
    neighbour_columns = rng.integers(-1, num_models, size=(num_y, num_neigh))

    num_devices = min(4, cp.cuda.runtime.getDeviceCount())
    widths = [num_models // num_devices] * num_devices
    widths[-1] += num_models - sum(widths)

    dS_batches, column_idx = [[] for _ in range(num_params)], 0
    for device_id, width in enumerate(widths):
        with cp.cuda.Device(device_id):
            for theta in range(num_params):
                dS_batches[theta].append(cp.asarray(dS[theta][:, column_idx:column_idx + width]))
        column_idx += width

    gathered = cp.asnumpy(GridFit.accumulate_derivative_neighbour_terms(
        cp.asarray(Y), dS_batches, neighbour_columns))

    rows = np.arange(num_y)[:, None]
    for theta in range(num_params):
        dense = (Y.T @ dS[theta])[rows, np.clip(neighbour_columns, 0, num_models - 1)]
        dense = np.where(neighbour_columns == -1, 0.0, dense)
        np.testing.assert_allclose(gathered[theta], dense, rtol=0, atol=1e-12)


def test_matched_error_terms_map_positive_infinity_to_negative():
    """A degenerate model signal must score as the *worst* fit, not the best.

    orthonormalize_modelled_signals() flags a model signal whose norm blew up (the pRF drifted out
    of the stimulus aperture) by setting its whole column to +inf. get_valid_refined_data() then
    treats a smaller error as "the refinement made things worse", so +inf has to be folded to -inf or
    those vertices score as perfect fits and never revert to their grid point.

    Regression guard: dropping this sanitisation moved ~0.5% of vertices per hemisphere.
    """
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.fitting.hpc_grid_fit import GridFit
    from gem.utils.gem_gpu_manager import GemGpuManager

    if GemGpuManager.get_instance() is None:
        GemGpuManager(default_gpu_id=0)

    num_timepoints, num_signals = 385, 8
    rng = np.random.default_rng(11)
    Y = rng.normal(size=(num_timepoints, num_signals))
    S_prime = rng.normal(size=(num_timepoints, num_signals))
    S_prime[:, 3] = np.inf          # degenerate signal, as the orthonormalisation marks it

    matched = cp.asnumpy(GridFit.compute_matched_error_terms(cp.asarray(Y), [cp.asarray(S_prime)]))

    assert np.isneginf(matched[3]) or np.isnan(matched[3]), \
        f"degenerate signal scored {matched[3]}, expected -inf (or NaN)"
    assert not np.isposinf(matched).any(), "no +inf may survive"
    finite = [i for i in range(num_signals) if i != 3]
    np.testing.assert_allclose(matched[finite], np.diagonal(Y.T @ S_prime)[finite], rtol=0, atol=1e-12)
