"""Regression tests for the single-GPU memory behaviour of the signal synthesis.

Two allocations dominate a run and both used to be sized as if memory were free. Across several GPUs
they are divided by the device count and stay unremarkable; on one GPU they are the run.

  * The per-GPU signal buffer was allocated at the stimulus' full frame count and only thinned out
    to the sampled frames once every chunk had been written into it. Each signal is still
    synthesized across every frame -- that is what high_temporal_resolution is for -- but the
    thinning now happens per finished chunk, so only one chunk, rather than the whole buffer, is
    held at the full frame count. On the roadmap01 config that is 2400 frames carried to keep 385.
  * `orthonormalize_modelled_signals` left every raw batch alive next to its orthonormalized
    counterpart. With three derivatives that is eight copies of the grid. `release_inputs=True`
    drops each raw batch as it is consumed, which must not change the output.
  * the derivative orthogonalization inside it was one expression building four full
    (num_timepoints, num_signals) arrays, two of which existed only to feed the next operator. They
    are folded in place now, which must reproduce the original expression exactly -- the operands
    and their order are unchanged, so this is bit-equality, not closeness.

`test_per_chunk_downsampling_equals_downsampling_the_full_buffer` is the one that matters for
correctness: a chunk holds complete timecourses, so thinning it is the same operation on the same
finished signals -- but only as long as every chunk uses the same index vector, which is what makes
hoisting that computation out of the loop legitimate.

Everything here imports `gem.signals.signal_synthesizer`, which imports CuPy at module level, so the
whole file skips without a CUDA GPU.
"""
import numpy as np
import pytest


def _legacy_downsample_indices(stim_frames, num_frames_downsampled, slice_time_ref):
    """The index arithmetic exactly as it was written inline in compute_signals_batches()."""
    idx = np.linspace(0, stim_frames, num_frames_downsampled, endpoint=False, dtype=int)
    slice_time_ref_adjusted_step_size = (np.diff(idx).mean() * slice_time_ref).round().astype(int)
    return (idx + slice_time_ref_adjusted_step_size).astype(int)


class _FakeStimulus:
    """Only the three temporal attributes downsample_frame_indices() reads."""

    def __init__(self, enabled=True, num_frames_downsampled=385, slice_time_ref=0.5):
        self.HighTemporalResolutionEnabled = enabled
        self.NumFramesDownsampled = num_frames_downsampled
        self.SliceTimeRef = slice_time_ref


# (stim_frames, num_frames_downsampled, slice_time_ref) -- the second row is the roadmap01 config
CASES = [(1200, 300, 0.0), (2400, 385, 0.5), (800, 200, 1.0), (997, 131, 0.25)]


@pytest.mark.parametrize("stim_frames, num_downsampled, slice_time_ref", CASES)
def test_downsample_frame_indices_matches_legacy_formula(stim_frames, num_downsampled, slice_time_ref):
    """The extracted helper picks exactly the frames the old inline code picked."""
    pytest.importorskip("cupy", reason="signal_synthesizer imports CuPy at module level")
    from gem.signals.signal_synthesizer import SignalSynthesizer

    stimulus = _FakeStimulus(True, num_downsampled, slice_time_ref)
    indices = SignalSynthesizer.downsample_frame_indices(stimulus, stim_frames)

    np.testing.assert_array_equal(indices, _legacy_downsample_indices(stim_frames, num_downsampled, slice_time_ref))
    assert len(indices) == num_downsampled


def test_downsample_frame_indices_is_none_when_disabled():
    """Without high_temporal_resolution the buffer keeps the stimulus' own frame count."""
    pytest.importorskip("cupy", reason="signal_synthesizer imports CuPy at module level")
    from gem.signals.signal_synthesizer import SignalSynthesizer

    assert SignalSynthesizer.downsample_frame_indices(_FakeStimulus(enabled=False), 2400) is None


def test_downsample_frame_indices_rejects_too_short_stimulus():
    """Fewer frames than requested is a config error and still exits rather than indexing garbage."""
    pytest.importorskip("cupy", reason="signal_synthesizer imports CuPy at module level")
    from gem.signals.signal_synthesizer import SignalSynthesizer

    with pytest.raises(SystemExit):
        SignalSynthesizer.downsample_frame_indices(_FakeStimulus(True, 385, 0.5), stim_frames=100)


def test_per_chunk_downsampling_equals_downsampling_the_full_buffer():
    """Thinning each finished chunk gives the buffer the old code produced, bit for bit.

    Every signal is synthesized across all frames either way; the only question is whether the
    unused frames are dropped as each chunk of complete signals arrives or after the last one. The
    kept indices do not depend on which chunk a signal landed in, so the two agree exactly.
    """
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.signals.signal_synthesizer import SignalSynthesizer

    stim_frames, num_signals = 2400, 97
    indices = SignalSynthesizer.downsample_frame_indices(_FakeStimulus(True, 385, 0.5), stim_frames)

    rng = np.random.default_rng(3)
    synthesized = cp.asarray(rng.normal(size=(num_signals, stim_frames))) # full-resolution signals

    # old: fill the buffer at full width, downsample once at the end
    full_width_buffer = cp.zeros((num_signals, stim_frames), dtype=cp.float64)
    # new: downsample each finished chunk on arrival into a narrow buffer
    narrow_buffer = cp.zeros((num_signals, len(indices)), dtype=cp.float64)

    for start in range(0, num_signals, 13): # deliberately not a divisor of num_signals
        rows = np.arange(start, min(start + 13, num_signals))
        chunk = synthesized[rows, :]
        full_width_buffer[rows, :] = chunk
        narrow_buffer[rows, :] = chunk[:, indices]

    np.testing.assert_array_equal(cp.asnumpy(full_width_buffer[:, indices]), cp.asnumpy(narrow_buffer))
    assert narrow_buffer.nbytes * (stim_frames / len(indices)) == pytest.approx(full_width_buffer.nbytes)


def test_get_available_gpus_without_cuda_visible_devices(monkeypatch):
    """An unset CUDA_VISIBLE_DEVICES used to raise KeyError instead of meaning "all of them"."""
    pytest.importorskip("cupy", reason="signal_synthesizer imports CuPy at module level")
    from gem.signals.signal_synthesizer import SignalSynthesizer
    from gem.utils.hpc_cupy_utils import HpcUtils

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    available_gpus, num_gpus = SignalSynthesizer.get_available_gpus(total_model_signals=1000, cfg=None)

    assert num_gpus == len(available_gpus) >= 1
    assert available_gpus == list(range(num_gpus))
    assert num_gpus == max(1, HpcUtils.get_number_of_gpus())


def test_derivative_orthogonalization_is_unchanged_by_the_in_place_evaluation():
    """The rewritten derivative step reproduces the original one-liner exactly, not approximately.

    dS'/dtheta = dS*/dtheta * n - (S* * n^3) * (S* . dS*/dtheta). Folding the two throwaway arrays
    into their predecessor keeps every operand pairing and every rounding, so `assert_array_equal`
    is the right assertion here -- `allclose` would let a genuine reassociation through.
    """
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.signals.signal_synthesizer import SignalSynthesizer
    from gem.utils.gem_gpu_manager import GemGpuManager

    if GemGpuManager.get_instance() is None:
        GemGpuManager(default_gpu_id=0)

    num_timepoints, num_signals, num_theta = 40, 23, 3
    rng = np.random.default_rng(19)

    O_gpu = cp.asarray(rng.normal(size=(num_timepoints, num_timepoints)))
    raw_S = cp.asarray(rng.normal(size=(num_signals, num_timepoints)))
    raw_dS = [cp.asarray(rng.normal(size=(num_signals, num_timepoints))) for _ in range(num_theta)]

    # the reference: the expression exactly as it used to be written, on the same intermediates
    S_star = cp.dot(O_gpu, raw_S.T)
    invroot = ((S_star ** 2).sum(axis=0)) ** (-1 / 2)
    expected = []
    for theta in range(num_theta):
        dS_star = cp.dot(O_gpu, raw_dS[theta].T)
        expected.append(dS_star * invroot - (S_star * (invroot ** 3)) * ((S_star * dS_star).sum(axis=0)))

    _, dS_prime_batches = SignalSynthesizer.orthonormalize_modelled_signals(
        O_gpu=O_gpu, model_signals_rm_batches=[raw_S],
        dS_dtheta_rm_batches_list=[[raw_dS[theta]] for theta in range(num_theta)])

    for theta in range(num_theta):
        np.testing.assert_array_equal(cp.asnumpy(dS_prime_batches[theta][0]), cp.asnumpy(expected[theta]))


def test_release_inputs_does_not_change_the_orthonormalized_signals():
    """Freeing each raw batch as it is consumed must be invisible in the result."""
    cp = pytest.importorskip("cupy", reason="needs a CUDA GPU")
    from gem.signals.signal_synthesizer import SignalSynthesizer
    from gem.utils.gem_gpu_manager import GemGpuManager

    if GemGpuManager.get_instance() is None:
        GemGpuManager(default_gpu_id=0)

    num_timepoints, num_theta = 40, 3
    chunk_widths = [11, 7, 5]
    rng = np.random.default_rng(11)

    O_gpu = cp.asarray(rng.normal(size=(num_timepoints, num_timepoints)))
    signals = [rng.normal(size=(width, num_timepoints)) for width in chunk_widths]
    derivatives = [[rng.normal(size=(width, num_timepoints)) for width in chunk_widths]
                   for _ in range(num_theta)]

    def run(release):
        S = [cp.asarray(batch) for batch in signals]
        dS = [[cp.asarray(batch) for batch in theta] for theta in derivatives]
        S_prime, dS_prime = SignalSynthesizer.orthonormalize_modelled_signals(
            O_gpu=O_gpu, model_signals_rm_batches=S, dS_dtheta_rm_batches_list=dS,
            release_inputs=release)
        return S, dS, S_prime, dS_prime

    kept_S, kept_dS, reference_S_prime, reference_dS_prime = run(release=False)
    released_S, released_dS, S_prime, dS_prime = run(release=True)

    for reference, actual in zip(reference_S_prime, S_prime):
        np.testing.assert_array_equal(cp.asnumpy(reference), cp.asnumpy(actual))
    for theta in range(num_theta):
        for reference, actual in zip(reference_dS_prime[theta], dS_prime[theta]):
            np.testing.assert_array_equal(cp.asnumpy(reference), cp.asnumpy(actual))

    # and the raw batches really are gone in the released run, still there in the other
    assert all(batch is None for batch in released_S)
    assert all(batch is None for theta in released_dS for batch in theta)
    assert all(batch is not None for batch in kept_S)
    assert all(batch is not None for theta in kept_dS for batch in theta)
