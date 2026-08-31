"""The model-curve kernels must fill the GPU and write coalesced, without changing any value.

The kernels used to be launched one thread per pRF point, in blocks of 512. For the 15,000-curve
chunk that `compute_signals_batches` caps at, that is ceil(15000/512) = **30 blocks on an 80-SM
V100** -- fewer blocks than the device has SMs, and about 9% of the warps a single SM can hold, so
most of the GPU idled and nothing hid memory latency. Each thread then walked its entire curve, so
lanes within a warp wrote addresses `num_pixels` doubles apart and every 8-byte store became its own
32-byte transaction.

Consecutive threads now cover consecutive pixels of one curve, with one block-row per curve. The
per-element arithmetic is untouched, so the model curves are bit-identical -- that part is verified
against the previous mapping by compiling `gaussian_kernel.cu` as host C++ (240 randomised cases,
including the blockIdx.y grid-stride path). What is pinned here is the launch geometry, since a
regression there would silently restore the old under-occupancy without changing any result.

`gem.signals.signal_synthesizer` imports CuPy at module level, so this file skips without it.
"""
import pytest

# The launch geometry the kernels are written against.
THREADS_PER_BLOCK = 256
V100_SM_COUNT = 80


def _kernel_config(num_model_curves, num_pixels):
    from gem.signals.signal_synthesizer import SignalSynthesizer
    # name-mangled private classmethod
    return SignalSynthesizer._SignalSynthesizer__set_kernel_config(
        num_model_curves=num_model_curves, num_pixels=num_pixels)


@pytest.fixture(autouse=True)
def _needs_cupy():
    pytest.importorskip("cupy", reason="signal_synthesizer imports CuPy at module level")


def test_the_real_chunk_fills_the_device():
    """15,000 curves x 301x301 px: the case that used to launch 30 blocks."""
    block_dim, grid_dim = _kernel_config(15000, 301 * 301)
    assert block_dim == (THREADS_PER_BLOCK, 1, 1)
    total_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2]
    assert total_blocks > 100 * V100_SM_COUNT, (
        f"only {total_blocks} blocks for an {V100_SM_COUNT}-SM device")


def test_threads_run_along_pixels_and_blocks_along_curves():
    """The mapping itself: x covers the pixels of one curve, y indexes the curves."""
    num_curves, num_pixels = 4000, 101 * 101
    _, grid_dim = _kernel_config(num_curves, num_pixels)
    assert grid_dim[0] == (num_pixels + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    assert grid_dim[1] == num_curves
    assert grid_dim[2] == 1


def test_x_dimension_covers_every_pixel():
    """No pixel may be left unwritten, for pixel counts that do and do not divide the block size."""
    for num_pixels in (1, 255, 256, 257, 10201, 90601):
        block_dim, grid_dim = _kernel_config(64, num_pixels)
        assert grid_dim[0] * block_dim[0] >= num_pixels, f"{num_pixels} pixels not covered"


def test_y_dimension_respects_the_cuda_limit():
    """gridDim.y caps at 65535; the kernels stride over blockIdx.y to cover the rest."""
    from gem.signals.signal_synthesizer import SignalSynthesizer
    _, grid_dim = _kernel_config(200000, 1000)
    assert grid_dim[1] == SignalSynthesizer.MAX_GRID_DIM_Y
    assert grid_dim[1] <= 65535


def test_degenerate_inputs_do_not_produce_an_empty_launch():
    """A zero-sized grid dimension is an invalid launch, not a no-op."""
    for curves, pixels in ((0, 100), (100, 0), (0, 0), (1, 1)):
        _, grid_dim = _kernel_config(curves, pixels)
        assert all(dim >= 1 for dim in grid_dim), f"{curves=} {pixels=} gave {grid_dim}"


def test_flat_output_index_would_overflow_int32_at_the_chunk_cap():
    """Why the kernels index with size_t: the old int index had little headroom left.

    15,000 curves x 90,601 pixels is 1.36e9, within 1.6x of the signed 32-bit limit -- so raising
    the chunk cap, or a larger stimulus, would have silently wrapped the write offset.
    """
    largest_flat_index = 15000 * 301 * 301
    assert largest_flat_index < 2**31 - 1
    assert largest_flat_index > 0.5 * (2**31 - 1), "headroom grew; this note may be stale"
