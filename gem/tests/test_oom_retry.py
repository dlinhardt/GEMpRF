"""Re-sizing a Y-batch from the numbers a CuPy out-of-memory error carries.

The concatenated cons01 run died with

    Out of memory allocating 2,980,583,424 bytes (allocated so far: 31,655,235,584 bytes)

which is 791 vertices x 471,015 columns x 8 bytes -- a GEMM transient that `get_y_batch_size`'s cost
model does not describe, because that model assumes the error matrix is the peak. Rather than keep
extending the model (this was its third wrong configuration), the retry divides the *failed*
allocation by the batch it was sized against and gets a bytes-per-vertex for the array that actually
ran out, whatever that array happens to be.

Everything here is arithmetic and string handling, so it runs without a GPU. `next_batch_size`
imports CuPy lazily and falls back to halving when it cannot measure, which is the path exercised
off the server.
"""
import pytest

from gem.utils import oom_retry


class _FakeOOM(Exception):
    """Stands in for cupy.cuda.memory.OutOfMemoryError, which cannot be imported without a GPU."""
    __name__ = "OutOfMemoryError"

    def __init__(self, size, total):
        super().__init__(size, total)
        self.size, self.total = size, total

    def __str__(self):
        return f"Out of memory allocating {self.size:,} bytes (allocated so far: {self.total:,} bytes)."


# The class name is what is_out_of_memory() matches on, so it has to be the real one.
_FakeOOM.__qualname__ = "OutOfMemoryError"
OutOfMemoryError = _FakeOOM
OutOfMemoryError.__name__ = "OutOfMemoryError"


# ------------------------------------------------------------------------------------ recognition

def test_recognises_an_out_of_memory_error_by_class_name():
    assert oom_retry.is_out_of_memory(OutOfMemoryError(1, 2))


def test_does_not_treat_other_failures_as_out_of_memory():
    for exc in (ValueError("nope"), RuntimeError("nope"), MemoryError("host ram")):
        assert not oom_retry.is_out_of_memory(exc)
        assert oom_retry.parse_oom(exc) == (None, None)


# ----------------------------------------------------------------------------------------- parsing

def test_parses_the_numbers_from_the_exception_args():
    assert oom_retry.parse_oom(OutOfMemoryError(2_980_583_424, 31_655_235_584)) == \
        (2_980_583_424, 31_655_235_584)


def test_parses_the_real_cons01_message_when_args_are_missing():
    """The fallback path: CuPy is free to stop passing the numbers positionally."""
    class _MessageOnly(Exception):
        pass
    _MessageOnly.__name__ = "OutOfMemoryError"
    exc = _MessageOnly("Out of memory allocating 2,980,583,424 bytes "
                       "(allocated so far: 31,655,235,584 bytes).")
    assert oom_retry.parse_oom(exc) == (2_980_583_424, 31_655_235_584)


def test_parsing_survives_an_unrecognised_message():
    class _Odd(Exception):
        pass
    _Odd.__name__ = "OutOfMemoryError"
    assert oom_retry.parse_oom(_Odd("something else entirely")) == (None, None)


# -------------------------------------------------------------------------------------- re-sizing

def _next(current, requested, free_bytes=None):
    """next_batch_size with the GPU query stubbed, or absent entirely when free_bytes is None."""
    if free_bytes is None:
        return oom_retry.next_batch_size(current, requested, device_id=0)

    class _Stub:
        @staticmethod
        def release_pool_on_all_devices():
            pass

        @staticmethod
        def device_available_mem_bytes(device_id):
            return free_bytes

    import sys
    import types
    module = types.ModuleType("gem.utils.hpc_cupy_utils")
    module.HpcUtils = _Stub
    saved = sys.modules.get("gem.utils.hpc_cupy_utils")
    sys.modules["gem.utils.hpc_cupy_utils"] = module
    try:
        return oom_retry.next_batch_size(current, requested, device_id=0)
    finally:
        if saved is None:
            del sys.modules["gem.utils.hpc_cupy_utils"]
        else:
            sys.modules["gem.utils.hpc_cupy_utils"] = saved


def test_uses_the_failed_allocation_as_bytes_per_vertex():
    """The cons01 numbers: 2.98 GB / 791 = 3.77 MB per vertex, against 8 GB free."""
    free_bytes = 8 * 1024 ** 3
    result = _next(791, 2_980_583_424, free_bytes=free_bytes)
    expected = int((free_bytes * oom_retry.RETRY_MEMORY_SAFETY_FRACTION) / (2_980_583_424 / 791.0))
    assert result == min(expected, 791 // 2)


def test_always_at_least_halves_the_batch():
    """A peak this arithmetic also mis-reads must not be retried at the same size forever."""
    # Plenty of free memory, so the scaled estimate would be far larger than the current batch.
    assert _next(800, 1000, free_bytes=1024 ** 4) == 400


def test_never_returns_something_larger_than_the_current_batch():
    for current in (2, 17, 128, 791, 4096):
        result = _next(current, 10 ** 6, free_bytes=10 ** 12)
        assert 1 <= result < current, f"{current} -> {result}"


def test_gives_up_when_the_batch_is_already_one():
    assert _next(1, 2_980_583_424, free_bytes=10 ** 12) is None
    assert _next(0, 2_980_583_424, free_bytes=10 ** 12) is None


def test_halves_when_the_allocation_size_is_unknown():
    """parse_oom() returning None must still make progress rather than abandoning the input."""
    assert _next(791, None, free_bytes=10 ** 12) == 395


def test_halves_when_the_device_cannot_be_queried():
    """No CuPy, or a failing query: the lazy import raises and the fallback takes over."""
    assert _next(791, 2_980_583_424) == 395


def test_shrinks_further_when_memory_is_scarcer():
    scarce = _next(791, 2_980_583_424, free_bytes=1 * 1024 ** 3)
    roomy = _next(791, 2_980_583_424, free_bytes=8 * 1024 ** 3)
    assert scarce < roomy


# ---------------------------------------------------------------------------------- announcement

def test_announcement_names_both_sizes_and_warns_about_comparability(capsys):
    """A silent recovery would turn a hard failure into a quiet inconsistency between subjects."""
    oom_retry.announce("sub-cons01_..._hemi-L_bold.nii.gz", 791, 395, 2_980_583_424)
    printed = capsys.readouterr().out
    assert "791" in printed and "395" in printed
    assert "2,980,583,424" in printed
    assert "not bit-comparable" in printed


def test_announcement_works_without_a_measured_size(capsys):
    oom_retry.announce("some/file.nii.gz", 791, 395, None)
    printed = capsys.readouterr().out
    assert "791" in printed and "395" in printed
