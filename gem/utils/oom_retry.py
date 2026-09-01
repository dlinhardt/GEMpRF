# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Medical University of Vienna",
"@Desc    :   Re-size a Y-batch from the numbers a CuPy out-of-memory error carries, so a run that
              would have failed finishes at a smaller batch instead.",
"""

import re

# NOTE: gem.utils.hpc_cupy_utils imports CuPy at module level, so it is imported inside
# next_batch_size() rather than here. Everything else in this module is plain arithmetic and string
# handling, and keeping it importable without a GPU is what lets it be tested off the server.
from gem.utils.logger import Logger


# How many times a single input may be retried at a smaller batch before the failure is reported.
# Each attempt at least halves the batch, so three retries reach B/8.
MAX_OOM_RETRIES = 3

# Fraction of the free memory measured after the pool release that the re-sized batch may claim.
# Deliberately below the sizer's own margin: by the time this runs, one estimate has already been
# wrong, so the second one should not be the aggressive kind.
RETRY_MEMORY_SAFETY_FRACTION = 0.35

# "Out of memory allocating 2,980,583,424 bytes (allocated so far: 31,655,235,584 bytes)."
_OOM_MESSAGE = re.compile(r"allocating\s+([\d,]+)\s+bytes.*?allocated so far:\s*([\d,]+)\s*bytes",
                          re.IGNORECASE | re.DOTALL)


def is_out_of_memory(exc):
    """True for a CuPy OutOfMemoryError, without importing cupy at module import time."""
    try:
        import cupy as cp
        if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
            return True
    except Exception:
        pass
    # The class name is checked as well so a driver-level or wrapped variant still counts.
    return type(exc).__name__ == "OutOfMemoryError"


def parse_oom(exc):
    """-> (requested_bytes, pool_bytes) from an OutOfMemoryError, or (None, None).

    CuPy raises ``OutOfMemoryError(size, total)``, so the numbers are usually in ``exc.args``. They
    are also in the rendered message, which is parsed as a fallback: the args are not part of any
    documented contract, and losing the retry entirely because a CuPy release changed how it builds
    the exception would be a poor trade.
    """
    if not is_out_of_memory(exc):
        return None, None

    args = [arg for arg in getattr(exc, "args", ()) if isinstance(arg, int)]
    if len(args) >= 2 and args[0] > 0:
        return args[0], args[1]

    match = _OOM_MESSAGE.search(str(exc))
    if match:
        return int(match.group(1).replace(",", "")), int(match.group(2).replace(",", ""))

    return None, None


def next_batch_size(current_batch_size, requested_bytes, device_id):
    """A smaller Y-batch that the measured free memory can hold, or None if there is nothing to try.

    ``requested_bytes / current_batch_size`` is the memory per vertex of *the array that actually
    ran out*. That is the whole point of re-sizing from the exception rather than from a model:
    get_y_batch_size() has to guess which allocation will be the peak, and it has now guessed wrong
    in three different configurations. This number needs no guess.

    The result is always at most half the current batch. A peak this arithmetic also mis-reads would
    otherwise be retried at the same size forever.
    """
    if not current_batch_size or current_batch_size <= 1:
        return None
    if not requested_bytes or requested_bytes <= 0:
        # Nothing measured to scale from -- fall back to halving, which is still progress.
        return max(1, current_batch_size // 2)

    try:
        from gem.utils.hpc_cupy_utils import HpcUtils as gpu_utils
        # A fragmentation-only failure can clear here on its own, and the reading below is
        # meaningless until the pool has handed its free blocks back.
        gpu_utils.release_pool_on_all_devices()
        free_bytes = gpu_utils.device_available_mem_bytes(device_id=device_id)
    except Exception:
        return max(1, current_batch_size // 2)

    bytes_per_vertex = requested_bytes / float(current_batch_size)
    affordable = int((free_bytes * RETRY_MEMORY_SAFETY_FRACTION) / max(1.0, bytes_per_vertex))
    return max(1, min(affordable, current_batch_size // 2))


def announce(input_desc, old_batch_size, new_batch_size, requested_bytes):
    """Say loudly that the batch was re-sized, because it changes the results.

    The batch size decides how many vertices are argmaxed together, and which of them end up on the
    grid rather than the refined fit; a run that quietly shrank its batch is not comparable with one
    that did not. Silently recovering would turn a hard failure into an inconsistency between
    subjects that nobody notices for months.
    """
    detail = f" after a {requested_bytes:,} byte allocation failed" if requested_bytes else ""
    Logger.print_red_message(
        f"Out of memory{detail}. Retrying this input with a Y-batch of {new_batch_size} instead of "
        f"{old_batch_size} vertices. NOTE: batch size affects which vertices revert to the grid, so "
        f"these results are not bit-comparable with a run that did not retry.\n  {input_desc}",
        print_file_name=False)
