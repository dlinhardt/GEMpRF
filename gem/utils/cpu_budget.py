# -*- coding: utf-8 -*-

"""How many CPU threads one GEMpRF process may use.

Everything CPU-parallel here -- the numba ``prange`` pinv loop, the joblib neighbour search, and the
BLAS underneath numpy/scipy -- defaults to "every core on the machine". That is right for a single
run and actively harmful for several: three concurrent analyses each spawning a full-machine thread
pool oversubscribe the CPU 3x and all three get slower. Measured on the same grid and machine, the
M-inverse thread took 2:23 alone and 4:44 with two other analyses running -- a 2x penalty for work
that was not sharing anything but the cores.

So a process takes HALF the available CPUs. On a machine running one analysis that leaves headroom
and costs little (these loops do not scale linearly to the last core anyway); on a machine running
two it is exactly right; on three or more it still oversubscribes, but far less badly than before.
"""

import os
import sys

# Deliberately fixed rather than configurable: a config knob here is one more thing to get wrong, and
# the failure mode (a run that is silently 2x slow) is invisible without a side-by-side comparison.
CPU_FRACTION = 0.5


def available_cpus():
    """CPUs this process may actually run on -- affinity, not the machine's core count.

    sched_getaffinity respects taskset/cgroup/SLURM pinning; os.cpu_count() does not and would hand
    out a budget the scheduler will never honour.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def thread_budget():
    """Threads one analysis may use. At least 1, whatever the machine looks like."""
    return max(1, int(available_cpus() * CPU_FRACTION))


# Read once when the library loads; setting them later is ignored but never an error.
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS")


def apply_thread_budget_env():
    """Cap the thread pools via the environment, for the entry point to call before numpy is imported.

    Existing values are left alone: an explicit choice in the environment outranks this default.

    NUMBA_NUM_THREADS is handled separately and only while numba is still unimported. Numba re-reads
    its configuration from the environment on *every* compile, and raises if what it finds disagrees
    with the thread count it has already launched -- so writing that variable into a process where
    numba is live turns the next compile into a RuntimeError, wherever it happens to occur. Once
    numba is imported the only safe lever is set_num_threads(), which apply_numba_thread_budget uses.
    """
    budget = thread_budget()
    for variable in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(variable, str(budget))

    if "numba" not in sys.modules:
        os.environ.setdefault("NUMBA_NUM_THREADS", str(budget))

    return budget


def apply_numba_thread_budget():
    """Cap numba's ``prange`` pool at runtime, for callers that import too late for the env var.

    Never raises: a thread cap is an optimisation, and a run that cannot apply it is still correct.
    """
    budget = thread_budget()
    try:
        import numba
        numba.set_num_threads(min(budget, numba.get_num_threads()))
    except Exception:
        pass
    return budget
