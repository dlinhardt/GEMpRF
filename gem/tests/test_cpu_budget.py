"""One analysis takes half the machine's CPUs, not all of them.

Everything CPU-parallel in GEMpRF defaults to "every core": the numba prange pinv loop, the joblib
neighbour search, and the BLAS under numpy/scipy. That is right for one run and harmful for several
at once -- measured on the same grid and machine, the M-inverse thread took 2:23 alone and 4:44 with
two other analyses running, a 2x penalty purely from oversubscription.
"""
import os

import pytest

from gem.utils import cpu_budget


def test_budget_is_half_the_available_cpus():
    assert cpu_budget.thread_budget() == max(1, int(cpu_budget.available_cpus() * 0.5))


def test_budget_is_never_zero(monkeypatch):
    """A one-core machine (or a tight cgroup) must still get a usable budget, not 0 threads."""
    monkeypatch.setattr(cpu_budget, "available_cpus", lambda: 1)
    assert cpu_budget.thread_budget() == 1


def test_budget_respects_affinity_not_the_machine_size():
    """sched_getaffinity, not os.cpu_count: a pinned process must not hand out cores it cannot use."""
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("affinity is Linux-only")
    assert cpu_budget.available_cpus() == len(os.sched_getaffinity(0))


def test_env_capping_does_not_override_an_explicit_choice(monkeypatch):
    """A thread count already set in the environment is the user's decision and outranks the default."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    cpu_budget.apply_thread_budget_env()
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == str(cpu_budget.thread_budget())


def test_env_capping_sets_every_blas_pool(monkeypatch):
    for name in cpu_budget.BLAS_THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    budget = cpu_budget.apply_thread_budget_env()
    for name in cpu_budget.BLAS_THREAD_VARIABLES:
        assert os.environ[name] == str(budget), name


def test_numba_env_var_is_left_alone_once_numba_is_imported():
    """Writing NUMBA_NUM_THREADS into a live process breaks the next compile, anywhere it happens.

    Numba re-reads its config from the environment on every compile and raises if the value disagrees
    with the thread count it already launched. This test itself is the regression: an earlier version
    set the variable unconditionally, and because numba is imported by the time the suite runs, every
    later test that compiled a jitted function died with
    "Cannot set NUMBA_NUM_THREADS to a different value once the threads have been launched".
    """
    import numba  # noqa: F401  -- the point is that it IS in sys.modules

    before = os.environ.get("NUMBA_NUM_THREADS")
    cpu_budget.apply_thread_budget_env()
    assert os.environ.get("NUMBA_NUM_THREADS") == before


def test_numba_cap_never_raises(monkeypatch):
    """It is an optimisation; a run that cannot apply it must still proceed."""
    monkeypatch.setattr(cpu_budget, "thread_budget", lambda: 10 ** 9)
    assert cpu_budget.apply_numba_thread_budget() == 10 ** 9
