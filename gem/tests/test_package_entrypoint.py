"""``gem.run(config)`` must stay callable for the whole life of a process.

``gem/run/`` is a subpackage and ``run()`` is the top-level entry point. The import system binds a
submodule onto its parent package, and that import happens lazily -- the first ``gem.run(...)`` call
reaches ``init_setup``, which does ``from gem.run.run_gem_prf_analysis import GEMpRFAnalysis`` --
so the function was replaced by the module *while the first call was running*. The first analysis in
a process worked and every later one died with ``TypeError: 'module' object is not callable``, which
is what disabled the second of the two end-to-end benchmark tests.

Nothing here touches CuPy: ``gem/__init__.py`` imports lazily and ``gem/run/__init__.py`` is empty,
so these run anywhere.
"""
import importlib

import pytest


def test_run_is_callable_before_anything_else_is_imported():
    import gem

    assert callable(gem.run)


def test_run_survives_the_run_subpackage_being_imported():
    """The regression itself: importing the subpackage must not shadow the entry point."""
    import gem

    importlib.import_module("gem.run")

    assert callable(gem.run), "the gem.run subpackage shadowed the run() entry point"


def test_run_subpackage_is_still_importable():
    """The fix must not cost us the module path -- plenty of code imports through gem.run.*."""
    import gem

    assert importlib.import_module("gem.run").__name__ == "gem.run"
    assert callable(gem.run)


def test_from_gem_import_run_gives_the_function():
    """`from gem import run` goes through the same attribute lookup and must also get the callable."""
    importlib.import_module("gem.run")
    from gem import run

    assert callable(run)


def test_unknown_attribute_still_raises():
    """The lazy submodule loader must keep rejecting names that are not submodules."""
    import gem

    with pytest.raises(AttributeError):
        gem.definitely_not_a_submodule
