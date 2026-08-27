"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2025, Medical University of Vienna",
"@Desc    :   None",

"""

# Lazy loader to avoid circular imports and behave identical to source structure.

import importlib
import os
import sys
import types

# Dynamically detect all submodules in the gem directory
_package_dir = os.path.dirname(__file__)
_submodules = {
    name
    for name in os.listdir(_package_dir)
    if os.path.isdir(os.path.join(_package_dir, name))
    and not name.startswith("_")
}

# expose run_gem.run at top-level:
def _run_analysis(*args, **kwargs):
    from .run_gem import run as _run
    return _run(*args, **kwargs)


class _GemPackage(types.ModuleType):
    """Keeps ``gem.run`` callable even once the ``gem.run`` *subpackage* has been imported.

    ``gem/run/`` is a subpackage and ``run()`` is the top-level entry point, so both want the same
    attribute. The import system binds a submodule onto its parent package, and here that import
    happens lazily -- inside the first ``gem.run(...)`` call, by way of ``init_setup`` -- so the
    first call succeeded and every later one in the same process raised
    ``TypeError: 'module' object is not callable``. A property is a data descriptor and therefore
    wins over whatever the import system writes into the module's ``__dict__``; the setter absorbs
    that write so it does not warn. ``import gem.run.x`` keeps working either way, because the
    import machinery resolves parent packages through ``sys.modules``, not through this attribute.
    """

    @property
    def run(self):
        return _run_analysis

    @run.setter
    def run(self, value):
        self.__dict__["_run_subpackage"] = value


sys.modules[__name__].__class__ = _GemPackage


def __getattr__(name):
    if name in _submodules:
        return importlib.import_module(f"gem.{name}")
    raise AttributeError(f"module 'gem' has no attribute '{name}'")