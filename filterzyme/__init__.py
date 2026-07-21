# filterzyme/__init__.py
"""Deprecated compatibility shim.

``filterzyme`` was renamed to ``structurezyme``. This package re-exports the
new location, emits a :class:`DeprecationWarning` on import, and lazily forwards
submodule access (e.g. ``filterzyme.pipeline`` -> ``structurezyme.pipeline``).

Scheduled for removal in 0.2.0.
"""
import warnings as _warnings
from structurezyme import *  # noqa: F401,F403
from structurezyme import __version__  # noqa: F401

_warnings.warn(
    "`filterzyme` is deprecated; import `structurezyme` instead. "
    "The `filterzyme` alias will be removed in 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

import importlib as _importlib
import sys as _sys
from importlib.abc import MetaPathFinder as _MetaPathFinder
from importlib.util import spec_from_loader as _spec_from_loader


def __getattr__(name):
    module = _importlib.import_module(f"structurezyme.{name}")
    _sys.modules[f"filterzyme.{name}"] = module
    return module


class _FilterzymeSubmoduleLoader:
    """Loads ``filterzyme.<sub>`` by aliasing ``structurezyme.<sub>``."""

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        return _importlib.import_module(self._target)

    def exec_module(self, module):  # already fully initialised
        return None


class _FilterzymeFinder(_MetaPathFinder):
    """Forwards ``filterzyme.<sub>`` imports to ``structurezyme.<sub>``.

    Needed so that submodule import statements such as
    ``from filterzyme.pipeline import Pipeline`` resolve, in addition to the
    attribute-style forwarding provided by module-level ``__getattr__``.
    """

    _prefix = "filterzyme."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix):
            return None
        target_name = "structurezyme." + fullname[len(self._prefix):]
        try:
            _importlib.util.find_spec(target_name)
        except (ImportError, AttributeError, ValueError):
            return None
        loader = _FilterzymeSubmoduleLoader(target_name)
        return _spec_from_loader(fullname, loader)


if not any(isinstance(f, _FilterzymeFinder) for f in _sys.meta_path):
    _sys.meta_path.append(_FilterzymeFinder())
