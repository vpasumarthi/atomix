"""Analysis modules for atomix.

Heavy scientific dependencies (ase, numpy) are imported lazily so that
``import atomix.analysis`` succeeds on a minimal (click-only) install.
Accessing a specific class triggers the import of its submodule.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "EnergyAnalyzer": "atomix.analysis.energy",
    "TrajectoryAnalyzer": "atomix.analysis.trajectory",
    "AdsorptionAnalyzer": "atomix.analysis.adsorption",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from atomix.analysis.adsorption import AdsorptionAnalyzer
    from atomix.analysis.energy import EnergyAnalyzer
    from atomix.analysis.trajectory import TrajectoryAnalyzer
