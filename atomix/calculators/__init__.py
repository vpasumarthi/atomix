"""Calculator interfaces for atomix.

Heavy scientific dependencies (ase, numpy, pymatgen) are imported lazily so
that ``import atomix.calculators`` succeeds on a minimal (click-only) install.
Accessing a specific class triggers the import of its submodule.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "VASPCalculator": "atomix.calculators.vasp",
    "VASPFrame": "atomix.calculators.vasp_segments",
    "VASPReadWarning": "atomix.calculators.vasp_segments",
    "VASPSegment": "atomix.calculators.vasp_segments",
    "VASPSegmentReader": "atomix.calculators.vasp_segments",
    "VASPSegmentReadResult": "atomix.calculators.vasp_segments",
    "VASPTextData": "atomix.calculators.vasp_segments",
    "MLIPCalculator": "atomix.calculators.mlip",
    "MACECalculator": "atomix.calculators.mlip",
    "NequIPCalculator": "atomix.calculators.mlip",
    "get_mlip_calculator": "atomix.calculators.mlip",
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
    from atomix.calculators.mlip import (
        MACECalculator,
        MLIPCalculator,
        NequIPCalculator,
        get_mlip_calculator,
    )
    from atomix.calculators.vasp import VASPCalculator
    from atomix.calculators.vasp_segments import (
        VASPFrame,
        VASPReadWarning,
        VASPSegment,
        VASPSegmentReader,
        VASPSegmentReadResult,
        VASPTextData,
    )
