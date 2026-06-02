"""AI/NL generation module for atomix.

The generator imports ase at module scope, so it is imported lazily here to
keep ``import atomix.ai`` working on a minimal (click-only) install.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "NLGenerator": "atomix.ai.generator",
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
    from atomix.ai.generator import NLGenerator
