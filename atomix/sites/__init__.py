"""Site identification modules for atomix.

Heavy scientific dependencies (ase, numpy, scipy) are imported lazily so that
``import atomix.sites`` succeeds on a minimal (click-only) install. Accessing a
specific class triggers the import of its submodule.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "SurfaceSite": "atomix.sites.surface",
    "find_surface_sites": "atomix.sites.surface",
    "add_adsorbate_at_site": "atomix.sites.surface",
    "BulkSite": "atomix.sites.bulk",
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
    from atomix.sites.bulk import BulkSite
    from atomix.sites.surface import (
        SurfaceSite,
        add_adsorbate_at_site,
        find_surface_sites,
    )
