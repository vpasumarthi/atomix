"""Surface site identification for catalysis workflows."""

from typing import Any

import numpy as np
from ase import Atoms


class SurfaceSite:
    """Represent an adsorption site on a surface.

    Parameters
    ----------
    position : ndarray
        3D coordinates of the site.
    site_type : str
        Type of site ('top', 'bridge', 'hollow', 'fcc', 'hcp').
    atoms_indices : list[int]
        Indices of surface atoms defining this site.
    """

    def __init__(
        self,
        position: np.ndarray,
        site_type: str,
        atoms_indices: list[int],
    ) -> None:
        self.position = np.asarray(position)
        self.site_type = site_type
        self.atoms_indices = atoms_indices

    def __repr__(self) -> str:
        return f"SurfaceSite({self.site_type}, {self.position})"


def find_surface_sites(
    slab: Atoms,
    height: float = 1.5,
    symprec: float = 0.1,
) -> list[SurfaceSite]:
    """Find unique adsorption sites on a surface slab.

    Parameters
    ----------
    slab : Atoms
        Surface slab structure.
    height : float
        Height above surface for site positions.
    symprec : float
        Symmetry precision for identifying unique sites.

    Returns
    -------
    list[SurfaceSite]
        List of unique adsorption sites.
    """
    raise NotImplementedError


def add_adsorbate_at_site(
    slab: Atoms,
    adsorbate: Atoms | str,
    site: SurfaceSite,
    height: float = 1.5,
) -> Atoms:
    """Add adsorbate at specified site.

    Parameters
    ----------
    slab : Atoms
        Surface slab.
    adsorbate : Atoms | str
        Adsorbate structure or element symbol.
    site : SurfaceSite
        Adsorption site.
    height : float
        Height of adsorbate above site.

    Returns
    -------
    Atoms
        Slab with adsorbate added.
    """
    raise NotImplementedError
