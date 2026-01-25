"""Bulk defect site identification."""

from typing import Any

import numpy as np
from ase import Atoms


class BulkSite:
    """Represent a site in bulk material (vacancy, interstitial, etc.).

    Parameters
    ----------
    position : ndarray
        3D coordinates of the site.
    site_type : str
        Type of site ('vacancy', 'interstitial', 'substitution').
    """

    def __init__(
        self,
        position: np.ndarray,
        site_type: str,
    ) -> None:
        self.position = np.asarray(position)
        self.site_type = site_type

    def __repr__(self) -> str:
        return f"BulkSite({self.site_type}, {self.position})"


def find_interstitial_sites(
    bulk: Atoms,
    min_dist: float = 1.5,
) -> list[BulkSite]:
    """Find interstitial sites in bulk structure.

    Parameters
    ----------
    bulk : Atoms
        Bulk structure.
    min_dist : float
        Minimum distance from existing atoms.

    Returns
    -------
    list[BulkSite]
        List of interstitial sites.
    """
    raise NotImplementedError


def create_vacancy(
    bulk: Atoms,
    index: int,
) -> Atoms:
    """Create vacancy by removing atom at index.

    Parameters
    ----------
    bulk : Atoms
        Bulk structure.
    index : int
        Index of atom to remove.

    Returns
    -------
    Atoms
        Structure with vacancy.
    """
    atoms = bulk.copy()
    del atoms[index]
    return atoms
