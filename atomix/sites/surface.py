"""Surface site identification for catalysis workflows."""

from typing import Any

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay


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

    Identifies top, bridge, and hollow (fcc/hcp) sites on the surface
    using Delaunay triangulation of surface atoms.

    Parameters
    ----------
    slab : Atoms
        Surface slab structure.
    height : float
        Height above surface for site positions.
    symprec : float
        Distance tolerance for identifying unique sites (Angstrom).

    Returns
    -------
    list[SurfaceSite]
        List of unique adsorption sites.
    """
    positions = slab.get_positions()
    cell = slab.get_cell()

    # Identify surface atoms (top layer based on z-coordinate)
    z_coords = positions[:, 2]
    z_max = z_coords.max()
    z_tolerance = 0.5  # Atoms within 0.5 Å of top are surface atoms
    surface_mask = z_coords >= (z_max - z_tolerance)
    surface_indices = np.where(surface_mask)[0]
    surface_positions = positions[surface_indices]

    if len(surface_indices) < 3:
        # Not enough surface atoms for triangulation
        sites = []
        for idx in surface_indices:
            pos = positions[idx].copy()
            pos[2] = z_max + height
            sites.append(SurfaceSite(pos, "top", [int(idx)]))
        return sites

    # Use 2D Delaunay triangulation on surface atoms (x, y only)
    points_2d = surface_positions[:, :2]

    # Handle periodic boundary conditions by replicating surface atoms
    replicated_points = []
    replicated_indices = []
    shifts = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    for di, dj in shifts:
        shift = di * cell[0, :2] + dj * cell[1, :2]
        for local_idx, global_idx in enumerate(surface_indices):
            replicated_points.append(points_2d[local_idx] + shift)
            replicated_indices.append((global_idx, di, dj))

    replicated_points = np.array(replicated_points)
    tri = Delaunay(replicated_points)

    sites: list[SurfaceSite] = []
    seen_positions: list[np.ndarray] = []

    def is_unique(pos: np.ndarray) -> bool:
        """Check if position is unique within symprec."""
        for seen in seen_positions:
            if np.linalg.norm(pos[:2] - seen[:2]) < symprec:
                return False
        return True

    def wrap_to_cell(pos: np.ndarray) -> np.ndarray:
        """Wrap position to unit cell."""
        frac = np.linalg.solve(cell.T, pos)
        frac[:2] = frac[:2] % 1.0
        return cell.T @ frac

    def is_in_cell(pos: np.ndarray) -> bool:
        """Check if position is inside the unit cell."""
        frac = np.linalg.solve(cell.T, pos)
        return (0 <= frac[0] < 1) and (0 <= frac[1] < 1)

    # Top sites - directly above each surface atom
    for local_idx, global_idx in enumerate(surface_indices):
        pos = surface_positions[local_idx].copy()
        pos[2] = z_max + height
        if is_unique(pos):
            sites.append(SurfaceSite(pos, "top", [int(global_idx)]))
            seen_positions.append(pos.copy())

    # Bridge and hollow sites from Delaunay triangles
    n_original = len(surface_indices)
    for simplex in tri.simplices:
        # Get original indices (unwrap periodic images)
        orig_indices = []
        orig_positions = []
        for vertex in simplex:
            global_idx, di, dj = replicated_indices[vertex]
            orig_indices.append(int(global_idx))
            orig_positions.append(replicated_points[vertex])

        orig_positions = np.array(orig_positions)

        # Check if at least one vertex is from the original cell
        any_in_original = any(
            replicated_indices[v][1] == 0 and replicated_indices[v][2] == 0
            for v in simplex
        )
        if not any_in_original:
            continue

        # Bridge sites - midpoints of triangle edges
        for i in range(3):
            j = (i + 1) % 3
            bridge_pos = (orig_positions[i] + orig_positions[j]) / 2
            bridge_3d = np.array([bridge_pos[0], bridge_pos[1], z_max + height])
            wrapped = wrap_to_cell(bridge_3d)

            if is_in_cell(wrapped) and is_unique(wrapped):
                bridge_indices = [orig_indices[i], orig_indices[j]]
                sites.append(SurfaceSite(wrapped, "bridge", bridge_indices))
                seen_positions.append(wrapped.copy())

        # Hollow site - center of triangle
        hollow_pos = orig_positions.mean(axis=0)
        hollow_3d = np.array([hollow_pos[0], hollow_pos[1], z_max + height])
        wrapped = wrap_to_cell(hollow_3d)

        if is_in_cell(wrapped) and is_unique(wrapped):
            # Determine fcc vs hcp based on atom below
            # fcc: hollow has no atom directly below in next layer
            # hcp: hollow has atom directly below
            hollow_xy = wrapped[:2]
            z_second_layer = z_max - 2.0  # Approximate second layer
            second_layer_mask = (z_coords < z_max - 0.5) & (z_coords > z_second_layer)
            second_layer_indices = np.where(second_layer_mask)[0]

            site_type = "hollow"
            if len(second_layer_indices) > 0:
                second_layer_xy = positions[second_layer_indices, :2]
                # Check for periodic images too
                min_dist = float("inf")
                for di, dj in shifts[:5]:  # Check original + 4 neighbors
                    shift = di * cell[0, :2] + dj * cell[1, :2]
                    dists = np.linalg.norm(second_layer_xy + shift - hollow_xy, axis=1)
                    min_dist = min(min_dist, dists.min())
                site_type = "hcp" if min_dist < 1.0 else "fcc"

            sites.append(SurfaceSite(wrapped, site_type, list(set(orig_indices))))
            seen_positions.append(wrapped.copy())

    return sites


def add_adsorbate_at_site(
    slab: Atoms,
    adsorbate: Atoms | str,
    site: SurfaceSite,
    height: float | None = None,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Atoms:
    """Add adsorbate at specified site.

    Parameters
    ----------
    slab : Atoms
        Surface slab.
    adsorbate : Atoms | str
        Adsorbate structure or element symbol (e.g., 'O', 'CO', 'OH').
    site : SurfaceSite
        Adsorption site from find_surface_sites().
    height : float | None
        Height of adsorbate above site. If None, uses the height
        already encoded in site.position.
    offset : tuple[float, float, float]
        Additional offset for adsorbate placement (x, y, z).

    Returns
    -------
    Atoms
        Copy of slab with adsorbate added.
    """
    from ase import Atom
    from ase.build import molecule

    # Create copy of slab
    result = slab.copy()

    # Handle adsorbate input
    if isinstance(adsorbate, str):
        # Try as molecule first, then as single atom
        try:
            ads_atoms = molecule(adsorbate)
        except (KeyError, ValueError):
            # Single atom
            ads_atoms = Atoms(adsorbate)
    else:
        ads_atoms = adsorbate.copy()

    # Determine position
    if height is not None:
        # Override z with explicit height above surface
        z_max = slab.get_positions()[:, 2].max()
        pos = np.array([site.position[0], site.position[1], z_max + height])
    else:
        # Use position from site (already includes height)
        pos = site.position.copy()

    # Apply offset
    pos += np.array(offset)

    # Center adsorbate and place at site
    if len(ads_atoms) == 1:
        # Single atom - just set position
        ads_atoms.positions[0] = pos
    else:
        # Multi-atom adsorbate - center on binding atom (first atom)
        # and orient perpendicular to surface (z-up)
        ads_atoms.center()
        # Shift so first atom is at the site position
        shift = pos - ads_atoms.positions[0]
        ads_atoms.positions += shift

    # Add adsorbate atoms to slab
    for atom in ads_atoms:
        result.append(Atom(atom.symbol, atom.position))

    return result
