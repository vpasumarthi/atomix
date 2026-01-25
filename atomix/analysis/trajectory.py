"""Trajectory analysis utilities for atomix."""

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms


class TrajectoryAnalyzer:
    """Analyze MD trajectories.

    Parameters
    ----------
    trajectory : list[Atoms] | Path | str
        Trajectory as list of Atoms or path to trajectory file.
    """

    def __init__(self, trajectory: list[Atoms] | Path | str) -> None:
        if isinstance(trajectory, (Path, str)):
            self._load_trajectory(trajectory)
        else:
            self.trajectory = trajectory

    def _load_trajectory(self, path: Path | str) -> None:
        """Load trajectory from file."""
        from ase.io import read

        self.trajectory = read(str(path), index=":")

    def rdf(
        self,
        elements: tuple[str, str],
        rmax: float = 6.0,
        nbins: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate radial distribution function.

        Computes g(r) averaged over trajectory frames.

        Parameters
        ----------
        elements : tuple[str, str]
            Pair of element symbols (e.g., ('O', 'O') or ('Cu', 'O')).
        rmax : float
            Maximum distance in Angstrom.
        nbins : int
            Number of histogram bins.

        Returns
        -------
        tuple[ndarray, ndarray]
            r values (bin centers) and g(r) values.
        """
        from ase.geometry import get_distances

        dr = rmax / nbins
        r_edges = np.linspace(0, rmax, nbins + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        hist_sum = np.zeros(nbins)

        elem1, elem2 = elements
        same_element = elem1 == elem2

        for atoms in self.trajectory:
            symbols = np.array(atoms.get_chemical_symbols())
            idx1 = np.where(symbols == elem1)[0]
            idx2 = np.where(symbols == elem2)[0]

            if len(idx1) == 0 or len(idx2) == 0:
                continue

            pos1 = atoms.positions[idx1]
            pos2 = atoms.positions[idx2]
            cell = atoms.get_cell()
            pbc = atoms.get_pbc()

            # Get all pairwise distances
            _, distances = get_distances(pos1, pos2, cell=cell, pbc=pbc)

            # Flatten and filter
            dists = distances.flatten()
            if same_element:
                # Exclude self-distances (diagonal)
                mask = dists > 0.1
                dists = dists[mask]

            # Histogram
            hist, _ = np.histogram(dists, bins=r_edges)
            hist_sum += hist

        # Normalize by number of frames
        n_frames = len(self.trajectory)
        if n_frames == 0:
            return r_centers, np.zeros(nbins)

        hist_avg = hist_sum / n_frames

        # Calculate normalization factor for g(r)
        # g(r) = n(r) / (rho * 4 * pi * r^2 * dr)
        atoms = self.trajectory[0]
        symbols = np.array(atoms.get_chemical_symbols())
        n1 = np.sum(symbols == elem1)
        n2 = np.sum(symbols == elem2)
        volume = atoms.get_volume()

        if same_element:
            # For same element: n_pairs = n1 * (n1 - 1) / 2
            rho = n1 / volume
            n_ref = n1
        else:
            # For different elements: n_pairs = n1 * n2
            rho = n2 / volume
            n_ref = n1

        # Shell volumes
        shell_volumes = (4 / 3) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
        ideal_count = rho * shell_volumes * n_ref

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            g_r = np.where(ideal_count > 0, hist_avg / ideal_count, 0.0)

        return r_centers, g_r

    def msd(
        self,
        element: str | None = None,
        unwrap: bool = True,
    ) -> np.ndarray:
        """Calculate mean squared displacement.

        Uses the efficient algorithm that computes MSD for all time lags.

        Parameters
        ----------
        element : str | None
            Element to analyze, or all atoms if None.
        unwrap : bool
            Whether to unwrap periodic boundary conditions.

        Returns
        -------
        ndarray
            MSD as function of time lag (in Angstrom^2).
        """
        n_frames = len(self.trajectory)
        if n_frames < 2:
            return np.array([0.0])

        # Get atom indices
        atoms = self.trajectory[0]
        if element is not None:
            symbols = np.array(atoms.get_chemical_symbols())
            indices = np.where(symbols == element)[0]
            if len(indices) == 0:
                return np.zeros(n_frames)
        else:
            indices = np.arange(len(atoms))

        # Extract positions for selected atoms
        positions = np.array([frame.positions[indices] for frame in self.trajectory])

        # Unwrap positions across periodic boundaries
        if unwrap:
            positions = self._unwrap_positions(positions)

        # Calculate MSD using direct method
        # MSD(t) = <|r(t0 + t) - r(t0)|^2>
        msd = np.zeros(n_frames)
        for lag in range(n_frames):
            if lag == 0:
                msd[0] = 0.0
            else:
                displacements = positions[lag:] - positions[:-lag]
                sq_disp = np.sum(displacements**2, axis=2)  # Sum over xyz
                msd[lag] = np.mean(sq_disp)  # Average over atoms and time origins

        return msd

    def _unwrap_positions(self, positions: np.ndarray) -> np.ndarray:
        """Unwrap positions across periodic boundaries.

        Parameters
        ----------
        positions : ndarray
            Shape (n_frames, n_atoms, 3).

        Returns
        -------
        ndarray
            Unwrapped positions.
        """
        atoms = self.trajectory[0]
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        if not np.any(pbc):
            return positions

        unwrapped = positions.copy()
        cell_lengths = np.linalg.norm(cell, axis=1)

        for i in range(1, len(positions)):
            diff = positions[i] - positions[i - 1]
            for dim in range(3):
                if pbc[dim]:
                    # Detect jumps larger than half cell
                    jumps = diff[:, dim] / cell_lengths[dim]
                    corrections = np.round(jumps)
                    unwrapped[i:, :, dim] -= corrections * cell_lengths[dim]

        return unwrapped

    def diffusion_coefficient(
        self,
        element: str | None = None,
        timestep: float = 1.0,
        fit_fraction: tuple[float, float] = (0.2, 0.8),
    ) -> dict[str, float]:
        """Calculate diffusion coefficient from MSD.

        Fits the linear regime of MSD to extract D.
        MSD = 6 * D * t (3D diffusion)

        Parameters
        ----------
        element : str | None
            Element to analyze.
        timestep : float
            MD timestep in femtoseconds.
        fit_fraction : tuple[float, float]
            Fraction of trajectory to use for linear fit (start, end).
            Default (0.2, 0.8) avoids ballistic regime and poor statistics.

        Returns
        -------
        dict[str, float]
            Dictionary with 'D' (diffusion coefficient in cm^2/s),
            'D_error' (fit error), and 'r_squared' (fit quality).
        """
        msd = self.msd(element=element)
        n_frames = len(msd)

        if n_frames < 10:
            return {"D": 0.0, "D_error": 0.0, "r_squared": 0.0}

        # Time array in femtoseconds
        time_fs = np.arange(n_frames) * timestep

        # Select fitting region
        start_idx = int(fit_fraction[0] * n_frames)
        end_idx = int(fit_fraction[1] * n_frames)
        start_idx = max(1, start_idx)  # Avoid t=0

        t_fit = time_fs[start_idx:end_idx]
        msd_fit = msd[start_idx:end_idx]

        if len(t_fit) < 3:
            return {"D": 0.0, "D_error": 0.0, "r_squared": 0.0}

        # Linear fit: MSD = 6*D*t (3D)
        # y = m*x, where m = 6*D
        coeffs = np.polyfit(t_fit, msd_fit, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # D in Å²/fs
        d_ang2_fs = slope / 6.0

        # Convert to cm²/s
        # 1 Å² = 1e-16 cm², 1 fs = 1e-15 s
        # D [cm²/s] = D [Å²/fs] * 1e-16 / 1e-15 = D [Å²/fs] * 0.1
        d_cm2_s = d_ang2_fs * 0.1

        # Calculate R² for fit quality
        msd_pred = slope * t_fit + intercept
        ss_res = np.sum((msd_fit - msd_pred) ** 2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Estimate error from residuals
        if len(t_fit) > 2:
            residuals = msd_fit - msd_pred
            std_err = np.sqrt(np.sum(residuals**2) / (len(t_fit) - 2))
            slope_err = std_err / np.sqrt(np.sum((t_fit - np.mean(t_fit)) ** 2))
            d_error = (slope_err / 6.0) * 0.1
        else:
            d_error = 0.0

        return {
            "D": d_cm2_s,
            "D_error": d_error,
            "r_squared": r_squared,
        }
