"""Trajectory analysis utilities for atomix."""

from pathlib import Path
from typing import Any

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
    ) -> tuple[Any, Any]:
        """Calculate radial distribution function.

        Parameters
        ----------
        elements : tuple[str, str]
            Pair of element symbols.
        rmax : float
            Maximum distance.
        nbins : int
            Number of bins.

        Returns
        -------
        tuple[ndarray, ndarray]
            r values and g(r) values.
        """
        raise NotImplementedError

    def msd(self, element: str | None = None) -> Any:
        """Calculate mean squared displacement.

        Parameters
        ----------
        element : str | None
            Element to analyze, or all atoms if None.

        Returns
        -------
        ndarray
            MSD as function of time.
        """
        raise NotImplementedError

    def diffusion_coefficient(self, element: str | None = None) -> float:
        """Calculate diffusion coefficient from MSD.

        Parameters
        ----------
        element : str | None
            Element to analyze.

        Returns
        -------
        float
            Diffusion coefficient in appropriate units.
        """
        raise NotImplementedError
