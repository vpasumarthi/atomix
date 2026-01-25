"""Screening workflows for MLIP pre-screening with DFT validation.

Provides workflows that use fast MLIP calculations to screen large
numbers of candidate structures, then validate promising candidates
with accurate DFT calculations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import Atoms

from atomix.calculators.mlip import MLIPCalculator


@dataclass
class ScreeningResult:
    """Result from screening a single structure.

    Attributes
    ----------
    atoms : Atoms
        The structure that was screened.
    mlip_energy : float | None
        Energy from MLIP calculation (eV).
    mlip_forces : np.ndarray | None
        Forces from MLIP (eV/Å).
    dft_energy : float | None
        Energy from DFT validation (if performed).
    dft_forces : np.ndarray | None
        Forces from DFT validation (if performed).
    selected_for_dft : bool
        Whether this structure was selected for DFT validation.
    rank : int
        Ranking among all screened structures (1 = best).
    metadata : dict
        Additional metadata (site info, identifiers, etc.).
    """

    atoms: Atoms
    mlip_energy: float | None = None
    mlip_forces: np.ndarray | None = None
    dft_energy: float | None = None
    dft_forces: np.ndarray | None = None
    selected_for_dft: bool = False
    rank: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def energy_error(self) -> float | None:
        """Absolute error between MLIP and DFT energy (eV)."""
        if self.mlip_energy is not None and self.dft_energy is not None:
            return abs(self.mlip_energy - self.dft_energy)
        return None

    @property
    def force_mae(self) -> float | None:
        """Mean absolute error of forces (eV/Å)."""
        if self.mlip_forces is not None and self.dft_forces is not None:
            return float(np.mean(np.abs(self.mlip_forces - self.dft_forces)))
        return None


@dataclass
class ScreeningConfig:
    """Configuration for screening workflow.

    Attributes
    ----------
    top_n : int
        Number of top candidates to select for DFT validation.
    top_fraction : float | None
        Alternative: select top fraction of candidates (0.0-1.0).
    energy_window : float | None
        Select structures within this energy window from minimum (eV).
    max_force_threshold : float | None
        Filter out structures with max force above threshold (eV/Å).
    custom_filter : Callable | None
        Custom filter function: (ScreeningResult) -> bool.
    """

    top_n: int = 10
    top_fraction: float | None = None
    energy_window: float | None = None
    max_force_threshold: float | None = None
    custom_filter: Callable[[ScreeningResult], bool] | None = None


class ScreeningWorkflow:
    """MLIP screening workflow with optional DFT validation.

    This workflow enables efficient exploration of large configuration
    spaces by using fast MLIP calculations to identify promising
    candidates, which are then validated with accurate DFT.

    Parameters
    ----------
    mlip_calculator : MLIPCalculator
        MLIP calculator for fast screening.
    config : ScreeningConfig | None
        Screening configuration. Defaults to top 10 candidates.

    Examples
    --------
    >>> from atomix.calculators.mlip import MACECalculator
    >>> from atomix.core.screening import ScreeningWorkflow, ScreeningConfig
    >>>
    >>> # Set up screening
    >>> mlip = MACECalculator(model="medium", device="cuda")
    >>> config = ScreeningConfig(top_n=5, energy_window=0.5)
    >>> workflow = ScreeningWorkflow(mlip, config)
    >>>
    >>> # Screen candidate structures
    >>> results = workflow.screen(candidates)
    >>> selected = workflow.get_selected()
    >>>
    >>> # Get structures for DFT validation
    >>> for result in selected:
    >>>     # Run DFT on result.atoms
    >>>     pass
    """

    def __init__(
        self,
        mlip_calculator: MLIPCalculator,
        config: ScreeningConfig | None = None,
    ) -> None:
        self.mlip = mlip_calculator
        self.config = config or ScreeningConfig()
        self._results: list[ScreeningResult] = []

    def screen(
        self,
        structures: list[Atoms],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[ScreeningResult]:
        """Screen structures with MLIP and rank by energy.

        Parameters
        ----------
        structures : list[Atoms]
            Candidate structures to screen.
        metadata : list[dict] | None
            Optional metadata for each structure (e.g., site labels).

        Returns
        -------
        list[ScreeningResult]
            Screening results sorted by energy (lowest first).
        """
        if metadata is None:
            metadata = [{} for _ in structures]

        if len(metadata) != len(structures):
            raise ValueError("metadata length must match structures length")

        # Calculate energies with MLIP
        self._results = []
        for atoms, meta in zip(structures, metadata):
            calc_result = self.mlip.calculate(atoms)

            result = ScreeningResult(
                atoms=atoms.copy(),
                mlip_energy=calc_result.get("energy"),
                mlip_forces=calc_result.get("forces"),
                metadata=meta,
            )
            self._results.append(result)

        # Sort by energy
        valid_results = [r for r in self._results if r.mlip_energy is not None]
        valid_results.sort(key=lambda r: r.mlip_energy)  # type: ignore

        # Assign ranks
        for i, result in enumerate(valid_results):
            result.rank = i + 1

        # Mark selected structures
        self._mark_selected(valid_results)

        # Put back invalid results at the end
        invalid = [r for r in self._results if r.mlip_energy is None]
        self._results = valid_results + invalid

        return self._results

    def _mark_selected(self, sorted_results: list[ScreeningResult]) -> None:
        """Mark structures selected for DFT validation."""
        if not sorted_results:
            return

        min_energy = sorted_results[0].mlip_energy
        config = self.config

        for result in sorted_results:
            selected = True

            # Check energy window
            if config.energy_window is not None and min_energy is not None:
                if result.mlip_energy is not None:
                    if result.mlip_energy - min_energy > config.energy_window:
                        selected = False

            # Check force threshold
            if config.max_force_threshold is not None and result.mlip_forces is not None:
                max_force = float(np.max(np.linalg.norm(result.mlip_forces, axis=1)))
                if max_force > config.max_force_threshold:
                    selected = False

            # Check custom filter
            if config.custom_filter is not None:
                if not config.custom_filter(result):
                    selected = False

            result.selected_for_dft = selected

        # Apply top_n or top_fraction limit
        selected = [r for r in sorted_results if r.selected_for_dft]

        if config.top_fraction is not None:
            n_select = max(1, int(len(sorted_results) * config.top_fraction))
        else:
            n_select = config.top_n

        # Limit to top_n/fraction
        for i, result in enumerate(selected):
            if i >= n_select:
                result.selected_for_dft = False

    def get_selected(self) -> list[ScreeningResult]:
        """Get structures selected for DFT validation.

        Returns
        -------
        list[ScreeningResult]
            Results marked for DFT validation.
        """
        return [r for r in self._results if r.selected_for_dft]

    def get_top_n(self, n: int) -> list[ScreeningResult]:
        """Get top N structures by MLIP energy.

        Parameters
        ----------
        n : int
            Number of structures to return.

        Returns
        -------
        list[ScreeningResult]
            Top N results sorted by energy.
        """
        valid = [r for r in self._results if r.mlip_energy is not None]
        return sorted(valid, key=lambda r: r.mlip_energy)[:n]  # type: ignore

    def get_within_window(self, window: float) -> list[ScreeningResult]:
        """Get structures within energy window of minimum.

        Parameters
        ----------
        window : float
            Energy window in eV from minimum.

        Returns
        -------
        list[ScreeningResult]
            Results within the energy window.
        """
        valid = [r for r in self._results if r.mlip_energy is not None]
        if not valid:
            return []

        valid.sort(key=lambda r: r.mlip_energy)  # type: ignore
        min_energy = valid[0].mlip_energy

        return [
            r for r in valid
            if r.mlip_energy is not None and r.mlip_energy - min_energy <= window  # type: ignore
        ]

    def add_dft_result(
        self,
        index: int,
        dft_energy: float,
        dft_forces: np.ndarray | None = None,
    ) -> None:
        """Add DFT validation result to a screened structure.

        Parameters
        ----------
        index : int
            Index in results list.
        dft_energy : float
            DFT energy (eV).
        dft_forces : np.ndarray | None
            DFT forces (eV/Å).
        """
        if 0 <= index < len(self._results):
            self._results[index].dft_energy = dft_energy
            self._results[index].dft_forces = dft_forces

    def get_validation_statistics(self) -> dict[str, float]:
        """Calculate statistics on MLIP vs DFT agreement.

        Returns
        -------
        dict[str, float]
            Statistics including:
            - energy_mae: Mean absolute error of energies
            - energy_max_error: Maximum energy error
            - force_mae: Mean absolute error of forces
            - n_validated: Number of structures with DFT results
        """
        validated = [
            r for r in self._results
            if r.mlip_energy is not None and r.dft_energy is not None
        ]

        if not validated:
            return {
                "energy_mae": float("nan"),
                "energy_max_error": float("nan"),
                "force_mae": float("nan"),
                "n_validated": 0,
            }

        energy_errors = [abs(r.mlip_energy - r.dft_energy) for r in validated]  # type: ignore

        stats = {
            "energy_mae": float(np.mean(energy_errors)),
            "energy_max_error": float(np.max(energy_errors)),
            "n_validated": len(validated),
        }

        # Force statistics if available
        force_errors = [r.force_mae for r in validated if r.force_mae is not None]
        stats["force_mae"] = float(np.mean(force_errors)) if force_errors else float("nan")

        return stats

    @property
    def results(self) -> list[ScreeningResult]:
        """Get all screening results."""
        return self._results

    def to_dataframe(self) -> Any:
        """Export results to pandas DataFrame if available.

        Returns
        -------
        DataFrame
            Results as pandas DataFrame.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        data = []
        for r in self._results:
            row = {
                "rank": r.rank,
                "mlip_energy": r.mlip_energy,
                "dft_energy": r.dft_energy,
                "energy_error": r.energy_error,
                "force_mae": r.force_mae,
                "selected": r.selected_for_dft,
                "n_atoms": len(r.atoms),
                "formula": r.atoms.get_chemical_formula(),
            }
            row.update(r.metadata)
            data.append(row)

        return pd.DataFrame(data)


class AdsorptionScreening:
    """Specialized screening for adsorption site enumeration.

    Combines site enumeration with MLIP screening to efficiently
    find optimal adsorption configurations.

    Parameters
    ----------
    mlip_calculator : MLIPCalculator
        MLIP calculator for screening.
    config : ScreeningConfig | None
        Screening configuration.
    """

    def __init__(
        self,
        mlip_calculator: MLIPCalculator,
        config: ScreeningConfig | None = None,
    ) -> None:
        self.mlip = mlip_calculator
        self.config = config or ScreeningConfig()

    def screen_sites(
        self,
        slab: Atoms,
        adsorbate: Atoms,
        sites: list[tuple[float, float, float]],
        height: float = 2.0,
        site_labels: list[str] | None = None,
    ) -> list[ScreeningResult]:
        """Screen adsorption at multiple sites.

        Parameters
        ----------
        slab : Atoms
            Surface slab structure.
        adsorbate : Atoms
            Adsorbate molecule/atom.
        sites : list[tuple]
            List of (x, y, z) site positions.
        height : float
            Adsorbate height above site (Å).
        site_labels : list[str] | None
            Labels for each site (e.g., 'top', 'bridge', 'hollow').

        Returns
        -------
        list[ScreeningResult]
            Screening results for each site.
        """
        from atomix.analysis.adsorption import AdsorptionAnalyzer

        if site_labels is None:
            site_labels = [f"site_{i}" for i in range(len(sites))]

        # Generate structures
        structures = []
        metadata = []

        for site, label in zip(sites, site_labels):
            # Create slab+adsorbate structure
            structure = slab.copy()

            # Position adsorbate at site
            ads = adsorbate.copy()
            ads.translate(np.array(site) + np.array([0, 0, height]))
            structure.extend(ads)

            structures.append(structure)
            metadata.append({"site": site, "site_label": label})

        # Run screening
        workflow = ScreeningWorkflow(self.mlip, self.config)
        results = workflow.screen(structures, metadata)

        return results

    def screen_coverages(
        self,
        slab: Atoms,
        adsorbate: Atoms,
        configurations: list[list[tuple[float, float, float]]],
        height: float = 2.0,
        labels: list[str] | None = None,
    ) -> list[ScreeningResult]:
        """Screen multiple coverage configurations.

        Parameters
        ----------
        slab : Atoms
            Surface slab structure.
        adsorbate : Atoms
            Adsorbate molecule/atom.
        configurations : list[list[tuple]]
            Each element is a list of sites for that configuration.
        height : float
            Adsorbate height above sites (Å).
        labels : list[str] | None
            Labels for each configuration.

        Returns
        -------
        list[ScreeningResult]
            Screening results for each configuration.
        """
        if labels is None:
            labels = [f"config_{i}" for i in range(len(configurations))]

        structures = []
        metadata = []

        for sites, label in zip(configurations, labels):
            structure = slab.copy()

            for site in sites:
                ads = adsorbate.copy()
                ads.translate(np.array(site) + np.array([0, 0, height]))
                structure.extend(ads)

            structures.append(structure)
            metadata.append({
                "n_adsorbates": len(sites),
                "sites": sites,
                "label": label,
            })

        workflow = ScreeningWorkflow(self.mlip, self.config)
        return workflow.screen(structures, metadata)
