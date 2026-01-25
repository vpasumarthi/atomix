"""Active learning helpers for MLIP training data generation.

Provides utilities for selecting training data, estimating model
uncertainty, and exporting data in formats suitable for MLIP training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from atomix.calculators.mlip import MLIPCalculator


@dataclass
class TrainingPoint:
    """A single training data point with DFT labels.

    Attributes
    ----------
    atoms : Atoms
        Structure (should have energy/forces attached).
    energy : float
        DFT total energy (eV).
    forces : np.ndarray
        DFT forces (eV/Å), shape (n_atoms, 3).
    stress : np.ndarray | None
        DFT stress tensor (eV/Å³), shape (3, 3).
    source : str
        Origin of this data point (e.g., 'md', 'relaxation', 'screening').
    metadata : dict
        Additional metadata.
    """

    atoms: Atoms
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None = None
    source: str = ""
    metadata: dict = field(default_factory=dict)


class TrainingDataExporter:
    """Export training data to formats for MLIP training.

    Supports extended XYZ format (for MACE, NequIP) and other
    common formats.

    Parameters
    ----------
    data : list[TrainingPoint]
        Training data to export.

    Examples
    --------
    >>> exporter = TrainingDataExporter(training_data)
    >>> exporter.to_extxyz("train.xyz", energy_key="REF_energy")
    >>> exporter.to_ase_db("train.db")
    """

    def __init__(self, data: list[TrainingPoint] | None = None) -> None:
        self._data: list[TrainingPoint] = data or []

    def add_point(self, point: TrainingPoint) -> None:
        """Add a training point."""
        self._data.append(point)

    def add_from_atoms(
        self,
        atoms: Atoms,
        source: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Add training point from Atoms with calculator results.

        The Atoms object should have a calculator attached with
        computed energy and forces.

        Parameters
        ----------
        atoms : Atoms
            Structure with calculator results.
        source : str
            Origin label.
        metadata : dict | None
            Additional metadata.
        """
        if atoms.calc is None:
            raise ValueError("Atoms must have calculator with results")

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        try:
            stress = atoms.get_stress(voigt=False)
        except Exception:
            stress = None

        point = TrainingPoint(
            atoms=atoms.copy(),
            energy=energy,
            forces=forces.copy(),
            stress=stress,
            source=source,
            metadata=metadata or {},
        )
        self._data.append(point)

    def to_extxyz(
        self,
        filename: Path | str,
        energy_key: str = "REF_energy",
        forces_key: str = "REF_forces",
        stress_key: str = "REF_stress",
        append: bool = False,
    ) -> Path:
        """Export to extended XYZ format.

        This format is compatible with MACE training.

        Parameters
        ----------
        filename : Path | str
            Output file path.
        energy_key : str
            Key for energy in atoms.info.
        forces_key : str
            Key for forces in atoms.arrays.
        stress_key : str
            Key for stress in atoms.info.
        append : bool
            Append to existing file.

        Returns
        -------
        Path
            Path to written file.
        """
        path = Path(filename)
        atoms_list = []

        for point in self._data:
            atoms = point.atoms.copy()

            # Store energy in info
            atoms.info[energy_key] = point.energy

            # Store forces in arrays
            atoms.arrays[forces_key] = point.forces

            # Store stress if available
            if point.stress is not None:
                # Convert to Voigt notation for storage
                atoms.info[stress_key] = point.stress

            # Add metadata
            atoms.info["source"] = point.source
            for key, value in point.metadata.items():
                atoms.info[key] = value

            atoms_list.append(atoms)

        ase_write(str(path), atoms_list, format="extxyz", append=append)
        return path

    def to_ase_db(self, filename: Path | str) -> Path:
        """Export to ASE database format.

        Parameters
        ----------
        filename : Path | str
            Output database path (.db).

        Returns
        -------
        Path
            Path to database file.
        """
        from ase.db import connect

        path = Path(filename)
        db = connect(str(path))

        for point in self._data:
            atoms = point.atoms.copy()

            # Prepare data dict
            data = {
                "energy": point.energy,
                "forces": point.forces,
                "source": point.source,
            }
            if point.stress is not None:
                data["stress"] = point.stress

            data.update(point.metadata)

            db.write(atoms, data=data)

        return path

    def split_train_val(
        self,
        val_fraction: float = 0.1,
        random_seed: int = 42,
    ) -> tuple["TrainingDataExporter", "TrainingDataExporter"]:
        """Split data into training and validation sets.

        Parameters
        ----------
        val_fraction : float
            Fraction of data for validation (0.0-1.0).
        random_seed : int
            Random seed for reproducibility.

        Returns
        -------
        tuple[TrainingDataExporter, TrainingDataExporter]
            (training_exporter, validation_exporter)
        """
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(len(self._data))

        n_val = max(1, int(len(self._data) * val_fraction))
        val_indices = set(indices[:n_val])

        train_data = [d for i, d in enumerate(self._data) if i not in val_indices]
        val_data = [d for i, d in enumerate(self._data) if i in val_indices]

        return TrainingDataExporter(train_data), TrainingDataExporter(val_data)

    def __len__(self) -> int:
        return len(self._data)


class UncertaintyEstimator:
    """Estimate model uncertainty using ensemble disagreement.

    Uses multiple MLIP models (e.g., from different training seeds)
    to estimate prediction uncertainty through ensemble disagreement.

    Parameters
    ----------
    calculators : list[MLIPCalculator]
        List of MLIP calculators forming the ensemble.

    Examples
    --------
    >>> models = [
    ...     MACECalculator(model_path="model_seed1.model"),
    ...     MACECalculator(model_path="model_seed2.model"),
    ...     MACECalculator(model_path="model_seed3.model"),
    ... ]
    >>> estimator = UncertaintyEstimator(models)
    >>> uncertainty = estimator.estimate(atoms)
    """

    def __init__(self, calculators: list[MLIPCalculator]) -> None:
        if len(calculators) < 2:
            raise ValueError("Ensemble requires at least 2 calculators")
        self.calculators = calculators

    def estimate(self, atoms: Atoms) -> dict[str, float]:
        """Estimate uncertainty for a structure.

        Parameters
        ----------
        atoms : Atoms
            Structure to evaluate.

        Returns
        -------
        dict[str, float]
            Uncertainty metrics:
            - energy_std: Standard deviation of energies (eV)
            - energy_mean: Mean energy (eV)
            - force_std_mean: Mean of per-atom force std (eV/Å)
            - force_std_max: Max per-atom force std (eV/Å)
        """
        energies = []
        all_forces = []

        for calc in self.calculators:
            result = calc.calculate(atoms)
            if result["energy"] is not None:
                energies.append(result["energy"])
            if result["forces"] is not None:
                all_forces.append(result["forces"])

        results = {
            "energy_mean": float("nan"),
            "energy_std": float("nan"),
            "force_std_mean": float("nan"),
            "force_std_max": float("nan"),
        }

        if energies:
            results["energy_mean"] = float(np.mean(energies))
            results["energy_std"] = float(np.std(energies))

        if len(all_forces) >= 2:
            # Stack forces: (n_models, n_atoms, 3)
            forces_array = np.stack(all_forces)
            # Per-atom force magnitude std
            force_mags = np.linalg.norm(forces_array, axis=2)  # (n_models, n_atoms)
            per_atom_std = np.std(force_mags, axis=0)  # (n_atoms,)

            results["force_std_mean"] = float(np.mean(per_atom_std))
            results["force_std_max"] = float(np.max(per_atom_std))

        return results

    def estimate_batch(
        self,
        structures: list[Atoms],
    ) -> list[dict[str, float]]:
        """Estimate uncertainty for multiple structures.

        Parameters
        ----------
        structures : list[Atoms]
            Structures to evaluate.

        Returns
        -------
        list[dict[str, float]]
            Uncertainty metrics for each structure.
        """
        return [self.estimate(atoms) for atoms in structures]


class ActiveLearningSelector:
    """Select structures for active learning based on uncertainty.

    Provides various strategies for selecting informative training
    data points from a pool of candidates.

    Parameters
    ----------
    uncertainty_estimator : UncertaintyEstimator | None
        Estimator for model uncertainty (required for uncertainty-based selection).

    Examples
    --------
    >>> selector = ActiveLearningSelector(estimator)
    >>> selected = selector.select_by_uncertainty(candidates, n=100)
    >>> selected = selector.select_diverse(candidates, n=50)
    """

    def __init__(
        self,
        uncertainty_estimator: UncertaintyEstimator | None = None,
    ) -> None:
        self.estimator = uncertainty_estimator

    def select_by_uncertainty(
        self,
        candidates: list[Atoms],
        n: int,
        metric: Literal["energy", "force", "combined"] = "combined",
    ) -> list[tuple[Atoms, float]]:
        """Select structures with highest uncertainty.

        Parameters
        ----------
        candidates : list[Atoms]
            Pool of candidate structures.
        n : int
            Number of structures to select.
        metric : str
            Uncertainty metric to use:
            - 'energy': energy standard deviation
            - 'force': max force standard deviation
            - 'combined': sum of normalized energy and force std

        Returns
        -------
        list[tuple[Atoms, float]]
            Selected (atoms, uncertainty_score) pairs, sorted by uncertainty.
        """
        if self.estimator is None:
            raise ValueError("UncertaintyEstimator required for uncertainty selection")

        uncertainties = self.estimator.estimate_batch(candidates)

        scored = []
        for atoms, unc in zip(candidates, uncertainties):
            if metric == "energy":
                score = unc["energy_std"]
            elif metric == "force":
                score = unc["force_std_max"]
            else:  # combined
                # Normalize roughly: energy in eV, force in eV/Å
                score = unc["energy_std"] + 0.5 * unc["force_std_max"]

            if not np.isnan(score):
                scored.append((atoms, score))

        # Sort by uncertainty (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:n]

    def select_diverse(
        self,
        candidates: list[Atoms],
        n: int,
        descriptor: Literal["soap", "composition", "energy"] = "composition",
    ) -> list[Atoms]:
        """Select diverse structures using farthest point sampling.

        Parameters
        ----------
        candidates : list[Atoms]
            Pool of candidate structures.
        n : int
            Number of structures to select.
        descriptor : str
            Diversity metric:
            - 'composition': based on element fractions
            - 'energy': based on MLIP energy (if estimator available)
            - 'soap': based on SOAP descriptors (requires dscribe)

        Returns
        -------
        list[Atoms]
            Diverse subset of structures.
        """
        if len(candidates) <= n:
            return candidates.copy()

        if descriptor == "composition":
            features = self._composition_features(candidates)
        elif descriptor == "energy" and self.estimator is not None:
            features = self._energy_features(candidates)
        elif descriptor == "soap":
            features = self._soap_features(candidates)
        else:
            features = self._composition_features(candidates)

        # Farthest point sampling
        selected_idx = self._farthest_point_sampling(features, n)

        return [candidates[i] for i in selected_idx]

    def _composition_features(self, structures: list[Atoms]) -> np.ndarray:
        """Generate composition-based feature vectors."""
        # Get all elements
        all_elements = set()
        for atoms in structures:
            all_elements.update(atoms.get_chemical_symbols())
        elements = sorted(all_elements)

        features = []
        for atoms in structures:
            symbols = atoms.get_chemical_symbols()
            counts = {e: 0 for e in elements}
            for s in symbols:
                counts[s] += 1
            # Normalize by total atoms
            n_atoms = len(atoms)
            feature = [counts[e] / n_atoms for e in elements]
            features.append(feature)

        return np.array(features)

    def _energy_features(self, structures: list[Atoms]) -> np.ndarray:
        """Generate energy-based features (per-atom energy)."""
        if self.estimator is None:
            raise ValueError("Estimator required for energy features")

        features = []
        for atoms in structures:
            unc = self.estimator.estimate(atoms)
            # Use energy per atom as feature
            e_per_atom = unc["energy_mean"] / len(atoms)
            features.append([e_per_atom])

        return np.array(features)

    def _soap_features(self, structures: list[Atoms]) -> np.ndarray:
        """Generate SOAP descriptor features."""
        try:
            from dscribe.descriptors import SOAP
        except ImportError:
            raise ImportError("dscribe is required for SOAP features")

        # Get all species
        all_species = set()
        for atoms in structures:
            all_species.update(atoms.get_chemical_symbols())

        soap = SOAP(
            species=sorted(all_species),
            r_cut=6.0,
            n_max=8,
            l_max=6,
            average="inner",
            periodic=True,
        )

        features = soap.create(structures)
        return features

    def _farthest_point_sampling(
        self,
        features: np.ndarray,
        n: int,
    ) -> list[int]:
        """Select diverse points using farthest point sampling."""
        n_samples = len(features)
        selected = [0]  # Start with first point

        # Precompute distances from first point
        distances = np.linalg.norm(features - features[0], axis=1)

        while len(selected) < n and len(selected) < n_samples:
            # Find farthest point from selected set
            farthest = np.argmax(distances)
            selected.append(int(farthest))

            # Update distances (min distance to any selected point)
            new_distances = np.linalg.norm(features - features[farthest], axis=1)
            distances = np.minimum(distances, new_distances)

            # Set selected points to 0 distance
            distances[farthest] = 0

        return selected

    def select_from_trajectory(
        self,
        trajectory: list[Atoms],
        n: int,
        interval: int | None = None,
        by_uncertainty: bool = True,
    ) -> list[Atoms]:
        """Select frames from MD or relaxation trajectory.

        Parameters
        ----------
        trajectory : list[Atoms]
            Trajectory frames.
        n : int
            Number of frames to select.
        interval : int | None
            If set, select every nth frame first, then fill with uncertainty.
        by_uncertainty : bool
            Use uncertainty for selection (requires estimator).

        Returns
        -------
        list[Atoms]
            Selected frames.
        """
        if len(trajectory) <= n:
            return trajectory.copy()

        if interval is not None:
            # Start with regular interval selection
            selected = trajectory[::interval]
            if len(selected) >= n:
                return selected[:n]
            # Fill remaining with uncertainty-based selection
            remaining = [t for i, t in enumerate(trajectory) if i % interval != 0]
            n_remaining = n - len(selected)
        else:
            remaining = trajectory
            n_remaining = n
            selected = []

        if by_uncertainty and self.estimator is not None:
            uncertain = self.select_by_uncertainty(remaining, n_remaining)
            selected.extend([atoms for atoms, _ in uncertain])
        else:
            # Uniform sampling
            indices = np.linspace(0, len(remaining) - 1, n_remaining, dtype=int)
            selected.extend([remaining[i] for i in indices])

        return selected[:n]
