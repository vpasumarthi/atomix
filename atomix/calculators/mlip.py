"""Machine learning interatomic potential (MLIP) interfaces for atomix.

Provides drop-in calculator replacements matching the VASPCalculator interface,
enabling seamless switching between DFT and MLIP for workflows.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

# Type alias for supported MACE foundation models
MACEModel = Literal[
    "small",
    "medium",
    "large",
    "mace-mp-0",
    "mace-off-small",
    "mace-off-medium",
    "mace-off-large",
]


class MLIPCalculator(ABC):
    """Base class for MLIP calculators.

    Provides a common interface for all MLIP implementations,
    matching the patterns used in VASPCalculator for seamless
    drop-in replacement.

    Parameters
    ----------
    model_path : Path | str | None
        Path to trained model file.
    device : str
        Device to run on ('cpu', 'cuda', 'mps').
    default_dtype : str
        Default dtype for calculations ('float32', 'float64').
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "cpu",
        default_dtype: str = "float64",
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.default_dtype = default_dtype
        self._calculator: Calculator | None = None
        self._model: Any = None

    @abstractmethod
    def get_calculator(self) -> Calculator:
        """Return ASE-compatible calculator.

        Returns
        -------
        Calculator
            ASE calculator instance ready for use with atoms.calc = calc.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate(self, atoms: Atoms) -> dict[str, Any]:
        """Calculate energy and forces for a structure.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object to calculate.

        Returns
        -------
        dict[str, Any]
            Results dictionary with keys:
            - converged: bool (always True for MLIP)
            - energy: float (eV)
            - forces: np.ndarray shape (n_atoms, 3)
            - stress: np.ndarray shape (6,) Voigt notation or None
            - atoms: Atoms (copy with results attached)
            - n_steps: int (always 1 for single-point)
            - trajectory: list[Atoms]
            - errors: list
            - warnings: list
        """
        raise NotImplementedError

    def calculate_batch(self, structures: list[Atoms]) -> list[dict[str, Any]]:
        """Calculate energies and forces for multiple structures.

        Default implementation iterates; subclasses may override
        for batched GPU inference.

        Parameters
        ----------
        structures : list[Atoms]
            List of structures to calculate.

        Returns
        -------
        list[dict[str, Any]]
            Results for each structure.
        """
        return [self.calculate(atoms) for atoms in structures]


class MACECalculator(MLIPCalculator):
    """MACE machine learning potential calculator.

    MACE (Multi-Atomic Cluster Expansion) provides accurate and fast
    interatomic potentials. Supports both custom trained models and
    pretrained foundation models.

    Parameters
    ----------
    model_path : Path | str | None
        Path to MACE model file (.model). If None, uses foundation model.
    model : str
        Foundation model to use if model_path is None.
        Options: 'small', 'medium', 'large', 'mace-mp-0',
                'mace-off-small', 'mace-off-medium', 'mace-off-large'.
    device : str
        Device to run on ('cpu', 'cuda', 'mps').
    default_dtype : str
        Default dtype ('float32', 'float64').
    dispersion : bool
        Add D3 dispersion correction (for mace-mp models).
    compile_mode : str | None
        Torch compile mode ('default', 'reduce-overhead', None).

    Examples
    --------
    >>> # Use pretrained foundation model
    >>> calc = MACECalculator(model="medium", device="cuda")
    >>> results = calc.calculate(atoms)

    >>> # Use custom trained model
    >>> calc = MACECalculator(model_path="my_model.model")
    >>> ase_calc = calc.get_calculator()
    >>> atoms.calc = ase_calc
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        model: MACEModel = "medium",
        device: str = "cpu",
        default_dtype: str = "float64",
        dispersion: bool = False,
        compile_mode: str | None = None,
    ) -> None:
        super().__init__(model_path, device, default_dtype)
        self.model_name = model
        self.dispersion = dispersion
        self.compile_mode = compile_mode
        self._mace_available: bool | None = None

    def _check_mace_available(self) -> bool:
        """Check if mace-torch is installed."""
        if self._mace_available is None:
            try:
                import mace  # noqa: F401

                self._mace_available = True
            except ImportError:
                self._mace_available = False
        return self._mace_available

    def _load_calculator(self) -> Calculator:
        """Load MACE calculator, handling imports gracefully."""
        if not self._check_mace_available():
            raise ImportError(
                "mace-torch is required for MACECalculator. "
                "Install with: pip install mace-torch"
            )

        from mace.calculators import MACECalculator as MACECalc
        from mace.calculators import mace_mp, mace_off

        if self.model_path is not None:
            # Load custom model
            calc = MACECalc(
                model_paths=str(self.model_path),
                device=self.device,
                default_dtype=self.default_dtype,
            )
        else:
            # Load foundation model
            model_lower = self.model_name.lower()
            if model_lower.startswith("mace-off"):
                # MACE-OFF for organic molecules
                size = model_lower.split("-")[-1] if "-" in model_lower else "medium"
                calc = mace_off(
                    model=size,
                    device=self.device,
                    default_dtype=self.default_dtype,
                )
            else:
                # MACE-MP for materials
                if model_lower in ("small", "medium", "large"):
                    size = model_lower
                else:
                    size = "medium"
                calc = mace_mp(
                    model=size,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    dispersion=self.dispersion,
                )

        return calc

    def get_calculator(self) -> Calculator:
        """Return MACE ASE calculator.

        Loads the model lazily on first call and caches it.

        Returns
        -------
        Calculator
            ASE-compatible MACE calculator.
        """
        if self._calculator is None:
            self._calculator = self._load_calculator()
        return self._calculator

    def calculate(self, atoms: Atoms) -> dict[str, Any]:
        """Calculate energy and forces with MACE.

        Parameters
        ----------
        atoms : Atoms
            Structure to calculate.

        Returns
        -------
        dict[str, Any]
            Results in format compatible with VASPCalculator.read_outputs().
        """
        results: dict[str, Any] = {
            "converged": True,  # MLIP always "converges"
            "energy": None,
            "forces": None,
            "stress": None,
            "atoms": None,
            "n_steps": 1,
            "trajectory": [],
            "errors": [],
            "warnings": [],
        }

        try:
            calc = self.get_calculator()

            # Create a copy to avoid modifying the original
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc

            # Calculate properties
            results["energy"] = atoms_copy.get_potential_energy()
            results["forces"] = atoms_copy.get_forces()

            # Stress is optional - only for periodic systems
            try:
                if atoms_copy.pbc.any():
                    stress = atoms_copy.get_stress(voigt=False)  # 3x3 tensor
                    results["stress"] = stress
            except Exception:
                # Some models don't support stress
                pass

            results["atoms"] = atoms_copy
            results["trajectory"] = [atoms_copy]

        except Exception as e:
            results["converged"] = False
            results["errors"].append(f"MACE calculation failed: {e}")

        return results

    def calculate_batch(self, structures: list[Atoms]) -> list[dict[str, Any]]:
        """Calculate energies and forces for multiple structures.

        Uses MACE's built-in batch processing for efficiency
        when running on GPU.

        Parameters
        ----------
        structures : list[Atoms]
            List of structures to calculate.

        Returns
        -------
        list[dict[str, Any]]
            Results for each structure.
        """
        # For now, use sequential calculation
        # Future: implement true batching with MACECalculator's batch mode
        return [self.calculate(atoms) for atoms in structures]


class NequIPCalculator(MLIPCalculator):
    """NequIP/Allegro machine learning potential calculator.

    NequIP is an equivariant neural network potential. Allegro is
    a faster, local version.

    Parameters
    ----------
    model_path : Path | str
        Path to deployed NequIP model (.pth file).
    device : str
        Device to run on ('cpu', 'cuda').
    species_to_type_name : dict | None
        Mapping from species to type names if needed.
    """

    def __init__(
        self,
        model_path: Path | str,
        device: str = "cpu",
        default_dtype: str = "float64",
        species_to_type_name: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model_path, device, default_dtype)
        self.species_to_type_name = species_to_type_name
        self._nequip_available: bool | None = None

    def _check_nequip_available(self) -> bool:
        """Check if nequip is installed."""
        if self._nequip_available is None:
            try:
                import nequip  # noqa: F401

                self._nequip_available = True
            except ImportError:
                self._nequip_available = False
        return self._nequip_available

    def get_calculator(self) -> Calculator:
        """Return NequIP ASE calculator."""
        if self._calculator is not None:
            return self._calculator

        if not self._check_nequip_available():
            raise ImportError(
                "nequip is required for NequIPCalculator. "
                "Install with: pip install nequip"
            )

        from nequip.ase import NequIPCalculator as NequIPCalc

        if self.model_path is None:
            raise ValueError("model_path is required for NequIPCalculator")

        self._calculator = NequIPCalc.from_deployed_model(
            model_path=str(self.model_path),
            device=self.device,
            species_to_type_name=self.species_to_type_name,
        )
        return self._calculator

    def calculate(self, atoms: Atoms) -> dict[str, Any]:
        """Calculate energy and forces with NequIP."""
        results: dict[str, Any] = {
            "converged": True,
            "energy": None,
            "forces": None,
            "stress": None,
            "atoms": None,
            "n_steps": 1,
            "trajectory": [],
            "errors": [],
            "warnings": [],
        }

        try:
            calc = self.get_calculator()
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc

            results["energy"] = atoms_copy.get_potential_energy()
            results["forces"] = atoms_copy.get_forces()

            try:
                if atoms_copy.pbc.any():
                    stress = atoms_copy.get_stress(voigt=False)
                    results["stress"] = stress
            except Exception:
                pass

            results["atoms"] = atoms_copy
            results["trajectory"] = [atoms_copy]

        except Exception as e:
            results["converged"] = False
            results["errors"].append(f"NequIP calculation failed: {e}")

        return results


def get_mlip_calculator(
    name: str,
    model_path: Path | str | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> MLIPCalculator:
    """Factory function to get MLIP calculator by name.

    Parameters
    ----------
    name : str
        Calculator name: 'mace', 'nequip', 'allegro'.
    model_path : Path | str | None
        Path to model file (required for nequip/allegro).
    device : str
        Device to run on.
    **kwargs
        Additional arguments passed to calculator.

    Returns
    -------
    MLIPCalculator
        Configured MLIP calculator instance.

    Examples
    --------
    >>> calc = get_mlip_calculator("mace", model="medium", device="cuda")
    >>> calc = get_mlip_calculator("nequip", model_path="model.pth")
    """
    name = name.lower()

    if name == "mace":
        return MACECalculator(model_path=model_path, device=device, **kwargs)
    elif name in ("nequip", "allegro"):
        if model_path is None:
            raise ValueError(f"{name} requires a model_path")
        return NequIPCalculator(model_path=model_path, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown MLIP calculator: {name}. Supported: mace, nequip")
