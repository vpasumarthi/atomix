"""Machine learning interatomic potential (MLIP) interfaces for atomix."""

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator


class MLIPCalculator:
    """Base class for MLIP calculators.

    Parameters
    ----------
    model_path : Path | str | None
        Path to trained model file.
    device : str
        Device to run on ('cpu', 'cuda').
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self._calculator: Calculator | None = None

    def get_calculator(self) -> Calculator:
        """Return ASE-compatible calculator."""
        raise NotImplementedError

    def calculate(self, atoms: Atoms) -> dict[str, Any]:
        """Calculate energy and forces."""
        raise NotImplementedError


class MACECalculator(MLIPCalculator):
    """MACE machine learning potential calculator.

    Parameters
    ----------
    model_path : Path | str | None
        Path to MACE model. If None, uses pretrained foundation model.
    device : str
        Device to run on ('cpu', 'cuda').
    """

    def get_calculator(self) -> Calculator:
        """Return MACE ASE calculator."""
        raise NotImplementedError

    def calculate(self, atoms: Atoms) -> dict[str, Any]:
        """Calculate energy and forces with MACE."""
        raise NotImplementedError
