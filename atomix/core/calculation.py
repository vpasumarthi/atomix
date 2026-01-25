"""Base calculation classes for atomix."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ase import Atoms


class BaseCalculation(ABC):
    """Abstract base class for all calculation types.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing the system.
    directory : Path | str
        Working directory for the calculation.
    **kwargs
        Additional calculation parameters.
    """

    def __init__(
        self,
        atoms: Atoms,
        directory: Path | str = ".",
        **kwargs: Any,
    ) -> None:
        self.atoms = atoms
        self.directory = Path(directory)
        self.parameters = kwargs
        self._results: dict[str, Any] = {}

    @abstractmethod
    def setup(self) -> None:
        """Set up input files for the calculation."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Execute the calculation."""
        pass

    @abstractmethod
    def parse_results(self) -> dict[str, Any]:
        """Parse output files and return results."""
        pass

    @property
    def results(self) -> dict[str, Any]:
        """Return calculation results."""
        return self._results


class StaticCalculation(BaseCalculation):
    """Single-point energy calculation."""

    def setup(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def parse_results(self) -> dict[str, Any]:
        raise NotImplementedError


class RelaxCalculation(BaseCalculation):
    """Geometry optimization calculation."""

    def setup(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def parse_results(self) -> dict[str, Any]:
        raise NotImplementedError


class AIMDCalculation(BaseCalculation):
    """Ab initio molecular dynamics calculation."""

    def setup(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def parse_results(self) -> dict[str, Any]:
        raise NotImplementedError


class NVTCalculation(AIMDCalculation):
    """NVT ensemble molecular dynamics."""

    pass


class NPTCalculation(AIMDCalculation):
    """NPT ensemble molecular dynamics."""

    pass


class NVECalculation(AIMDCalculation):
    """NVE ensemble molecular dynamics."""

    pass


class NEBCalculation(BaseCalculation):
    """Nudged elastic band calculation for transition states."""

    def setup(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def parse_results(self) -> dict[str, Any]:
        raise NotImplementedError


class FrequencyCalculation(BaseCalculation):
    """Vibrational frequency calculation."""

    def setup(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def parse_results(self) -> dict[str, Any]:
        raise NotImplementedError
