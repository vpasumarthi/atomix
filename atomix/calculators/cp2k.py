"""CP2K calculator interface for atomix."""

from pathlib import Path
from typing import Any

from ase import Atoms


class CP2KCalculator:
    """CP2K calculator wrapper.

    Parameters
    ----------
    directory : Path | str
        Working directory for CP2K calculations.
    **kwargs
        CP2K parameters.
    """

    def __init__(self, directory: Path | str = ".", **kwargs: Any) -> None:
        self.directory = Path(directory)
        self.parameters = kwargs

    def write_inputs(self, atoms: Atoms) -> None:
        """Write CP2K input files."""
        raise NotImplementedError

    def read_outputs(self) -> dict[str, Any]:
        """Parse CP2K output files."""
        raise NotImplementedError
