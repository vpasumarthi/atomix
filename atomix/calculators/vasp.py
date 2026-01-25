"""VASP calculator interface for atomix."""

from pathlib import Path
from typing import Any

from ase import Atoms


class VASPCalculator:
    """VASP calculator wrapper with atomix conventions.

    Parameters
    ----------
    directory : Path | str
        Working directory for VASP calculations.
    **kwargs
        VASP parameters (INCAR settings).
    """

    # Standard INCAR defaults for different calculation types
    DEFAULTS = {
        "static": {
            "NSW": 0,
            "IBRION": -1,
            "EDIFF": 1e-5,
        },
        "relax": {
            "IBRION": 2,
            "NSW": 100,
            "EDIFF": 1e-5,
            "EDIFFG": -0.02,
            "ISIF": 2,
        },
        "aimd_nvt": {
            "IBRION": 0,
            "NSW": 1000,
            "POTIM": 1.0,
            "SMASS": 0,
            "EDIFF": 1e-5,
        },
    }

    def __init__(self, directory: Path | str = ".", **kwargs: Any) -> None:
        self.directory = Path(directory)
        self.parameters = kwargs

    def write_inputs(self, atoms: Atoms, calc_type: str = "static") -> None:
        """Write VASP input files.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object.
        calc_type : str
            Calculation type: 'static', 'relax', 'aimd_nvt', etc.
        """
        raise NotImplementedError

    def read_outputs(self) -> dict[str, Any]:
        """Parse VASP output files."""
        raise NotImplementedError

    def get_incar_dict(self, calc_type: str = "static") -> dict[str, Any]:
        """Generate INCAR dictionary with defaults and user overrides."""
        incar = self.DEFAULTS.get(calc_type, {}).copy()
        incar.update(self.parameters)
        return incar
