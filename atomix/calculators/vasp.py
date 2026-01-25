"""VASP calculator interface for atomix."""

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write as ase_write


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

    def write_inputs(
        self,
        atoms: Atoms,
        calc_type: str = "static",
        kpoints: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Write VASP input files.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object.
        calc_type : str
            Calculation type: 'static', 'relax', 'aimd_nvt', etc.
        kpoints : dict | None
            K-points specification. If None, auto-generate.

        Returns
        -------
        dict[str, Path]
            Paths to generated files.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        # Write POSCAR
        poscar_path = self.directory / "POSCAR"
        ase_write(str(poscar_path), atoms, format="vasp", vasp5=True)

        # Write INCAR
        incar_path = self.directory / "INCAR"
        incar_dict = self.get_incar_dict(calc_type)
        self._write_incar(incar_path, incar_dict)

        # Write KPOINTS
        kpoints_path = self.directory / "KPOINTS"
        if kpoints is None:
            kpoints = self.estimate_kpoints(atoms)
        self._write_kpoints(kpoints_path, kpoints)

        return {
            "POSCAR": poscar_path,
            "INCAR": incar_path,
            "KPOINTS": kpoints_path,
        }

    def _write_incar(self, path: Path, incar: dict[str, Any]) -> None:
        """Write INCAR file."""
        lines = []
        for key, value in sorted(incar.items()):
            # Format value appropriately
            if isinstance(value, bool):
                val_str = ".TRUE." if value else ".FALSE."
            elif isinstance(value, float):
                # Scientific notation for small numbers
                if abs(value) < 0.01 and value != 0:
                    val_str = f"{value:.1E}"
                else:
                    val_str = str(value)
            else:
                val_str = str(value)
            lines.append(f"{key} = {val_str}")

        path.write_text("\n".join(lines) + "\n")

    def _write_kpoints(self, path: Path, kpoints: dict[str, Any]) -> None:
        """Write KPOINTS file."""
        ktype = kpoints.get("type", "automatic").lower()
        grid = kpoints.get("grid", [1, 1, 1])
        shift = kpoints.get("shift", [0, 0, 0])

        lines = ["Automatic mesh"]
        lines.append("0")  # Auto-generate

        if ktype == "gamma":
            lines.append("Gamma")
        else:
            lines.append("Monkhorst-Pack")

        lines.append(f"{grid[0]} {grid[1]} {grid[2]}")
        lines.append(f"{shift[0]} {shift[1]} {shift[2]}")

        path.write_text("\n".join(lines) + "\n")

    def estimate_kpoints(
        self,
        atoms: Atoms,
        density: float = 40.0,
    ) -> dict[str, Any]:
        """Estimate k-point mesh from structure.

        Parameters
        ----------
        atoms : Atoms
            Structure to generate k-points for.
        density : float
            K-point density in points per Å⁻¹.

        Returns
        -------
        dict
            K-points specification with grid and type.
        """
        cell = atoms.get_cell()
        # Get reciprocal lattice vector lengths
        reciprocal = cell.reciprocal() * 2 * np.pi
        rec_lengths = np.linalg.norm(reciprocal, axis=1)

        # Calculate grid size from density
        # density is points per Å⁻¹, rec_lengths are in Å⁻¹
        grid = []
        for length in rec_lengths:
            n = max(1, int(np.ceil(density * length / (2 * np.pi))))
            grid.append(n)

        # Use Gamma-centered for odd grids or slabs (one direction has k=1)
        use_gamma = any(g % 2 == 1 for g in grid) or min(grid) == 1

        return {
            "type": "gamma" if use_gamma else "monkhorst-pack",
            "grid": grid,
            "shift": [0, 0, 0],
        }

    def read_outputs(self) -> dict[str, Any]:
        """Parse VASP output files."""
        raise NotImplementedError

    def get_incar_dict(self, calc_type: str = "static") -> dict[str, Any]:
        """Generate INCAR dictionary with defaults and user overrides."""
        incar = self.DEFAULTS.get(calc_type, {}).copy()
        incar.update(self.parameters)
        return incar
