"""VASP calculator interface for atomix."""

import re
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
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
        """Parse VASP output files.

        Returns
        -------
        dict[str, Any]
            Parsed results including:
            - energy: Final total energy (eV)
            - forces: Forces on atoms (eV/Å)
            - stress: Stress tensor (kB)
            - atoms: Final structure
            - converged: Whether calculation converged
            - n_steps: Number of ionic steps
            - trajectory: List of atoms for relaxations
        """
        results: dict[str, Any] = {
            "converged": False,
            "energy": None,
            "forces": None,
            "stress": None,
            "atoms": None,
            "n_steps": 0,
            "trajectory": [],
            "errors": [],
            "warnings": [],
        }

        outcar = self.directory / "OUTCAR"
        contcar = self.directory / "CONTCAR"
        oszicar = self.directory / "OSZICAR"
        vasprun = self.directory / "vasprun.xml"

        # Try to read from vasprun.xml first (most reliable)
        if vasprun.exists():
            try:
                results.update(self._parse_vasprun(vasprun))
            except Exception as e:
                results["warnings"].append(f"Could not parse vasprun.xml: {e}")

        # Parse OUTCAR for additional info or if vasprun failed
        if outcar.exists():
            try:
                outcar_results = self._parse_outcar(outcar)
                # Only update if vasprun didn't provide values
                for key, value in outcar_results.items():
                    if results.get(key) is None:
                        results[key] = value
                    elif key in ("converged", "errors", "warnings"):
                        if key == "converged":
                            results[key] = results[key] or value
                        else:
                            results[key].extend(value)
            except Exception as e:
                results["warnings"].append(f"Could not parse OUTCAR: {e}")

        # Read final structure from CONTCAR
        if contcar.exists() and contcar.stat().st_size > 0:
            try:
                results["atoms"] = ase_read(str(contcar), format="vasp")
            except Exception as e:
                results["warnings"].append(f"Could not read CONTCAR: {e}")

        # Count ionic steps from OSZICAR
        if oszicar.exists():
            try:
                lines = oszicar.read_text().strip().split("\n")
                # Count lines starting with a number (ionic steps)
                n_steps = sum(1 for line in lines if line.strip() and line.strip()[0].isdigit())
                results["n_steps"] = max(results["n_steps"], n_steps)
            except Exception:
                pass

        return results

    def _parse_vasprun(self, path: Path) -> dict[str, Any]:
        """Parse vasprun.xml using ASE."""
        results: dict[str, Any] = {}

        # Read all images (for relaxations)
        try:
            trajectory = ase_read(str(path), index=":", format="vasp-xml")
            if isinstance(trajectory, Atoms):
                trajectory = [trajectory]
            results["trajectory"] = trajectory

            if trajectory:
                final = trajectory[-1]
                results["atoms"] = final
                results["n_steps"] = len(trajectory)

                # Get energy and forces from final structure
                if final.calc is not None:
                    try:
                        results["energy"] = final.get_potential_energy()
                    except Exception:
                        pass
                    try:
                        results["forces"] = final.get_forces()
                    except Exception:
                        pass
                    try:
                        results["stress"] = final.get_stress()
                    except Exception:
                        pass
        except Exception as e:
            raise ValueError(f"Failed to parse vasprun.xml: {e}")

        return results

    def _parse_outcar(self, path: Path) -> dict[str, Any]:
        """Parse OUTCAR file for results and status."""
        results: dict[str, Any] = {
            "converged": False,
            "errors": [],
            "warnings": [],
        }

        content = path.read_text()

        # Check for convergence
        if "reached required accuracy" in content:
            results["converged"] = True
        elif "General timing and accounting" in content:
            # Calculation finished but may not have converged
            results["converged"] = True

        # Check for errors
        if "VERY BAD NEWS" in content:
            results["errors"].append("VASP encountered serious error (VERY BAD NEWS)")
            results["converged"] = False
        if "Error EDDDAV" in content:
            results["errors"].append("Electronic convergence failed (EDDDAV)")
        if "ZBRENT: fatal error" in content:
            results["errors"].append("Ionic convergence failed (ZBRENT)")

        # Extract final energy
        energy_matches = re.findall(r"free  energy   TOTEN\s*=\s*([-\d.]+)", content)
        if energy_matches:
            results["energy"] = float(energy_matches[-1])

        # Extract forces (last occurrence)
        force_block = re.search(
            r"TOTAL-FORCE \(eV/Angst\)\s*-+\s*([\s\S]*?)\s*-+",
            content,
        )
        if force_block:
            force_lines = force_block.group(1).strip().split("\n")
            forces = []
            for line in force_lines:
                parts = line.split()
                if len(parts) >= 6:
                    forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
            if forces:
                results["forces"] = np.array(forces)

        # Extract stress tensor (last occurrence)
        stress_match = re.search(
            r"in kB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
            content,
        )
        if stress_match:
            stress = [float(stress_match.group(i)) for i in range(1, 7)]
            results["stress"] = np.array(stress)

        # Count ionic steps
        n_steps = len(re.findall(r"Iteration\s+\d+\s*\(", content))
        if n_steps > 0:
            results["n_steps"] = n_steps

        return results

    def read_trajectory(self) -> list[Atoms]:
        """Read relaxation/MD trajectory.

        Returns
        -------
        list[Atoms]
            List of structures from each ionic step.
        """
        vasprun = self.directory / "vasprun.xml"
        xdatcar = self.directory / "XDATCAR"

        if vasprun.exists():
            try:
                traj = ase_read(str(vasprun), index=":", format="vasp-xml")
                return traj if isinstance(traj, list) else [traj]
            except Exception:
                pass

        if xdatcar.exists():
            try:
                traj = ase_read(str(xdatcar), index=":", format="vasp-xdatcar")
                return traj if isinstance(traj, list) else [traj]
            except Exception:
                pass

        return []

    def is_converged(self) -> bool:
        """Check if calculation converged."""
        outcar = self.directory / "OUTCAR"
        if not outcar.exists():
            return False

        content = outcar.read_text()
        return "reached required accuracy" in content or "General timing" in content

    def get_energy(self) -> float | None:
        """Get final energy from calculation."""
        results = self.read_outputs()
        return results.get("energy")

    def get_incar_dict(self, calc_type: str = "static") -> dict[str, Any]:
        """Generate INCAR dictionary with defaults and user overrides."""
        incar = self.DEFAULTS.get(calc_type, {}).copy()
        incar.update(self.parameters)
        return incar
