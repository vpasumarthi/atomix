"""VASP calculator interface for atomix.

Uses pymatgen for robust VASP I/O handling.
"""

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Incar, Kpoints, Outcar, Poscar, Vasprun


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
        """Write VASP input files using pymatgen.

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

        # Convert ASE Atoms to pymatgen Structure
        structure = AseAtomsAdaptor.get_structure(atoms)

        # Write POSCAR using pymatgen
        poscar_path = self.directory / "POSCAR"
        poscar = Poscar(structure)
        poscar.write_file(str(poscar_path))

        # Write INCAR using pymatgen
        incar_path = self.directory / "INCAR"
        incar_dict = self.get_incar_dict(calc_type)
        incar = Incar(incar_dict)
        incar.write_file(str(incar_path))

        # Write KPOINTS using pymatgen
        kpoints_path = self.directory / "KPOINTS"
        if kpoints is None:
            kpoints = self.estimate_kpoints(atoms)
        kpoints_obj = self._create_kpoints(kpoints)
        kpoints_obj.write_file(str(kpoints_path))

        return {
            "POSCAR": poscar_path,
            "INCAR": incar_path,
            "KPOINTS": kpoints_path,
        }

    def _create_kpoints(self, kpoints: dict[str, Any]) -> Kpoints:
        """Create pymatgen Kpoints object from specification.

        Parameters
        ----------
        kpoints : dict
            K-points specification with 'type', 'grid', and optional 'shift'.

        Returns
        -------
        Kpoints
            Pymatgen Kpoints object.
        """
        ktype = kpoints.get("type", "automatic").lower()
        grid = kpoints.get("grid", [1, 1, 1])
        shift = kpoints.get("shift", [0, 0, 0])

        if ktype == "gamma":
            return Kpoints.gamma_automatic(kpts=tuple(grid), shift=tuple(shift))
        else:
            return Kpoints.monkhorst_automatic(kpts=tuple(grid), shift=tuple(shift))

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
        """Parse VASP output files using pymatgen.

        Returns
        -------
        dict[str, Any]
            Parsed results including:
            - energy: Final total energy (eV)
            - forces: Forces on atoms (eV/Å)
            - stress: Stress tensor (kB)
            - atoms: Final structure (ASE Atoms)
            - converged: Whether calculation converged
            - n_steps: Number of ionic steps
            - trajectory: List of ASE Atoms for relaxations
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

        outcar_path = self.directory / "OUTCAR"
        vasprun_path = self.directory / "vasprun.xml"

        # Try to read from vasprun.xml first (most complete)
        if vasprun_path.exists():
            try:
                vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
                results["converged"] = vr.converged
                results["energy"] = vr.final_energy
                results["n_steps"] = len(vr.ionic_steps)

                # Get forces and stress from final ionic step
                if vr.ionic_steps:
                    final_step = vr.ionic_steps[-1]
                    if "forces" in final_step:
                        results["forces"] = np.array(final_step["forces"])
                    if "stress" in final_step:
                        results["stress"] = np.array(final_step["stress"])

                # Get final structure as ASE Atoms
                final_structure = vr.final_structure
                results["atoms"] = AseAtomsAdaptor.get_atoms(final_structure)

                # Build trajectory as list of ASE Atoms
                trajectory = []
                for step in vr.ionic_steps:
                    if "structure" in step:
                        atoms = AseAtomsAdaptor.get_atoms(step["structure"])
                        trajectory.append(atoms)
                results["trajectory"] = trajectory

            except Exception as e:
                results["warnings"].append(f"Could not parse vasprun.xml: {e}")

        # Parse OUTCAR for additional info or if vasprun failed
        if outcar_path.exists():
            try:
                outcar = Outcar(str(outcar_path))

                # Fill in missing values from OUTCAR
                if results["energy"] is None and outcar.final_energy is not None:
                    results["energy"] = outcar.final_energy

                if results["forces"] is None and hasattr(outcar, "read_table_pattern"):
                    # pymatgen Outcar stores forces internally
                    pass  # forces already parsed from vasprun if available

                # Check for convergence indicators
                if not results["converged"]:
                    results["converged"] = outcar.converged

                # Check for run stats (indicates completion)
                if outcar.run_stats:
                    results["converged"] = True

            except Exception as e:
                results["warnings"].append(f"Could not parse OUTCAR: {e}")

        return results

    def read_trajectory(self) -> list[Atoms]:
        """Read relaxation/MD trajectory using pymatgen.

        Returns
        -------
        list[Atoms]
            List of ASE Atoms from each ionic step.
        """
        vasprun_path = self.directory / "vasprun.xml"
        xdatcar_path = self.directory / "XDATCAR"

        if vasprun_path.exists():
            try:
                vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
                trajectory = []
                for step in vr.ionic_steps:
                    if "structure" in step:
                        atoms = AseAtomsAdaptor.get_atoms(step["structure"])
                        trajectory.append(atoms)
                return trajectory
            except Exception:
                pass

        if xdatcar_path.exists():
            try:
                # Fall back to ASE for XDATCAR (pymatgen Xdatcar exists but ASE is simpler)
                traj = ase_read(str(xdatcar_path), index=":", format="vasp-xdatcar")
                return traj if isinstance(traj, list) else [traj]
            except Exception:
                pass

        return []

    def is_converged(self) -> bool:
        """Check if calculation converged using pymatgen."""
        vasprun_path = self.directory / "vasprun.xml"
        outcar_path = self.directory / "OUTCAR"

        # Prefer vasprun.xml for convergence check
        if vasprun_path.exists():
            try:
                vr = Vasprun(str(vasprun_path), parse_dos=False, parse_eigen=False)
                return vr.converged
            except Exception:
                pass

        # Fall back to OUTCAR
        if outcar_path.exists():
            try:
                outcar = Outcar(str(outcar_path))
                return outcar.converged or bool(outcar.run_stats)
            except Exception:
                pass

        return False

    def get_energy(self) -> float | None:
        """Get final energy from calculation."""
        results = self.read_outputs()
        return results.get("energy")

    def get_incar_dict(self, calc_type: str = "static") -> dict[str, Any]:
        """Generate INCAR dictionary with defaults and user overrides."""
        incar = self.DEFAULTS.get(calc_type, {}).copy()
        incar.update(self.parameters)
        return incar

    def setup_restart(self, calc_type: str = "relax") -> dict[str, Path]:
        """Set up calculation for restart from incomplete run.

        Uses CONTCAR as new POSCAR and sets ISTART/ICHARG for wavefunction
        and charge density restart if available.

        Parameters
        ----------
        calc_type : str
            Calculation type for INCAR defaults.

        Returns
        -------
        dict[str, Path]
            Paths to modified/created files.

        Raises
        ------
        FileNotFoundError
            If no CONTCAR exists for restart.
        """
        contcar = self.directory / "CONTCAR"
        wavecar = self.directory / "WAVECAR"
        chgcar = self.directory / "CHGCAR"
        poscar = self.directory / "POSCAR"
        incar_path = self.directory / "INCAR"

        if not contcar.exists() or contcar.stat().st_size == 0:
            raise FileNotFoundError(f"No valid CONTCAR found in {self.directory}")

        files: dict[str, Path] = {}

        # Copy CONTCAR to POSCAR
        import shutil
        shutil.copy(contcar, poscar)
        files["POSCAR"] = poscar

        # Update INCAR for restart
        incar_dict = self.get_incar_dict(calc_type)

        # Set restart flags based on available files
        if wavecar.exists() and wavecar.stat().st_size > 0:
            incar_dict["ISTART"] = 1  # Read existing wavefunction
        if chgcar.exists() and chgcar.stat().st_size > 0:
            incar_dict["ICHARG"] = 1  # Read existing charge density

        incar = Incar(incar_dict)
        incar.write_file(str(incar_path))
        files["INCAR"] = incar_path

        return files

    def validate_outputs(self) -> dict[str, Any]:
        """Validate calculation outputs and check for common errors.

        Returns
        -------
        dict[str, Any]
            Validation results with keys:
            - valid: bool, whether calculation completed successfully
            - status: str, one of 'completed', 'incomplete', 'failed', 'not_started'
            - errors: list of error messages
            - warnings: list of warning messages
            - can_restart: bool, whether restart is possible
        """
        result = {
            "valid": False,
            "status": "not_started",
            "errors": [],
            "warnings": [],
            "can_restart": False,
        }

        outcar = self.directory / "OUTCAR"
        contcar = self.directory / "CONTCAR"
        oszicar = self.directory / "OSZICAR"

        if not outcar.exists():
            return result

        content = outcar.read_text()
        result["status"] = "incomplete"

        # Check for fatal errors
        if "VERY BAD NEWS" in content:
            result["status"] = "failed"
            result["errors"].append("VASP encountered fatal error (VERY BAD NEWS)")

        if "Error EDDDAV" in content:
            result["errors"].append("Electronic convergence failed (EDDDAV)")
            result["warnings"].append("Try increasing NELM or adjusting ALGO")

        if "ZBRENT: fatal error" in content:
            result["status"] = "failed"
            result["errors"].append("Ionic convergence failed (ZBRENT)")
            result["warnings"].append("Check initial geometry or use different optimizer")

        if "BRMIX: very serious problems" in content:
            result["errors"].append("Charge mixing failed (BRMIX)")
            result["warnings"].append("Try IMIX=1 or reduce AMIX")

        if "Sub-Space-Matrix is not hermitian" in content:
            result["errors"].append("Sub-space matrix not hermitian")
            result["warnings"].append("Try ALGO=Normal instead of ALGO=Fast")

        if "EDDRMM: call to ZHEGV failed" in content:
            result["errors"].append("EDDRMM eigenvalue problem failed")
            result["warnings"].append("Try reducing POTIM or use ALGO=VeryFast")

        # Check for memory issues
        if "malloc" in content.lower() or "out of memory" in content.lower():
            result["errors"].append("Memory allocation failed")
            result["warnings"].append("Reduce NCORE/NPAR or request more memory")

        # Check for successful completion
        if "reached required accuracy" in content:
            result["status"] = "completed"
            result["valid"] = True
        elif "General timing and accounting" in content:
            # Finished but may not have converged
            if not result["errors"]:
                result["status"] = "completed"
                result["valid"] = True
                result["warnings"].append("Calculation finished but check convergence")

        # Check restart possibility
        if contcar.exists() and contcar.stat().st_size > 0:
            result["can_restart"] = True

        # Add ionic step info
        if oszicar.exists():
            lines = oszicar.read_text().strip().split("\n")
            n_steps = sum(1 for line in lines if line.strip() and line.strip()[0].isdigit())
            if n_steps > 0:
                result["n_ionic_steps"] = n_steps

        return result

    def needs_restart(self) -> bool:
        """Check if calculation needs restart (incomplete but restartable)."""
        validation = self.validate_outputs()
        return (
            validation["status"] in ("incomplete", "failed")
            and validation["can_restart"]
        )
