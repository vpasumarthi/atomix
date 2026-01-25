"""Base calculation classes for atomix."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS, FIRE, LBFGS

from atomix.calculators.vasp import VASPCalculator


class BaseCalculation(ABC):
    """Abstract base class for all calculation types.

    Supports two execution modes:
    1. Direct mode: Using an ASE calculator (for MLIP, EMT, etc.)
    2. File mode: Writing input files and optionally submitting jobs (for VASP)

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing the system.
    directory : Path | str
        Working directory for the calculation.
    calculator : Calculator | None
        ASE calculator for direct execution. If None, uses file-based mode.
    **kwargs
        Additional calculation parameters.
    """

    def __init__(
        self,
        atoms: Atoms,
        directory: Path | str = ".",
        calculator: Calculator | None = None,
        **kwargs: Any,
    ) -> None:
        self.atoms = atoms
        self.directory = Path(directory)
        self.calculator = calculator
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

    def _run_with_ase_calculator(self) -> dict[str, Any]:
        """Run calculation using attached ASE calculator.

        Returns
        -------
        dict[str, Any]
            Results with energy, forces, etc.
        """
        if self.calculator is None:
            raise ValueError("No calculator attached for direct execution")

        result: dict[str, Any] = {
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
            atoms = self.atoms.copy()
            atoms.calc = self.calculator

            result["energy"] = atoms.get_potential_energy()
            result["forces"] = atoms.get_forces()

            try:
                if atoms.pbc.any():
                    result["stress"] = atoms.get_stress(voigt=False)
            except Exception:
                pass

            result["atoms"] = atoms
            result["trajectory"] = [atoms]

        except Exception as e:
            result["converged"] = False
            result["errors"].append(str(e))

        return result


class StaticCalculation(BaseCalculation):
    """Single-point energy calculation.

    Examples
    --------
    >>> # Direct mode with MLIP
    >>> from atomix.calculators import MACECalculator
    >>> mlip = MACECalculator(model="medium")
    >>> calc = StaticCalculation(atoms, calculator=mlip.get_calculator())
    >>> calc.run()
    >>> print(calc.results["energy"])

    >>> # File mode for VASP
    >>> calc = StaticCalculation(atoms, directory="./static", ENCUT=400)
    >>> calc.setup()  # Writes VASP input files
    >>> # Submit job externally, then:
    >>> results = calc.parse_results()
    """

    calc_type = "static"

    def setup(self) -> dict[str, Path]:
        """Set up VASP input files for static calculation.

        Returns
        -------
        dict[str, Path]
            Paths to generated input files.
        """
        kpoints = self.parameters.pop("kpoints", None)
        calculator = VASPCalculator(self.directory, **self.parameters)
        return calculator.write_inputs(self.atoms, self.calc_type, kpoints=kpoints)

    def run(self) -> None:
        """Execute the calculation.

        If a calculator is attached, runs directly.
        Otherwise, raises NotImplementedError (use job submission externally).
        """
        if self.calculator is not None:
            self._results = self._run_with_ase_calculator()
            self.atoms = self._results.get("atoms", self.atoms)
        else:
            # File-based mode - need external job submission
            raise NotImplementedError(
                "File-based execution requires external job submission. "
                "Use atomix submit or integrate with job scheduler."
            )

    def parse_results(self) -> dict[str, Any]:
        """Parse static calculation results.

        Returns
        -------
        dict[str, Any]
            Results including energy, forces, stress.
        """
        if self._results:
            return self._results

        calculator = VASPCalculator(self.directory)
        self._results = calculator.read_outputs()
        return self._results


class RelaxCalculation(BaseCalculation):
    """Geometry optimization calculation.

    Parameters
    ----------
    atoms : Atoms
        Initial structure.
    directory : Path | str
        Working directory.
    calculator : Calculator | None
        ASE calculator for direct mode.
    fmax : float
        Force convergence criterion (eV/Å) for direct mode.
    steps : int
        Maximum optimization steps for direct mode.
    optimizer : str
        Optimizer for direct mode: 'BFGS', 'LBFGS', 'FIRE'.
    **kwargs
        VASP parameters for file mode.
    """

    calc_type = "relax"

    def __init__(
        self,
        atoms: Atoms,
        directory: Path | str = ".",
        calculator: Calculator | None = None,
        fmax: float = 0.05,
        steps: int = 500,
        optimizer: str = "BFGS",
        **kwargs: Any,
    ) -> None:
        super().__init__(atoms, directory, calculator, **kwargs)
        self.fmax = fmax
        self.max_steps = steps
        self.optimizer = optimizer

    def setup(self) -> dict[str, Path]:
        """Set up VASP input files for relaxation calculation.

        Returns
        -------
        dict[str, Path]
            Paths to generated input files.
        """
        kpoints = self.parameters.pop("kpoints", None)
        calculator = VASPCalculator(self.directory, **self.parameters)
        return calculator.write_inputs(self.atoms, self.calc_type, kpoints=kpoints)

    def run(self) -> None:
        """Execute the relaxation.

        If a calculator is attached, runs ASE optimizer directly.
        Otherwise, raises NotImplementedError.
        """
        if self.calculator is not None:
            self._run_ase_relaxation()
        else:
            raise NotImplementedError(
                "File-based execution requires external job submission. "
                "Use atomix submit or integrate with job scheduler."
            )

    def _run_ase_relaxation(self) -> None:
        """Run relaxation using ASE optimizer."""
        atoms = self.atoms.copy()
        atoms.calc = self.calculator

        trajectory: list[Atoms] = []

        def save_frame() -> None:
            trajectory.append(atoms.copy())

        # Select optimizer
        optimizers = {
            "BFGS": BFGS,
            "LBFGS": LBFGS,
            "FIRE": FIRE,
        }
        opt_class = optimizers.get(self.optimizer.upper(), BFGS)
        opt = opt_class(atoms, logfile=None)
        opt.attach(save_frame)

        # Run
        converged = opt.run(fmax=self.fmax, steps=self.max_steps)

        # Store results
        forces = atoms.get_forces()
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))

        self._results = {
            "converged": converged and max_force <= self.fmax,
            "energy": atoms.get_potential_energy(),
            "forces": forces,
            "stress": None,
            "atoms": atoms,
            "n_steps": opt.nsteps,
            "trajectory": trajectory,
            "errors": [],
            "warnings": [],
        }

        try:
            if atoms.pbc.any():
                self._results["stress"] = atoms.get_stress(voigt=False)
        except Exception:
            pass

        if not self._results["converged"]:
            self._results["warnings"].append(
                f"Max force {max_force:.4f} > fmax {self.fmax}"
            )

        # Update atoms with relaxed structure
        self.atoms = atoms

    def parse_results(self) -> dict[str, Any]:
        """Parse relaxation calculation results.

        Returns
        -------
        dict[str, Any]
            Results including final energy, structure, trajectory.
        """
        if self._results:
            return self._results

        calculator = VASPCalculator(self.directory)
        self._results = calculator.read_outputs()

        # Update atoms with final structure
        if self._results.get("atoms") is not None:
            self.atoms = self._results["atoms"]

        return self._results


class AIMDCalculation(BaseCalculation):
    """Ab initio molecular dynamics calculation.

    Parameters
    ----------
    atoms : Atoms
        Initial structure.
    directory : Path | str
        Working directory.
    calculator : Calculator | None
        ASE calculator for direct mode.
    temperature : float
        Temperature in Kelvin.
    timestep : float
        Time step in fs.
    steps : int
        Number of MD steps.
    **kwargs
        Additional parameters (VASP settings for file mode).
    """

    def __init__(
        self,
        atoms: Atoms,
        directory: Path | str = ".",
        calculator: Calculator | None = None,
        temperature: float = 300.0,
        timestep: float = 1.0,
        steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(atoms, directory, calculator, **kwargs)
        self.temperature = temperature
        self.timestep = timestep
        self.md_steps = steps

    def setup(self) -> dict[str, Path]:
        """Set up VASP input files for AIMD calculation."""
        params = self.parameters.copy()
        params.update({
            "IBRION": 0,
            "NSW": self.md_steps,
            "POTIM": self.timestep,
            "SMASS": 0,  # Nose-Hoover thermostat
            "TEBEG": self.temperature,
            "TEEND": self.temperature,
        })

        kpoints = params.pop("kpoints", None)
        calculator = VASPCalculator(self.directory, **params)
        return calculator.write_inputs(self.atoms, "aimd_nvt", kpoints=kpoints)

    def run(self) -> None:
        """Execute MD simulation."""
        if self.calculator is not None:
            self._run_ase_md()
        else:
            raise NotImplementedError(
                "File-based execution requires external job submission."
            )

    def _run_ase_md(self) -> None:
        """Run MD using ASE dynamics."""
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        atoms = self.atoms.copy()
        atoms.calc = self.calculator

        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        trajectory: list[Atoms] = []

        def save_frame() -> None:
            trajectory.append(atoms.copy())

        # Set up Langevin dynamics (NVT)
        dyn = Langevin(
            atoms,
            timestep=self.timestep * units.fs,
            temperature_K=self.temperature,
            friction=0.01 / units.fs,
            logfile=None,
        )
        dyn.attach(save_frame, interval=1)

        # Run
        dyn.run(self.md_steps)

        self._results = {
            "converged": True,
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stress": None,
            "atoms": atoms,
            "n_steps": self.md_steps,
            "trajectory": trajectory,
            "errors": [],
            "warnings": [],
        }

        self.atoms = atoms

    def parse_results(self) -> dict[str, Any]:
        """Parse MD calculation results."""
        if self._results:
            return self._results

        calculator = VASPCalculator(self.directory)
        self._results = calculator.read_outputs()
        return self._results


class NVTCalculation(AIMDCalculation):
    """NVT ensemble molecular dynamics."""

    pass


class NPTCalculation(AIMDCalculation):
    """NPT ensemble molecular dynamics.

    Note: ASE direct mode uses Langevin barostat.
    VASP mode requires appropriate MDALGO settings.
    """

    def __init__(
        self,
        atoms: Atoms,
        directory: Path | str = ".",
        calculator: Calculator | None = None,
        temperature: float = 300.0,
        pressure: float = 1.0,  # bar
        timestep: float = 1.0,
        steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(atoms, directory, calculator, temperature, timestep, steps, **kwargs)
        self.pressure = pressure

    def setup(self) -> dict[str, Path]:
        """Set up VASP input files for NPT AIMD."""
        params = self.parameters.copy()
        params.update({
            "IBRION": 0,
            "NSW": self.md_steps,
            "POTIM": self.timestep,
            "MDALGO": 3,  # Langevin thermostat with Parrinello-Rahman
            "TEBEG": self.temperature,
            "TEEND": self.temperature,
            "PSTRESS": self.pressure * 10,  # kbar
            "LANGEVIN_GAMMA_L": 10,
        })

        kpoints = params.pop("kpoints", None)
        calculator = VASPCalculator(self.directory, **params)
        return calculator.write_inputs(self.atoms, "aimd_nvt", kpoints=kpoints)


class NVECalculation(AIMDCalculation):
    """NVE ensemble molecular dynamics."""

    def _run_ase_md(self) -> None:
        """Run NVE MD using ASE VelocityVerlet."""
        from ase import units
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet

        atoms = self.atoms.copy()
        atoms.calc = self.calculator

        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)

        trajectory: list[Atoms] = []

        def save_frame() -> None:
            trajectory.append(atoms.copy())

        dyn = VelocityVerlet(atoms, timestep=self.timestep * units.fs, logfile=None)
        dyn.attach(save_frame, interval=1)
        dyn.run(self.md_steps)

        self._results = {
            "converged": True,
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stress": None,
            "atoms": atoms,
            "n_steps": self.md_steps,
            "trajectory": trajectory,
            "errors": [],
            "warnings": [],
        }

        self.atoms = atoms


class NEBCalculation(BaseCalculation):
    """Nudged elastic band calculation for transition states."""

    def __init__(
        self,
        atoms: Atoms,
        final_atoms: Atoms,
        directory: Path | str = ".",
        calculator: Calculator | None = None,
        n_images: int = 7,
        climb: bool = True,
        fmax: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(atoms, directory, calculator, **kwargs)
        self.final_atoms = final_atoms
        self.n_images = n_images
        self.climb = climb
        self.fmax = fmax

    def setup(self) -> None:
        """Set up NEB calculation.

        For VASP, this would create multiple image directories.
        """
        raise NotImplementedError("NEB setup not yet implemented")

    def run(self) -> None:
        """Execute NEB calculation."""
        if self.calculator is not None:
            self._run_ase_neb()
        else:
            raise NotImplementedError("VASP NEB requires external job submission")

    def _run_ase_neb(self) -> None:
        """Run NEB using ASE."""
        from ase.mep import NEB

        # Create images
        images = [self.atoms.copy()]
        for _ in range(self.n_images - 2):
            images.append(self.atoms.copy())
        images.append(self.final_atoms.copy())

        # Interpolate
        neb = NEB(images, climb=self.climb)
        neb.interpolate()

        # Set calculators
        for image in images[1:-1]:
            image.calc = self.calculator

        # Optimize
        opt = BFGS(neb, logfile=None)
        opt.run(fmax=self.fmax)

        # Get barrier
        energies = [img.get_potential_energy() for img in images]

        self._results = {
            "converged": True,
            "images": images,
            "energies": energies,
            "barrier": max(energies) - energies[0],
            "n_steps": opt.nsteps,
            "errors": [],
            "warnings": [],
        }

    def parse_results(self) -> dict[str, Any]:
        if self._results:
            return self._results
        raise NotImplementedError("VASP NEB parsing not yet implemented")


class FrequencyCalculation(BaseCalculation):
    """Vibrational frequency calculation."""

    def __init__(
        self,
        atoms: Atoms,
        directory: Path | str = ".",
        calculator: Calculator | None = None,
        delta: float = 0.01,
        indices: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(atoms, directory, calculator, **kwargs)
        self.delta = delta
        self.indices = indices  # Atoms to displace (None = all)

    def setup(self) -> dict[str, Path]:
        """Set up VASP input files for frequency calculation."""
        params = self.parameters.copy()
        params.update({
            "IBRION": 5,  # Finite differences
            "NSW": 1,
            "POTIM": self.delta,
            "NFREE": 2,  # Central differences
        })

        kpoints = params.pop("kpoints", None)
        calculator = VASPCalculator(self.directory, **params)
        return calculator.write_inputs(self.atoms, "static", kpoints=kpoints)

    def run(self) -> None:
        """Execute frequency calculation."""
        if self.calculator is not None:
            self._run_ase_vibrations()
        else:
            raise NotImplementedError("VASP frequency requires external job submission")

    def _run_ase_vibrations(self) -> None:
        """Run vibrational analysis using ASE."""
        from ase.vibrations import Vibrations

        atoms = self.atoms.copy()
        atoms.calc = self.calculator

        vib = Vibrations(atoms, delta=self.delta, indices=self.indices)
        vib.run()

        # Get frequencies
        frequencies = vib.get_frequencies()
        energies = vib.get_energies()

        self._results = {
            "converged": True,
            "frequencies_cm": frequencies,  # cm^-1
            "energies_meV": energies * 1000,  # meV
            "n_modes": len(frequencies),
            "errors": [],
            "warnings": [],
        }

        # Check for imaginary frequencies
        imaginary = [f for f in frequencies if f.imag > 0]
        if imaginary:
            self._results["warnings"].append(
                f"Found {len(imaginary)} imaginary frequencies"
            )
            self._results["imaginary_frequencies"] = imaginary

        vib.clean()

    def parse_results(self) -> dict[str, Any]:
        if self._results:
            return self._results
        raise NotImplementedError("VASP frequency parsing not yet implemented")
