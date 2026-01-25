"""Workflow orchestration for atomix."""

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS, FIRE, LBFGS

from atomix.core.calculation import BaseCalculation


class Workflow:
    """Orchestrate multi-step calculation workflows.

    Supports both ASE-style direct execution (for MLIP calculators)
    and file-based execution (for DFT codes like VASP).

    Parameters
    ----------
    name : str
        Name identifier for the workflow.
    directory : Path | str
        Base directory for workflow outputs.

    Examples
    --------
    >>> from atomix.core import Workflow
    >>> from atomix.calculators import MACECalculator
    >>>
    >>> # Direct execution with MLIP
    >>> workflow = Workflow("screening")
    >>> mlip = MACECalculator(model="medium")
    >>> results = workflow.run_direct(atoms, mlip.get_calculator())
    >>>
    >>> # File-based execution (VASP)
    >>> workflow = Workflow("relaxation", directory="./calcs")
    >>> workflow.add_step(RelaxCalculation(atoms, "./calcs/step1"))
    >>> results = workflow.run_sequential()
    """

    def __init__(self, name: str, directory: Path | str = ".") -> None:
        self.name = name
        self.directory = Path(directory)
        self._steps: list[BaseCalculation] = []
        self._results: list[dict[str, Any]] = []

    def add_step(self, calculation: BaseCalculation) -> None:
        """Add a calculation step to the workflow."""
        self._steps.append(calculation)

    def run(self, atoms: Atoms, calculator: Calculator) -> list[dict[str, Any]]:
        """Execute the workflow with the given atoms and calculator.

        This is a convenience method that dispatches to run_direct()
        for ASE-style execution with the provided calculator.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object to run calculations on.
        calculator : Calculator
            ASE-compatible calculator to use.

        Returns
        -------
        list[dict[str, Any]]
            Results from each step of the workflow.
        """
        return self.run_direct(atoms, calculator)

    def run_direct(
        self,
        atoms: Atoms,
        calculator: Calculator,
        fmax: float = 0.05,
        steps: int = 100,
        optimizer: str = "BFGS",
    ) -> list[dict[str, Any]]:
        """Execute workflow directly using ASE calculator.

        Suitable for MLIP calculators where calculations run in-process.
        Performs single-point or relaxation based on steps parameter.

        Parameters
        ----------
        atoms : Atoms
            Structure to calculate.
        calculator : Calculator
            ASE-compatible calculator.
        fmax : float
            Maximum force for relaxation convergence (eV/Å).
        steps : int
            Maximum optimization steps. If 0, only single-point.
        optimizer : str
            Optimizer to use: 'BFGS', 'LBFGS', 'FIRE'.

        Returns
        -------
        list[dict[str, Any]]
            Results including energy, forces, and trajectory.
        """
        import numpy as np

        atoms_copy = atoms.copy()
        atoms_copy.calc = calculator

        result: dict[str, Any] = {
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

        try:
            if steps == 0:
                # Single-point calculation
                result["energy"] = atoms_copy.get_potential_energy()
                result["forces"] = atoms_copy.get_forces()
                try:
                    result["stress"] = atoms_copy.get_stress(voigt=False)
                except Exception:
                    pass
                result["atoms"] = atoms_copy.copy()
                result["trajectory"] = [atoms_copy.copy()]
                result["n_steps"] = 1
                result["converged"] = True
            else:
                # Relaxation
                trajectory: list[Atoms] = []

                def save_frame() -> None:
                    trajectory.append(atoms_copy.copy())

                # Select optimizer
                optimizers = {
                    "BFGS": BFGS,
                    "LBFGS": LBFGS,
                    "FIRE": FIRE,
                }
                opt_class = optimizers.get(optimizer.upper(), BFGS)
                opt = opt_class(atoms_copy, logfile=None)
                opt.attach(save_frame)

                # Run optimization
                converged = opt.run(fmax=fmax, steps=steps)

                result["energy"] = atoms_copy.get_potential_energy()
                result["forces"] = atoms_copy.get_forces()
                try:
                    result["stress"] = atoms_copy.get_stress(voigt=False)
                except Exception:
                    pass
                result["atoms"] = atoms_copy.copy()
                result["trajectory"] = trajectory
                result["n_steps"] = opt.nsteps
                result["converged"] = converged

                # Check if actually converged
                max_force = float(np.max(np.linalg.norm(result["forces"], axis=1)))
                if max_force > fmax:
                    result["converged"] = False
                    result["warnings"].append(
                        f"Max force {max_force:.4f} > fmax {fmax}"
                    )

        except Exception as e:
            result["errors"].append(str(e))

        self._results = [result]
        return self._results

    def run_sequential(self) -> list[dict[str, Any]]:
        """Execute workflow steps sequentially.

        Runs setup, execution, and parsing for each step in order.
        Steps must have been added via add_step().

        Returns
        -------
        list[dict[str, Any]]
            Results from each step.
        """
        self._results = []

        for i, step in enumerate(self._steps):
            try:
                # Setup input files
                step.setup()

                # Run the calculation
                step.run()

                # Parse results
                result = step.parse_results()
                self._results.append(result)

                # Pass final structure to next step if available
                if i < len(self._steps) - 1 and result.get("atoms") is not None:
                    self._steps[i + 1].atoms = result["atoms"]

            except Exception as e:
                self._results.append({
                    "converged": False,
                    "errors": [str(e)],
                    "step": i,
                })
                break

        return self._results

    def run_step(self, step_index: int) -> dict[str, Any]:
        """Execute a single workflow step.

        Parameters
        ----------
        step_index : int
            Index of the step to run.

        Returns
        -------
        dict[str, Any]
            Results from the step.
        """
        if step_index >= len(self._steps):
            raise IndexError(f"Step {step_index} does not exist")

        step = self._steps[step_index]
        step.setup()
        step.run()
        result = step.parse_results()

        # Store in results list
        while len(self._results) <= step_index:
            self._results.append({})
        self._results[step_index] = result

        return result

    @property
    def results(self) -> list[dict[str, Any]]:
        """Return results from all workflow steps."""
        return self._results

    @property
    def steps(self) -> list[BaseCalculation]:
        """Return workflow steps."""
        return self._steps

    def clear(self) -> None:
        """Clear all steps and results."""
        self._steps = []
        self._results = []


class RelaxationWorkflow(Workflow):
    """Convenience workflow for structure relaxation.

    Parameters
    ----------
    name : str
        Workflow name.
    fmax : float
        Force convergence criterion (eV/Å).
    steps : int
        Maximum optimization steps.
    optimizer : str
        Optimizer: 'BFGS', 'LBFGS', 'FIRE'.
    """

    def __init__(
        self,
        name: str = "relaxation",
        fmax: float = 0.05,
        steps: int = 500,
        optimizer: str = "BFGS",
    ) -> None:
        super().__init__(name)
        self.fmax = fmax
        self.max_steps = steps
        self.optimizer = optimizer

    def run(self, atoms: Atoms, calculator: Calculator) -> list[dict[str, Any]]:
        """Run relaxation workflow."""
        return self.run_direct(
            atoms,
            calculator,
            fmax=self.fmax,
            steps=self.max_steps,
            optimizer=self.optimizer,
        )


class ScreeningWorkflowSimple(Workflow):
    """Simple screening workflow for multiple structures.

    Evaluates multiple structures with single-point calculations
    and ranks by energy.
    """

    def run_screening(
        self,
        structures: list[Atoms],
        calculator: Calculator,
    ) -> list[dict[str, Any]]:
        """Screen multiple structures.

        Parameters
        ----------
        structures : list[Atoms]
            Structures to screen.
        calculator : Calculator
            ASE calculator to use.

        Returns
        -------
        list[dict[str, Any]]
            Results for each structure, sorted by energy.
        """
        results = []

        for i, atoms in enumerate(structures):
            result = self.run_direct(atoms, calculator, steps=0)[0]
            result["structure_index"] = i
            results.append(result)

        # Sort by energy
        valid = [r for r in results if r.get("energy") is not None]
        valid.sort(key=lambda r: r["energy"])

        # Add ranking
        for rank, r in enumerate(valid, 1):
            r["rank"] = rank

        self._results = valid + [r for r in results if r.get("energy") is None]
        return self._results
