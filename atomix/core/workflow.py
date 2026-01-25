"""Workflow orchestration for atomix."""

from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from atomix.core.calculation import BaseCalculation


class Workflow:
    """Orchestrate multi-step calculation workflows.

    Parameters
    ----------
    name : str
        Name identifier for the workflow.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[BaseCalculation] = []
        self._results: list[dict[str, Any]] = []

    def add_step(self, calculation: BaseCalculation) -> None:
        """Add a calculation step to the workflow."""
        self._steps.append(calculation)

    def run(self, atoms: Atoms, calculator: Calculator) -> list[dict[str, Any]]:
        """Execute the workflow with the given atoms and calculator.

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
        raise NotImplementedError

    @property
    def results(self) -> list[dict[str, Any]]:
        """Return results from all workflow steps."""
        return self._results
