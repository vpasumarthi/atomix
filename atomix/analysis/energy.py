"""Energy analysis utilities for atomix."""

from typing import Any

from ase import Atoms


class EnergyAnalyzer:
    """Analyze energies from calculations.

    Parameters
    ----------
    reference_energies : dict[str, float] | None
        Reference energies for species (e.g., gas phase molecules).
    """

    def __init__(self, reference_energies: dict[str, float] | None = None) -> None:
        self.reference_energies = reference_energies or {}

    def formation_energy(
        self,
        atoms: Atoms,
        total_energy: float,
        composition_refs: dict[str, float],
    ) -> float:
        """Calculate formation energy.

        Parameters
        ----------
        atoms : Atoms
            Structure.
        total_energy : float
            DFT total energy.
        composition_refs : dict[str, float]
            Reference energies per atom for each element.

        Returns
        -------
        float
            Formation energy.
        """
        raise NotImplementedError

    def reaction_energy(
        self,
        reactants: list[tuple[float, float]],
        products: list[tuple[float, float]],
    ) -> float:
        """Calculate reaction energy.

        Parameters
        ----------
        reactants : list[tuple[float, float]]
            List of (stoichiometry, energy) for reactants.
        products : list[tuple[float, float]]
            List of (stoichiometry, energy) for products.

        Returns
        -------
        float
            Reaction energy.
        """
        e_reactants = sum(n * e for n, e in reactants)
        e_products = sum(n * e for n, e in products)
        return e_products - e_reactants
