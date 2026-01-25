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
        composition_refs: dict[str, float] | None = None,
    ) -> float:
        """Calculate formation energy.

        E_form = E_total - sum(n_i * E_ref_i)

        where n_i is the number of atoms of element i and E_ref_i is
        the reference energy per atom for element i.

        Parameters
        ----------
        atoms : Atoms
            Structure.
        total_energy : float
            DFT total energy of the structure.
        composition_refs : dict[str, float] | None
            Reference energies per atom for each element.
            If None, uses stored reference_energies.

        Returns
        -------
        float
            Formation energy in eV.
        """
        refs = composition_refs or self.reference_energies

        # Count atoms of each element
        symbols = atoms.get_chemical_symbols()
        element_counts: dict[str, int] = {}
        for sym in symbols:
            element_counts[sym] = element_counts.get(sym, 0) + 1

        # Calculate reference energy sum
        ref_energy_sum = 0.0
        for element, count in element_counts.items():
            if element not in refs:
                raise ValueError(
                    f"No reference energy for element '{element}'. "
                    f"Available: {list(refs.keys())}"
                )
            ref_energy_sum += count * refs[element]

        return total_energy - ref_energy_sum

    def formation_energy_per_atom(
        self,
        atoms: Atoms,
        total_energy: float,
        composition_refs: dict[str, float] | None = None,
    ) -> float:
        """Calculate formation energy per atom.

        Parameters
        ----------
        atoms : Atoms
            Structure.
        total_energy : float
            DFT total energy of the structure.
        composition_refs : dict[str, float] | None
            Reference energies per atom for each element.

        Returns
        -------
        float
            Formation energy per atom in eV/atom.
        """
        e_form = self.formation_energy(atoms, total_energy, composition_refs)
        return e_form / len(atoms)

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
