"""Adsorption energy analysis for catalysis workflows."""

from typing import Any

from ase import Atoms


class AdsorptionAnalyzer:
    """Calculate adsorption energies for surface catalysis.

    E_ads = E(slab+adsorbate) - E(slab) - E(adsorbate_gas)

    Parameters
    ----------
    slab_energy : float
        Energy of clean slab.
    gas_references : dict[str, float]
        Reference energies for gas phase species.
    """

    def __init__(
        self,
        slab_energy: float,
        gas_references: dict[str, float] | None = None,
    ) -> None:
        self.slab_energy = slab_energy
        self.gas_references = gas_references or {}

    def adsorption_energy(
        self,
        slab_adsorbate_energy: float,
        adsorbate: str,
        gas_reference: float | None = None,
    ) -> float:
        """Calculate adsorption energy.

        Parameters
        ----------
        slab_adsorbate_energy : float
            Energy of slab with adsorbate.
        adsorbate : str
            Adsorbate species identifier.
        gas_reference : float | None
            Gas phase reference energy. If None, uses stored reference.

        Returns
        -------
        float
            Adsorption energy.
        """
        if gas_reference is None:
            if adsorbate not in self.gas_references:
                raise ValueError(f"No gas reference for {adsorbate}")
            gas_reference = self.gas_references[adsorbate]

        return slab_adsorbate_energy - self.slab_energy - gas_reference

    def coverage_energy(
        self,
        energies: list[float],
        n_adsorbates: list[int],
        adsorbate: str,
    ) -> list[float]:
        """Calculate adsorption energy as function of coverage.

        Parameters
        ----------
        energies : list[float]
            Total energies at each coverage.
        n_adsorbates : list[int]
            Number of adsorbates at each coverage.
        adsorbate : str
            Adsorbate species.

        Returns
        -------
        list[float]
            Adsorption energy per adsorbate at each coverage.
        """
        raise NotImplementedError
