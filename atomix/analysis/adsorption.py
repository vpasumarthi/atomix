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
        gas_reference: float | None = None,
    ) -> dict[str, list[float]]:
        """Calculate adsorption energy as function of coverage.

        Computes both average and differential adsorption energies.

        Parameters
        ----------
        energies : list[float]
            Total energies at each coverage (must be sorted by n_adsorbates).
        n_adsorbates : list[int]
            Number of adsorbates at each coverage.
        adsorbate : str
            Adsorbate species identifier.
        gas_reference : float | None
            Gas phase reference energy. If None, uses stored reference.

        Returns
        -------
        dict[str, list[float]]
            Dictionary with:
            - 'n': Number of adsorbates
            - 'average': Average E_ads per adsorbate at each coverage
            - 'differential': Differential E_ads for adding each adsorbate
            - 'total': Total adsorption energy at each coverage
        """
        if gas_reference is None:
            if adsorbate not in self.gas_references:
                raise ValueError(f"No gas reference for {adsorbate}")
            gas_reference = self.gas_references[adsorbate]

        # Sort by number of adsorbates
        sorted_pairs = sorted(zip(n_adsorbates, energies))
        n_sorted = [p[0] for p in sorted_pairs]
        e_sorted = [p[1] for p in sorted_pairs]

        average_e_ads: list[float] = []
        differential_e_ads: list[float] = []
        total_e_ads: list[float] = []

        prev_energy = self.slab_energy
        prev_n = 0

        for n, energy in zip(n_sorted, e_sorted):
            if n == 0:
                # Clean slab reference
                average_e_ads.append(0.0)
                differential_e_ads.append(0.0)
                total_e_ads.append(0.0)
            else:
                # Total adsorption energy
                e_tot = energy - self.slab_energy - n * gas_reference
                total_e_ads.append(e_tot)

                # Average adsorption energy per adsorbate
                e_avg = e_tot / n
                average_e_ads.append(e_avg)

                # Differential adsorption energy
                dn = n - prev_n
                if dn > 0:
                    e_diff = (energy - prev_energy - dn * gas_reference) / dn
                else:
                    e_diff = e_avg  # Fallback if dn=0
                differential_e_ads.append(e_diff)

            prev_energy = energy
            prev_n = n

        return {
            "n": n_sorted,
            "average": average_e_ads,
            "differential": differential_e_ads,
            "total": total_e_ads,
        }

    def batch_adsorption_energies(
        self,
        directories: list[str],
        adsorbate: str,
        gas_reference: float | None = None,
    ) -> list[dict[str, Any]]:
        """Calculate adsorption energies from multiple calculation directories.

        Parameters
        ----------
        directories : list[str]
            Paths to calculation directories containing VASP outputs.
        adsorbate : str
            Adsorbate species identifier.
        gas_reference : float | None
            Gas phase reference energy. If None, uses stored reference.

        Returns
        -------
        list[dict]
            List of results with directory, energy, and E_ads for each.
        """
        from pathlib import Path

        from atomix.calculators.vasp import VASPCalculator

        results = []
        for directory in directories:
            dir_path = Path(directory)
            calc = VASPCalculator(directory)

            try:
                outputs = calc.read_outputs()
                energy = outputs.get("energy")
                converged = outputs.get("converged", False)

                if energy is not None:
                    e_ads = self.adsorption_energy(
                        energy, adsorbate, gas_reference
                    )
                    results.append({
                        "directory": str(dir_path),
                        "name": dir_path.name,
                        "energy": energy,
                        "e_ads": e_ads,
                        "converged": converged,
                    })
                else:
                    results.append({
                        "directory": str(dir_path),
                        "name": dir_path.name,
                        "energy": None,
                        "e_ads": None,
                        "converged": False,
                        "error": "No energy found",
                    })
            except Exception as e:
                results.append({
                    "directory": str(dir_path),
                    "name": dir_path.name,
                    "energy": None,
                    "e_ads": None,
                    "converged": False,
                    "error": str(e),
                })

        return results
