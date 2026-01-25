"""Tests for analysis modules (trajectory and energy)."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from atomix.analysis.energy import EnergyAnalyzer
from atomix.analysis.trajectory import TrajectoryAnalyzer


class TestTrajectoryAnalyzer:
    """Tests for TrajectoryAnalyzer class."""

    @pytest.fixture
    def simple_trajectory(self) -> list[Atoms]:
        """Create a simple trajectory for testing."""
        # Create a simple cubic cell with a few atoms
        atoms = Atoms(
            "Cu4",
            positions=[
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [2, 2, 0],
            ],
            cell=[4, 4, 4],
            pbc=True,
        )

        # Create trajectory with small displacements
        trajectory = []
        for i in range(20):
            frame = atoms.copy()
            # Add small random displacements
            frame.positions += np.random.randn(4, 3) * 0.1 * (i + 1) / 20
            trajectory.append(frame)

        return trajectory

    @pytest.fixture
    def diffusive_trajectory(self) -> list[Atoms]:
        """Create a trajectory with known diffusive motion."""
        # Create atoms in a box
        atoms = Atoms(
            "O4",
            positions=[
                [5, 5, 5],
                [5, 5, 15],
                [5, 15, 5],
                [15, 5, 5],
            ],
            cell=[20, 20, 20],
            pbc=True,
        )

        # Create trajectory with random walk (diffusive motion)
        np.random.seed(42)
        trajectory = [atoms.copy()]
        current_positions = atoms.positions.copy()

        for _ in range(100):
            # Random walk step
            step = np.random.randn(4, 3) * 0.5
            current_positions += step
            frame = atoms.copy()
            frame.positions = current_positions.copy()
            trajectory.append(frame)

        return trajectory

    def test_init_from_list(self, simple_trajectory: list[Atoms]) -> None:
        """Test initialization from list of Atoms."""
        analyzer = TrajectoryAnalyzer(simple_trajectory)
        assert len(analyzer.trajectory) == 20

    def test_rdf_basic(self, simple_trajectory: list[Atoms]) -> None:
        """Test basic RDF calculation."""
        analyzer = TrajectoryAnalyzer(simple_trajectory)
        r, g_r = analyzer.rdf(("Cu", "Cu"), rmax=4.0, nbins=50)

        assert len(r) == 50
        assert len(g_r) == 50
        assert r[0] > 0  # First bin center should be positive
        assert r[-1] < 4.0  # Should not exceed rmax

    def test_rdf_shape(self, simple_trajectory: list[Atoms]) -> None:
        """Test RDF output shape."""
        analyzer = TrajectoryAnalyzer(simple_trajectory)
        r, g_r = analyzer.rdf(("Cu", "Cu"), rmax=6.0, nbins=100)

        assert r.shape == (100,)
        assert g_r.shape == (100,)

    def test_rdf_peak_location(self) -> None:
        """Test that RDF shows peak at expected distance."""
        # Create a simple FCC structure
        cu_bulk = bulk("Cu", "fcc", a=3.6)
        cu_supercell = cu_bulk * (3, 3, 3)

        # Single frame trajectory
        trajectory = [cu_supercell]
        analyzer = TrajectoryAnalyzer(trajectory)

        r, g_r = analyzer.rdf(("Cu", "Cu"), rmax=5.0, nbins=100)

        # FCC nearest neighbor distance is a/sqrt(2) ~ 2.55 Å
        peak_idx = np.argmax(g_r)
        peak_r = r[peak_idx]
        assert 2.4 < peak_r < 2.7  # Should be around 2.55 Å

    def test_msd_basic(self, simple_trajectory: list[Atoms]) -> None:
        """Test basic MSD calculation."""
        analyzer = TrajectoryAnalyzer(simple_trajectory)
        msd = analyzer.msd()

        assert len(msd) == 20
        assert msd[0] == 0.0  # MSD at t=0 should be 0

    def test_msd_increasing(self, diffusive_trajectory: list[Atoms]) -> None:
        """Test that MSD generally increases with time."""
        analyzer = TrajectoryAnalyzer(diffusive_trajectory)
        msd = analyzer.msd()

        # MSD should generally increase (allow some noise)
        # Check that later values are larger than early values
        assert msd[-1] > msd[10]

    def test_msd_element_filter(self, simple_trajectory: list[Atoms]) -> None:
        """Test MSD with element filter."""
        analyzer = TrajectoryAnalyzer(simple_trajectory)
        msd = analyzer.msd(element="Cu")

        assert len(msd) == 20

    def test_msd_nonexistent_element(self, simple_trajectory: list[Atoms]) -> None:
        """Test MSD with non-existent element."""
        analyzer = TrajectoryAnalyzer(simple_trajectory)
        msd = analyzer.msd(element="O")

        # Should return zeros for non-existent element
        assert np.all(msd == 0)

    def test_diffusion_coefficient(self, diffusive_trajectory: list[Atoms]) -> None:
        """Test diffusion coefficient calculation."""
        analyzer = TrajectoryAnalyzer(diffusive_trajectory)
        result = analyzer.diffusion_coefficient(timestep=1.0)

        assert "D" in result
        assert "D_error" in result
        assert "r_squared" in result
        assert result["D"] >= 0  # Diffusion coefficient should be non-negative

    def test_diffusion_coefficient_units(self, diffusive_trajectory: list[Atoms]) -> None:
        """Test that diffusion coefficient has reasonable magnitude."""
        analyzer = TrajectoryAnalyzer(diffusive_trajectory)
        result = analyzer.diffusion_coefficient(timestep=1.0)

        # For random walk with step ~0.5 Å per fs, D should be on order of 1e-5 cm²/s
        # This is a rough check for unit conversion
        assert result["D"] > 0
        assert result["D"] < 1.0  # Should not be unreasonably large


class TestEnergyAnalyzer:
    """Tests for EnergyAnalyzer class."""

    def test_formation_energy_basic(self) -> None:
        """Test basic formation energy calculation."""
        # Cu reference: -3.5 eV/atom
        refs = {"Cu": -3.5}
        analyzer = EnergyAnalyzer(reference_energies=refs)

        # 4-atom Cu structure with total energy -15.0 eV
        atoms = Atoms("Cu4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        total_energy = -15.0

        # E_form = -15.0 - 4*(-3.5) = -15.0 + 14.0 = -1.0 eV
        e_form = analyzer.formation_energy(atoms, total_energy)
        assert abs(e_form - (-1.0)) < 1e-10

    def test_formation_energy_binary(self) -> None:
        """Test formation energy for binary compound."""
        # References: Cu = -3.5 eV/atom, O = -4.0 eV/atom
        refs = {"Cu": -3.5, "O": -4.0}
        analyzer = EnergyAnalyzer(reference_energies=refs)

        # CuO structure (1 Cu + 1 O)
        atoms = Atoms("CuO", positions=[[0, 0, 0], [1, 0, 0]])
        total_energy = -10.0

        # E_form = -10.0 - (1*(-3.5) + 1*(-4.0)) = -10.0 + 7.5 = -2.5 eV
        e_form = analyzer.formation_energy(atoms, total_energy)
        assert abs(e_form - (-2.5)) < 1e-10

    def test_formation_energy_explicit_refs(self) -> None:
        """Test formation energy with explicit references."""
        analyzer = EnergyAnalyzer()  # No stored references

        atoms = Atoms("Cu4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        total_energy = -14.0

        # Pass references explicitly
        refs = {"Cu": -3.5}
        e_form = analyzer.formation_energy(atoms, total_energy, composition_refs=refs)
        assert abs(e_form - (0.0)) < 1e-10  # -14 - 4*(-3.5) = 0

    def test_formation_energy_missing_ref(self) -> None:
        """Test error when reference is missing."""
        refs = {"Cu": -3.5}  # Missing O reference
        analyzer = EnergyAnalyzer(reference_energies=refs)

        atoms = Atoms("CuO", positions=[[0, 0, 0], [1, 0, 0]])

        with pytest.raises(ValueError, match="No reference energy"):
            analyzer.formation_energy(atoms, -10.0)

    def test_formation_energy_per_atom(self) -> None:
        """Test formation energy per atom."""
        refs = {"Cu": -3.5}
        analyzer = EnergyAnalyzer(reference_energies=refs)

        atoms = Atoms("Cu4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        total_energy = -15.0

        # E_form = -1.0 eV, E_form/atom = -0.25 eV/atom
        e_form_per_atom = analyzer.formation_energy_per_atom(atoms, total_energy)
        assert abs(e_form_per_atom - (-0.25)) < 1e-10

    def test_reaction_energy(self) -> None:
        """Test reaction energy calculation."""
        analyzer = EnergyAnalyzer()

        # Example: 2H2 + O2 -> 2H2O
        # Energies (made up): H2 = -6.0 eV, O2 = -9.0 eV, H2O = -14.0 eV
        reactants = [(2, -6.0), (1, -9.0)]  # 2 H2 + 1 O2
        products = [(2, -14.0)]  # 2 H2O

        # E_rxn = 2*(-14) - (2*(-6) + 1*(-9)) = -28 + 21 = -7 eV
        e_rxn = analyzer.reaction_energy(reactants, products)
        assert abs(e_rxn - (-7.0)) < 1e-10

    def test_reaction_energy_endothermic(self) -> None:
        """Test endothermic reaction energy."""
        analyzer = EnergyAnalyzer()

        # Reverse reaction (decomposition)
        reactants = [(2, -14.0)]  # 2 H2O
        products = [(2, -6.0), (1, -9.0)]  # 2 H2 + O2

        # E_rxn = +7 eV (endothermic)
        e_rxn = analyzer.reaction_energy(reactants, products)
        assert abs(e_rxn - 7.0) < 1e-10
