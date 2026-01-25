"""Tests for adsorption energy workflows."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from atomix.analysis.adsorption import AdsorptionAnalyzer
from atomix.sites.surface import (
    SurfaceSite,
    add_adsorbate_at_site,
    find_surface_sites,
)


class TestSurfaceSite:
    """Tests for SurfaceSite data class."""

    def test_create_site(self) -> None:
        """Test creating a surface site."""
        pos = np.array([1.0, 2.0, 3.0])
        site = SurfaceSite(pos, "top", [0])
        assert site.site_type == "top"
        assert site.atoms_indices == [0]
        np.testing.assert_array_equal(site.position, pos)

    def test_repr(self) -> None:
        """Test string representation."""
        site = SurfaceSite(np.array([0, 0, 5]), "bridge", [0, 1])
        assert "bridge" in repr(site)


class TestFindSurfaceSites:
    """Tests for find_surface_sites function."""

    @pytest.fixture
    def cu111_slab(self) -> Atoms:
        """Create a Cu(111) 2x2 slab for testing."""
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        return slab

    def test_find_sites_returns_list(self, cu111_slab: Atoms) -> None:
        """Test that find_surface_sites returns a list of sites."""
        sites = find_surface_sites(cu111_slab, height=1.5)
        assert isinstance(sites, list)
        assert len(sites) > 0
        assert all(isinstance(s, SurfaceSite) for s in sites)

    def test_find_top_sites(self, cu111_slab: Atoms) -> None:
        """Test that top sites are found."""
        sites = find_surface_sites(cu111_slab, height=1.5)
        top_sites = [s for s in sites if s.site_type == "top"]
        # 2x2 slab should have 4 top sites
        assert len(top_sites) == 4

    def test_find_bridge_sites(self, cu111_slab: Atoms) -> None:
        """Test that bridge sites are found."""
        sites = find_surface_sites(cu111_slab, height=1.5)
        bridge_sites = [s for s in sites if s.site_type == "bridge"]
        assert len(bridge_sites) > 0

    def test_find_hollow_sites(self, cu111_slab: Atoms) -> None:
        """Test that hollow sites (fcc/hcp) are found."""
        sites = find_surface_sites(cu111_slab, height=1.5)
        hollow_sites = [s for s in sites if s.site_type in ("fcc", "hcp", "hollow")]
        assert len(hollow_sites) > 0

    def test_site_height(self, cu111_slab: Atoms) -> None:
        """Test that sites are at correct height above surface."""
        height = 2.0
        sites = find_surface_sites(cu111_slab, height=height)
        z_max = cu111_slab.get_positions()[:, 2].max()
        expected_z = z_max + height

        for site in sites:
            assert abs(site.position[2] - expected_z) < 0.1

    def test_small_slab(self) -> None:
        """Test with minimal slab (edge case)."""
        # Single atom slab
        slab = Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 10], pbc=True)
        sites = find_surface_sites(slab, height=1.5)
        # Should still return at least the top site
        assert len(sites) >= 1


class TestAddAdsorbateAtSite:
    """Tests for add_adsorbate_at_site function."""

    @pytest.fixture
    def cu111_slab(self) -> Atoms:
        """Create a Cu(111) slab for testing."""
        return fcc111("Cu", size=(2, 2, 3), vacuum=10.0)

    def test_add_single_atom(self, cu111_slab: Atoms) -> None:
        """Test adding a single atom adsorbate."""
        site = SurfaceSite(np.array([1.0, 1.0, 15.0]), "top", [0])
        result = add_adsorbate_at_site(cu111_slab, "O", site)

        # Original slab should be unchanged
        assert len(cu111_slab) == 12  # 2x2x3 = 12 atoms

        # New structure should have one more atom
        assert len(result) == 13
        assert "O" in result.get_chemical_symbols()

    def test_add_molecule(self, cu111_slab: Atoms) -> None:
        """Test adding a molecule adsorbate (CO)."""
        site = SurfaceSite(np.array([1.0, 1.0, 15.0]), "top", [0])
        result = add_adsorbate_at_site(cu111_slab, "CO", site)

        # Should have 2 more atoms (C and O)
        assert len(result) == 14
        symbols = result.get_chemical_symbols()
        assert "C" in symbols
        assert "O" in symbols

    def test_original_unchanged(self, cu111_slab: Atoms) -> None:
        """Test that original slab is not modified."""
        original_positions = cu111_slab.get_positions().copy()
        site = SurfaceSite(np.array([1.0, 1.0, 15.0]), "top", [0])
        _ = add_adsorbate_at_site(cu111_slab, "O", site)

        np.testing.assert_array_equal(cu111_slab.get_positions(), original_positions)

    def test_custom_height(self, cu111_slab: Atoms) -> None:
        """Test adding adsorbate at custom height."""
        z_max = cu111_slab.get_positions()[:, 2].max()
        site = SurfaceSite(np.array([1.0, 1.0, z_max + 1.5]), "top", [0])

        custom_height = 2.5
        result = add_adsorbate_at_site(cu111_slab, "O", site, height=custom_height)

        # O atom should be at z_max + custom_height
        o_positions = result.get_positions()[-1]
        assert abs(o_positions[2] - (z_max + custom_height)) < 0.1


class TestAdsorptionAnalyzer:
    """Tests for AdsorptionAnalyzer class."""

    def test_adsorption_energy_basic(self) -> None:
        """Test basic adsorption energy calculation."""
        # E_ads = E(slab+ads) - E(slab) - E(gas)
        slab_energy = -100.0
        gas_reference = -5.0
        slab_ads_energy = -108.0

        analyzer = AdsorptionAnalyzer(
            slab_energy=slab_energy,
            gas_references={"O": gas_reference},
        )

        e_ads = analyzer.adsorption_energy(slab_ads_energy, "O")

        # -108 - (-100) - (-5) = -108 + 100 + 5 = -3.0
        assert abs(e_ads - (-3.0)) < 1e-10

    def test_adsorption_energy_explicit_ref(self) -> None:
        """Test adsorption energy with explicit gas reference."""
        analyzer = AdsorptionAnalyzer(slab_energy=-100.0)

        e_ads = analyzer.adsorption_energy(-107.0, "O", gas_reference=-4.0)
        # -107 - (-100) - (-4) = -107 + 100 + 4 = -3.0
        assert abs(e_ads - (-3.0)) < 1e-10

    def test_missing_gas_reference(self) -> None:
        """Test error when gas reference is missing."""
        analyzer = AdsorptionAnalyzer(slab_energy=-100.0)

        with pytest.raises(ValueError, match="No gas reference"):
            analyzer.adsorption_energy(-107.0, "O")


class TestCoverageEnergy:
    """Tests for coverage_energy method."""

    def test_coverage_energy_calculation(self) -> None:
        """Test coverage energy calculation."""
        slab_energy = -100.0
        gas_ref = -5.0

        analyzer = AdsorptionAnalyzer(
            slab_energy=slab_energy,
            gas_references={"O": gas_ref},
        )

        # Energies for 1, 2, 3 adsorbates
        # Assume each additional O stabilizes by -3 eV
        energies = [-108.0, -116.0, -124.0]  # slab + 1O, 2O, 3O
        n_adsorbates = [1, 2, 3]

        result = analyzer.coverage_energy(energies, n_adsorbates, "O")

        assert "n" in result
        assert "average" in result
        assert "differential" in result
        assert "total" in result

        # Check n values
        assert result["n"] == [1, 2, 3]

        # Total E_ads for n=1: -108 - (-100) - 1*(-5) = -3
        # Total E_ads for n=2: -116 - (-100) - 2*(-5) = -6
        # Total E_ads for n=3: -124 - (-100) - 3*(-5) = -9
        np.testing.assert_array_almost_equal(result["total"], [-3.0, -6.0, -9.0])

        # Average E_ads per adsorbate: total / n
        np.testing.assert_array_almost_equal(result["average"], [-3.0, -3.0, -3.0])

    def test_coverage_energy_differential(self) -> None:
        """Test differential adsorption energy."""
        slab_energy = -100.0
        gas_ref = -5.0

        analyzer = AdsorptionAnalyzer(
            slab_energy=slab_energy,
            gas_references={"O": gas_ref},
        )

        # First O: -3 eV, second O: -2.5 eV (weaker due to repulsion)
        energies = [-108.0, -115.5]  # slab + 1O, 2O
        n_adsorbates = [1, 2]

        result = analyzer.coverage_energy(energies, n_adsorbates, "O")

        # Differential for n=1: -108 - (-100) - 1*(-5) = -3
        # Differential for n=2: -115.5 - (-108) - 1*(-5) = -2.5
        np.testing.assert_array_almost_equal(result["differential"], [-3.0, -2.5])

    def test_coverage_energy_unsorted(self) -> None:
        """Test that coverage_energy handles unsorted input."""
        analyzer = AdsorptionAnalyzer(
            slab_energy=-100.0,
            gas_references={"O": -5.0},
        )

        # Unsorted input
        energies = [-116.0, -108.0, -124.0]
        n_adsorbates = [2, 1, 3]

        result = analyzer.coverage_energy(energies, n_adsorbates, "O")

        # Should be sorted by n
        assert result["n"] == [1, 2, 3]
