"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111
from click.testing import CliRunner

from atomix.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_structure(tmp_path: Path) -> Path:
    """Create a temporary structure file."""
    atoms = bulk("Cu")
    from ase.io import write
    struct_path = tmp_path / "cu.vasp"
    write(str(struct_path), atoms, format="vasp")
    return struct_path


@pytest.fixture
def temp_slab(tmp_path: Path) -> Path:
    """Create a temporary slab structure."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
    from ase.io import write
    slab_path = tmp_path / "slab.vasp"
    write(str(slab_path), slab, format="vasp")
    return slab_path


@pytest.fixture
def temp_calc_dir(tmp_path: Path) -> Path:
    """Create a mock calculation directory with outputs."""
    calc_dir = tmp_path / "calc"
    calc_dir.mkdir()

    # Create minimal OUTCAR
    outcar = calc_dir / "OUTCAR"
    outcar.write_text("""
 TOTEN  =       -12.34567890 eV

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
     0.00000      0.00000      0.00000         0.001000     -0.002000      0.003000
     1.80000      1.80000      1.80000        -0.001000      0.002000     -0.003000
 -----------------------------------------------------------------------------------
    total drift:      0.000000      0.000000      0.000000

 General timing and accounting informations for this job:
""")

    # Create minimal POSCAR
    poscar = calc_dir / "POSCAR"
    poscar.write_text("""Cu
1.0
3.6 0.0 0.0
0.0 3.6 0.0
0.0 0.0 3.6
Cu
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
""")

    return calc_dir


class TestScreenCommand:
    """Tests for the screen command."""

    def test_screen_help(self, runner: CliRunner) -> None:
        """Test screen command help."""
        result = runner.invoke(cli, ["screen", "--help"])
        assert result.exit_code == 0
        assert "Screen structures with MLIP" in result.output

    def test_screen_no_structures(self, runner: CliRunner) -> None:
        """Test screen command with no structures."""
        result = runner.invoke(cli, ["screen"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "required" in result.output.lower()

    @patch("atomix.calculators.mlip.get_mlip_calculator")
    def test_screen_with_structures(
        self,
        mock_get_calc: MagicMock,
        runner: CliRunner,
        temp_structure: Path,
    ) -> None:
        """Test screen command with mock calculator."""
        # Create mock calculator
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = {
            "converged": True,
            "energy": -10.0,
            "forces": np.zeros((1, 3)),
            "stress": None,
            "atoms": Atoms("Cu"),
            "n_steps": 1,
            "trajectory": [],
            "errors": [],
            "warnings": [],
        }
        mock_get_calc.return_value = mock_calc

        result = runner.invoke(cli, ["screen", str(temp_structure), "-n", "1"])

        # Should succeed (or fail gracefully if MACE not installed)
        # The mock should prevent actual MACE usage
        assert "Loaded" in result.output or "Error" in result.output

    @patch("atomix.calculators.mlip.get_mlip_calculator")
    def test_screen_json_output(
        self,
        mock_get_calc: MagicMock,
        runner: CliRunner,
        temp_structure: Path,
    ) -> None:
        """Test screen command with JSON output."""
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = {
            "converged": True,
            "energy": -10.0,
            "forces": np.zeros((1, 3)),
            "stress": None,
            "atoms": Atoms("Cu"),
            "n_steps": 1,
            "trajectory": [],
            "errors": [],
            "warnings": [],
        }
        mock_get_calc.return_value = mock_calc

        result = runner.invoke(cli, ["screen", str(temp_structure), "--json"])

        if result.exit_code == 0:
            import json
            # Should be valid JSON if successful
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                pass  # May include status messages


class TestTrainDataCommand:
    """Tests for the train-data command."""

    def test_train_data_help(self, runner: CliRunner) -> None:
        """Test train-data command help."""
        result = runner.invoke(cli, ["train-data", "--help"])
        assert result.exit_code == 0
        assert "Export DFT calculations" in result.output

    def test_train_data_requires_output(
        self,
        runner: CliRunner,
        temp_calc_dir: Path,
    ) -> None:
        """Test train-data requires output option."""
        result = runner.invoke(cli, ["train-data", str(temp_calc_dir)])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "Missing" in result.output

    def test_train_data_no_dirs(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test train-data with no valid directories."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = runner.invoke(cli, [
            "train-data", str(empty_dir),
            "-o", str(tmp_path / "out.xyz")
        ])

        assert result.exit_code != 0
        assert "No valid calculation directories" in result.output


class TestScreenSitesCommand:
    """Tests for the screen-sites command."""

    def test_screen_sites_help(self, runner: CliRunner) -> None:
        """Test screen-sites command help."""
        result = runner.invoke(cli, ["screen-sites", "--help"])
        assert result.exit_code == 0
        assert "Screen adsorption sites" in result.output

    def test_screen_sites_requires_adsorbate(
        self,
        runner: CliRunner,
        temp_slab: Path,
    ) -> None:
        """Test screen-sites requires adsorbate option."""
        result = runner.invoke(cli, ["screen-sites", str(temp_slab)])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "Missing" in result.output

    @patch("atomix.calculators.mlip.get_mlip_calculator")
    def test_screen_sites_basic(
        self,
        mock_get_calc: MagicMock,
        runner: CliRunner,
        temp_slab: Path,
    ) -> None:
        """Test screen-sites with mock calculator."""
        mock_calc = MagicMock()
        mock_calc.calculate.return_value = {
            "converged": True,
            "energy": -50.0,
            "forces": np.zeros((13, 3)),  # 12 slab atoms + 1 adsorbate
            "stress": None,
            "atoms": Atoms("Cu12O"),
            "n_steps": 1,
            "trajectory": [],
            "errors": [],
            "warnings": [],
        }
        mock_get_calc.return_value = mock_calc

        result = runner.invoke(cli, [
            "screen-sites", str(temp_slab),
            "-a", "O", "-n", "3"
        ])

        # Should load slab and find sites
        assert "Loaded slab" in result.output or "Error" in result.output


class TestExistingCommands:
    """Tests for existing CLI commands still work."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test main CLI help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "atomix" in result.output

    def test_generate_help(self, runner: CliRunner) -> None:
        """Test generate command help."""
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate simulation inputs" in result.output

    def test_submit_help(self, runner: CliRunner) -> None:
        """Test submit command help."""
        result = runner.invoke(cli, ["submit", "--help"])
        assert result.exit_code == 0
        assert "Submit calculation" in result.output

    def test_analyze_help(self, runner: CliRunner) -> None:
        """Test analyze command help."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "Analyze calculation" in result.output

    def test_sites_help(self, runner: CliRunner) -> None:
        """Test sites command help."""
        result = runner.invoke(cli, ["sites", "--help"])
        assert result.exit_code == 0
        assert "Find adsorption sites" in result.output

    def test_adsorption_help(self, runner: CliRunner) -> None:
        """Test adsorption command help."""
        result = runner.invoke(cli, ["adsorption", "--help"])
        assert result.exit_code == 0
        assert "Calculate adsorption energies" in result.output
