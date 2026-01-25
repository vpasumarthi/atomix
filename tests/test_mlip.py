"""Tests for MLIP calculators and screening workflows."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule

from atomix.calculators.mlip import (
    MACECalculator,
    MLIPCalculator,
    NequIPCalculator,
    get_mlip_calculator,
)
from atomix.core.active_learning import (
    ActiveLearningSelector,
    TrainingDataExporter,
    TrainingPoint,
    UncertaintyEstimator,
)
from atomix.core.screening import (
    AdsorptionScreening,
    ScreeningConfig,
    ScreeningResult,
    ScreeningWorkflow,
)


class MockASECalculator:
    """Mock ASE calculator for testing."""

    def __init__(self, energy: float = -10.0, forces: np.ndarray | None = None) -> None:
        self._energy = energy
        self._forces = forces
        self.implemented_properties = ["energy", "forces"]

    def get_potential_energy(self, atoms: Atoms = None) -> float:
        return self._energy

    def get_forces(self, atoms: Atoms = None) -> np.ndarray:
        if self._forces is not None:
            return self._forces
        n_atoms = len(atoms) if atoms else 4
        return np.random.randn(n_atoms, 3) * 0.1


class MockMLIPCalculator(MLIPCalculator):
    """Mock MLIP calculator for testing without actual MACE."""

    def __init__(
        self,
        energy: float = -10.0,
        forces: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._mock_energy = energy
        self._mock_forces = forces

    def get_calculator(self):
        return MockASECalculator(self._mock_energy, self._mock_forces)

    def calculate(self, atoms: Atoms) -> dict:
        n_atoms = len(atoms)
        forces = self._mock_forces if self._mock_forces is not None else np.random.randn(n_atoms, 3) * 0.1
        return {
            "converged": True,
            "energy": self._mock_energy,
            "forces": forces,
            "stress": None,
            "atoms": atoms.copy(),
            "n_steps": 1,
            "trajectory": [atoms.copy()],
            "errors": [],
            "warnings": [],
        }


class TestMACECalculator:
    """Tests for MACECalculator class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        calc = MACECalculator()
        assert calc.model_path is None
        assert calc.model_name == "medium"
        assert calc.device == "cpu"
        assert calc.dispersion is False

    def test_init_custom_model(self) -> None:
        """Test initialization with custom model path."""
        calc = MACECalculator(model_path="/path/to/model.model")
        assert calc.model_path == Path("/path/to/model.model")

    def test_init_foundation_model(self) -> None:
        """Test initialization with foundation model."""
        calc = MACECalculator(model="large", device="cuda", dispersion=True)
        assert calc.model_name == "large"
        assert calc.device == "cuda"
        assert calc.dispersion is True

    def test_mace_not_available(self) -> None:
        """Test error when mace-torch is not installed."""
        calc = MACECalculator()
        calc._mace_available = False

        with pytest.raises(ImportError, match="mace-torch is required"):
            calc.get_calculator()

    @patch("atomix.calculators.mlip.MACECalculator._check_mace_available")
    @patch("atomix.calculators.mlip.MACECalculator._load_calculator")
    def test_get_calculator_caching(self, mock_load, mock_check) -> None:
        """Test that calculator is cached after first load."""
        mock_check.return_value = True
        mock_calc = MockASECalculator()
        mock_load.return_value = mock_calc

        calc = MACECalculator()
        result1 = calc.get_calculator()
        result2 = calc.get_calculator()

        # Should only load once
        assert mock_load.call_count == 1
        assert result1 is result2


class TestNequIPCalculator:
    """Tests for NequIPCalculator class."""

    def test_init_requires_model_path(self) -> None:
        """Test that model_path is required for get_calculator."""
        calc = NequIPCalculator(model_path="/path/to/model.pth")
        assert calc.model_path == Path("/path/to/model.pth")

    def test_nequip_not_available(self) -> None:
        """Test error when nequip is not installed."""
        calc = NequIPCalculator(model_path="/path/to/model.pth")
        calc._nequip_available = False

        with pytest.raises(ImportError, match="nequip is required"):
            calc.get_calculator()


class TestGetMLIPCalculator:
    """Tests for get_mlip_calculator factory function."""

    def test_get_mace(self) -> None:
        """Test getting MACE calculator."""
        calc = get_mlip_calculator("mace", model="small")
        assert isinstance(calc, MACECalculator)
        assert calc.model_name == "small"

    def test_get_nequip(self) -> None:
        """Test getting NequIP calculator."""
        calc = get_mlip_calculator("nequip", model_path="/path/model.pth")
        assert isinstance(calc, NequIPCalculator)

    def test_nequip_requires_path(self) -> None:
        """Test that NequIP requires model_path."""
        with pytest.raises(ValueError, match="requires a model_path"):
            get_mlip_calculator("nequip")

    def test_unknown_calculator(self) -> None:
        """Test error for unknown calculator."""
        with pytest.raises(ValueError, match="Unknown MLIP calculator"):
            get_mlip_calculator("unknown")


class TestScreeningResult:
    """Tests for ScreeningResult dataclass."""

    def test_energy_error(self) -> None:
        """Test energy error calculation."""
        atoms = Atoms("Cu4", positions=np.zeros((4, 3)))
        result = ScreeningResult(
            atoms=atoms,
            mlip_energy=-10.0,
            dft_energy=-10.5,
        )
        assert abs(result.energy_error - 0.5) < 1e-10

    def test_energy_error_none(self) -> None:
        """Test energy error when DFT not available."""
        atoms = Atoms("Cu4", positions=np.zeros((4, 3)))
        result = ScreeningResult(atoms=atoms, mlip_energy=-10.0)
        assert result.energy_error is None

    def test_force_mae(self) -> None:
        """Test force MAE calculation."""
        atoms = Atoms("Cu4", positions=np.zeros((4, 3)))
        mlip_forces = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        dft_forces = np.array([[1.1, 0, 0], [0, 0.9, 0], [0, 0, 1.1], [0, 0, 0]])

        result = ScreeningResult(
            atoms=atoms,
            mlip_forces=mlip_forces,
            dft_forces=dft_forces,
        )
        # MAE should be around 0.1 * 3 / 12 = 0.025
        assert result.force_mae is not None
        assert result.force_mae < 0.1


class TestScreeningWorkflow:
    """Tests for ScreeningWorkflow class."""

    @pytest.fixture
    def mock_calculator(self) -> MockMLIPCalculator:
        """Create mock MLIP calculator."""
        return MockMLIPCalculator(energy=-10.0)

    @pytest.fixture
    def candidate_structures(self) -> list[Atoms]:
        """Create candidate structures for screening."""
        structures = []
        for i in range(10):
            atoms = Atoms(
                "Cu4",
                positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
                cell=[4, 4, 4],
                pbc=True,
            )
            # Slightly different positions
            atoms.positions += np.random.randn(4, 3) * 0.1
            structures.append(atoms)
        return structures

    def test_screen_basic(
        self,
        mock_calculator: MockMLIPCalculator,
        candidate_structures: list[Atoms],
    ) -> None:
        """Test basic screening workflow."""
        workflow = ScreeningWorkflow(mock_calculator)
        results = workflow.screen(candidate_structures)

        assert len(results) == 10
        assert all(r.mlip_energy is not None for r in results)
        assert all(r.rank > 0 for r in results)

    def test_screen_with_metadata(self, mock_calculator: MockMLIPCalculator) -> None:
        """Test screening with metadata."""
        structures = [Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]]) for _ in range(3)]
        metadata = [{"site": "top"}, {"site": "bridge"}, {"site": "hollow"}]

        workflow = ScreeningWorkflow(mock_calculator)
        results = workflow.screen(structures, metadata)

        assert results[0].metadata["site"] in ["top", "bridge", "hollow"]

    def test_screen_selection_top_n(self, mock_calculator: MockMLIPCalculator) -> None:
        """Test that top N structures are selected."""
        # Create calculator with varying energies
        structures = []
        for i in range(10):
            atoms = Atoms("Cu2", positions=[[0, 0, 0], [2 + i * 0.1, 0, 0]])
            structures.append(atoms)

        config = ScreeningConfig(top_n=3)
        workflow = ScreeningWorkflow(mock_calculator, config)
        workflow.screen(structures)

        selected = workflow.get_selected()
        assert len(selected) == 3

    def test_screen_selection_energy_window(self) -> None:
        """Test selection by energy window."""
        # Create calculator returning different energies
        class VariableEnergyCalc(MockMLIPCalculator):
            def __init__(self):
                super().__init__()
                self._call_count = 0

            def calculate(self, atoms):
                result = super().calculate(atoms)
                result["energy"] = -10.0 + self._call_count * 0.2
                self._call_count += 1
                return result

        structures = [Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]]) for _ in range(10)]
        config = ScreeningConfig(top_n=100, energy_window=0.5)

        workflow = ScreeningWorkflow(VariableEnergyCalc(), config)
        workflow.screen(structures)

        # Only structures within 0.5 eV of minimum should be selected
        selected = workflow.get_selected()
        assert len(selected) <= 3  # 0, 0.2, 0.4 are within 0.5 of 0

    def test_get_top_n(
        self,
        mock_calculator: MockMLIPCalculator,
        candidate_structures: list[Atoms],
    ) -> None:
        """Test get_top_n method."""
        workflow = ScreeningWorkflow(mock_calculator)
        workflow.screen(candidate_structures)

        top3 = workflow.get_top_n(3)
        assert len(top3) == 3
        # Should be sorted by energy
        energies = [r.mlip_energy for r in top3]
        assert energies == sorted(energies)

    def test_add_dft_result(
        self,
        mock_calculator: MockMLIPCalculator,
        candidate_structures: list[Atoms],
    ) -> None:
        """Test adding DFT validation results."""
        workflow = ScreeningWorkflow(mock_calculator)
        workflow.screen(candidate_structures)

        workflow.add_dft_result(0, dft_energy=-10.5)
        assert workflow.results[0].dft_energy == -10.5

    def test_validation_statistics(
        self,
        mock_calculator: MockMLIPCalculator,
    ) -> None:
        """Test validation statistics calculation."""
        structures = [Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]]) for _ in range(5)]

        workflow = ScreeningWorkflow(mock_calculator)
        workflow.screen(structures)

        # Add some DFT results
        for i in range(3):
            workflow.add_dft_result(i, dft_energy=-10.0 + i * 0.1)

        stats = workflow.get_validation_statistics()
        assert stats["n_validated"] == 3
        assert "energy_mae" in stats
        assert "energy_max_error" in stats


class TestAdsorptionScreening:
    """Tests for AdsorptionScreening class."""

    @pytest.fixture
    def mock_calculator(self) -> MockMLIPCalculator:
        return MockMLIPCalculator(energy=-50.0)

    @pytest.fixture
    def slab_and_adsorbate(self) -> tuple[Atoms, Atoms]:
        """Create slab and adsorbate for testing."""
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        adsorbate = Atoms("O", positions=[[0, 0, 0]])
        return slab, adsorbate

    def test_screen_sites(
        self,
        mock_calculator: MockMLIPCalculator,
        slab_and_adsorbate: tuple[Atoms, Atoms],
    ) -> None:
        """Test screening adsorption sites."""
        slab, adsorbate = slab_and_adsorbate

        sites = [
            (0, 0, slab.positions[:, 2].max()),
            (1, 1, slab.positions[:, 2].max()),
            (2, 2, slab.positions[:, 2].max()),
        ]

        screening = AdsorptionScreening(mock_calculator)
        results = screening.screen_sites(slab, adsorbate, sites)

        assert len(results) == 3
        # Each result should have the adsorbate added
        for r in results:
            assert len(r.atoms) == len(slab) + len(adsorbate)

    def test_screen_coverages(
        self,
        mock_calculator: MockMLIPCalculator,
        slab_and_adsorbate: tuple[Atoms, Atoms],
    ) -> None:
        """Test screening different coverage configurations."""
        slab, adsorbate = slab_and_adsorbate

        z_top = slab.positions[:, 2].max()
        configs = [
            [(0, 0, z_top)],  # 1 adsorbate
            [(0, 0, z_top), (2, 2, z_top)],  # 2 adsorbates
        ]

        screening = AdsorptionScreening(mock_calculator)
        results = screening.screen_coverages(slab, adsorbate, configs)

        assert len(results) == 2
        assert results[0].metadata["n_adsorbates"] == 1
        assert results[1].metadata["n_adsorbates"] == 2


class TestTrainingDataExporter:
    """Tests for TrainingDataExporter class."""

    @pytest.fixture
    def sample_training_data(self) -> list[TrainingPoint]:
        """Create sample training data."""
        data = []
        for i in range(5):
            atoms = Atoms("Cu4", positions=np.random.randn(4, 3))
            point = TrainingPoint(
                atoms=atoms,
                energy=-10.0 - i,
                forces=np.random.randn(4, 3),
                source="test",
                metadata={"index": i},
            )
            data.append(point)
        return data

    def test_add_point(self) -> None:
        """Test adding training points."""
        exporter = TrainingDataExporter()
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]])
        point = TrainingPoint(atoms=atoms, energy=-5.0, forces=np.zeros((2, 3)))
        exporter.add_point(point)
        assert len(exporter) == 1

    def test_to_extxyz(
        self,
        sample_training_data: list[TrainingPoint],
        tmp_path: Path,
    ) -> None:
        """Test export to extended XYZ format."""
        exporter = TrainingDataExporter(sample_training_data)
        output_path = exporter.to_extxyz(tmp_path / "train.xyz")

        assert output_path.exists()
        # Read back and verify
        from ase.io import read
        atoms_list = read(str(output_path), index=":")
        assert len(atoms_list) == 5
        assert "REF_energy" in atoms_list[0].info

    def test_split_train_val(self, sample_training_data: list[TrainingPoint]) -> None:
        """Test train/val split."""
        exporter = TrainingDataExporter(sample_training_data)
        train, val = exporter.split_train_val(val_fraction=0.2)

        assert len(train) + len(val) == len(sample_training_data)
        assert len(val) == 1  # 20% of 5

    def test_to_ase_db(
        self,
        sample_training_data: list[TrainingPoint],
        tmp_path: Path,
    ) -> None:
        """Test export to ASE database."""
        exporter = TrainingDataExporter(sample_training_data)
        db_path = exporter.to_ase_db(tmp_path / "train.db")

        assert db_path.exists()
        from ase.db import connect
        db = connect(str(db_path))
        assert len(db) == 5


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator class."""

    @pytest.fixture
    def ensemble_calculators(self) -> list[MockMLIPCalculator]:
        """Create ensemble of mock calculators with varying predictions."""
        return [
            MockMLIPCalculator(energy=-10.0 + i * 0.1)
            for i in range(3)
        ]

    def test_init_requires_multiple(self) -> None:
        """Test that ensemble requires at least 2 calculators."""
        with pytest.raises(ValueError, match="at least 2"):
            UncertaintyEstimator([MockMLIPCalculator()])

    def test_estimate_basic(
        self,
        ensemble_calculators: list[MockMLIPCalculator],
    ) -> None:
        """Test basic uncertainty estimation."""
        estimator = UncertaintyEstimator(ensemble_calculators)
        atoms = Atoms("Cu4", positions=np.zeros((4, 3)))

        result = estimator.estimate(atoms)

        assert "energy_mean" in result
        assert "energy_std" in result
        assert result["energy_std"] > 0  # Should have some variance

    def test_estimate_batch(
        self,
        ensemble_calculators: list[MockMLIPCalculator],
    ) -> None:
        """Test batch uncertainty estimation."""
        estimator = UncertaintyEstimator(ensemble_calculators)
        structures = [
            Atoms("Cu4", positions=np.random.randn(4, 3))
            for _ in range(5)
        ]

        results = estimator.estimate_batch(structures)
        assert len(results) == 5


class TestActiveLearningSelector:
    """Tests for ActiveLearningSelector class."""

    @pytest.fixture
    def selector_with_estimator(self) -> ActiveLearningSelector:
        """Create selector with uncertainty estimator."""
        calcs = [MockMLIPCalculator(energy=-10.0 + i * 0.2) for i in range(3)]
        estimator = UncertaintyEstimator(calcs)
        return ActiveLearningSelector(estimator)

    @pytest.fixture
    def candidate_structures(self) -> list[Atoms]:
        """Create candidate structures."""
        return [
            Atoms("Cu4", positions=np.random.randn(4, 3))
            for _ in range(20)
        ]

    def test_select_by_uncertainty(
        self,
        selector_with_estimator: ActiveLearningSelector,
        candidate_structures: list[Atoms],
    ) -> None:
        """Test uncertainty-based selection."""
        selected = selector_with_estimator.select_by_uncertainty(
            candidate_structures, n=5
        )

        assert len(selected) == 5
        # Should return tuples of (atoms, uncertainty)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in selected)

    def test_select_diverse_composition(
        self,
        candidate_structures: list[Atoms],
    ) -> None:
        """Test diversity-based selection."""
        selector = ActiveLearningSelector()
        selected = selector.select_diverse(
            candidate_structures, n=5, descriptor="composition"
        )

        assert len(selected) == 5

    def test_select_from_trajectory(
        self,
        selector_with_estimator: ActiveLearningSelector,
    ) -> None:
        """Test selection from trajectory."""
        trajectory = [
            Atoms("Cu4", positions=np.random.randn(4, 3))
            for _ in range(50)
        ]

        selected = selector_with_estimator.select_from_trajectory(
            trajectory, n=10, interval=5
        )

        assert len(selected) == 10

    def test_select_without_estimator(
        self,
        candidate_structures: list[Atoms],
    ) -> None:
        """Test that uncertainty selection requires estimator."""
        selector = ActiveLearningSelector()

        with pytest.raises(ValueError, match="UncertaintyEstimator required"):
            selector.select_by_uncertainty(candidate_structures, n=5)


class TestConfigMLIP:
    """Tests for MLIP configuration."""

    def test_default_mlip_config(self) -> None:
        """Test default MLIP configuration values."""
        from atomix.core.config import Config

        config = Config()

        assert config.get("mlip", "default_calculator") == "mace"
        assert config.get("mlip", "mace", "model") == "medium"
        assert config.get("mlip", "mace", "device") == "cpu"
        assert config.get("mlip", "mace", "dispersion") is False

    def test_set_mlip_config(self) -> None:
        """Test setting MLIP configuration."""
        from atomix.core.config import Config

        config = Config()
        config.set("mlip", "mace", "device", value="cuda")

        assert config.get("mlip", "mace", "device") == "cuda"
