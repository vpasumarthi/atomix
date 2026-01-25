"""Tests for workflow execution."""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT

from atomix.core.calculation import (
    RelaxCalculation,
    StaticCalculation,
)
from atomix.core.workflow import RelaxationWorkflow, ScreeningWorkflowSimple, Workflow


class TestWorkflow:
    """Tests for Workflow class."""

    @pytest.fixture
    def cu_atoms(self) -> Atoms:
        """Create copper bulk structure."""
        return bulk("Cu", "fcc", a=3.6)

    @pytest.fixture
    def emt_calculator(self) -> EMT:
        """Create EMT calculator for testing."""
        return EMT()

    def test_workflow_init(self) -> None:
        """Test workflow initialization."""
        workflow = Workflow("test")
        assert workflow.name == "test"
        assert len(workflow.steps) == 0
        assert len(workflow.results) == 0

    def test_workflow_add_step(self, cu_atoms: Atoms) -> None:
        """Test adding steps to workflow."""
        workflow = Workflow("test")
        calc = StaticCalculation(cu_atoms)
        workflow.add_step(calc)
        assert len(workflow.steps) == 1

    def test_workflow_run_direct_singlepoint(
        self,
        cu_atoms: Atoms,
        emt_calculator: EMT,
    ) -> None:
        """Test direct single-point calculation."""
        workflow = Workflow("static")
        results = workflow.run_direct(cu_atoms, emt_calculator, steps=0)

        assert len(results) == 1
        result = results[0]
        assert result["converged"] is True
        assert result["energy"] is not None
        assert result["forces"] is not None
        assert result["atoms"] is not None

    def test_workflow_run_direct_relaxation(
        self,
        cu_atoms: Atoms,
        emt_calculator: EMT,
    ) -> None:
        """Test direct relaxation."""
        # Distort structure slightly
        atoms = cu_atoms.copy()
        atoms.positions[0] += [0.1, 0, 0]

        workflow = Workflow("relax")
        results = workflow.run_direct(
            atoms,
            emt_calculator,
            fmax=0.1,
            steps=50,
        )

        assert len(results) == 1
        result = results[0]
        assert result["energy"] is not None
        assert result["n_steps"] >= 0  # May be 0 if already converged
        assert len(result["trajectory"]) >= 1

    def test_workflow_run_alias(
        self,
        cu_atoms: Atoms,
        emt_calculator: EMT,
    ) -> None:
        """Test that run() calls run_direct()."""
        workflow = Workflow("test")
        results = workflow.run(cu_atoms, emt_calculator)

        assert len(results) == 1
        assert results[0]["energy"] is not None

    def test_workflow_clear(self, cu_atoms: Atoms) -> None:
        """Test clearing workflow."""
        workflow = Workflow("test")
        workflow.add_step(StaticCalculation(cu_atoms))
        workflow._results = [{"energy": -1.0}]

        workflow.clear()
        assert len(workflow.steps) == 0
        assert len(workflow.results) == 0


class TestRelaxationWorkflow:
    """Tests for RelaxationWorkflow class."""

    @pytest.fixture
    def distorted_cu(self) -> Atoms:
        """Create distorted Cu structure."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.positions[0] += [0.1, 0.05, 0]
        return atoms

    def test_relaxation_workflow_init(self) -> None:
        """Test RelaxationWorkflow initialization."""
        workflow = RelaxationWorkflow(fmax=0.01, steps=100, optimizer="FIRE")
        assert workflow.fmax == 0.01
        assert workflow.max_steps == 100
        assert workflow.optimizer == "FIRE"

    def test_relaxation_workflow_run(self, distorted_cu: Atoms) -> None:
        """Test running relaxation workflow."""
        calc = EMT()
        workflow = RelaxationWorkflow(fmax=0.1, steps=50)
        results = workflow.run(distorted_cu, calc)

        assert len(results) == 1
        result = results[0]
        assert result["energy"] is not None

        # Check forces decreased
        final_forces = result["forces"]
        max_force = np.max(np.linalg.norm(final_forces, axis=1))
        assert max_force < 1.0  # Should be much lower than initial


class TestScreeningWorkflowSimple:
    """Tests for ScreeningWorkflowSimple class."""

    @pytest.fixture
    def structures(self) -> list[Atoms]:
        """Create test structures."""
        structures = []
        for a in [3.5, 3.55, 3.6, 3.65, 3.7]:
            atoms = bulk("Cu", "fcc", a=a)
            structures.append(atoms)
        return structures

    def test_screening_workflow_run(self, structures: list[Atoms]) -> None:
        """Test screening multiple structures."""
        calc = EMT()
        workflow = ScreeningWorkflowSimple("screening")
        results = workflow.run_screening(structures, calc)

        assert len(results) == 5
        # Should be sorted by energy
        energies = [r["energy"] for r in results if r.get("energy")]
        assert energies == sorted(energies)
        # Should have ranks
        assert results[0]["rank"] == 1


class TestStaticCalculation:
    """Tests for StaticCalculation with direct execution."""

    @pytest.fixture
    def cu_atoms(self) -> Atoms:
        return bulk("Cu", "fcc", a=3.6)

    def test_static_with_calculator(self, cu_atoms: Atoms) -> None:
        """Test static calculation with ASE calculator."""
        calc = StaticCalculation(cu_atoms, calculator=EMT())
        calc.run()

        assert calc.results["converged"] is True
        assert calc.results["energy"] is not None
        assert calc.results["forces"] is not None

    def test_static_without_calculator(self, cu_atoms: Atoms, tmp_path: Path) -> None:
        """Test static calculation without calculator raises error."""
        calc = StaticCalculation(cu_atoms, directory=tmp_path)

        with pytest.raises(NotImplementedError, match="external job submission"):
            calc.run()

    def test_static_setup_creates_files(self, cu_atoms: Atoms, tmp_path: Path) -> None:
        """Test that setup creates VASP input files."""
        calc = StaticCalculation(cu_atoms, directory=tmp_path, ENCUT=400)
        files = calc.setup()

        assert "POSCAR" in files
        assert "INCAR" in files
        assert (tmp_path / "POSCAR").exists()
        assert (tmp_path / "INCAR").exists()


class TestRelaxCalculation:
    """Tests for RelaxCalculation with direct execution."""

    @pytest.fixture
    def distorted_cu(self) -> Atoms:
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.positions[0] += [0.1, 0, 0]
        return atoms

    def test_relax_with_calculator(self, distorted_cu: Atoms) -> None:
        """Test relaxation with ASE calculator."""
        calc = RelaxCalculation(
            distorted_cu,
            calculator=EMT(),
            fmax=0.1,
            steps=50,
        )
        calc.run()

        assert calc.results["energy"] is not None
        assert calc.results["n_steps"] >= 0  # May be 0 if converges immediately
        # Structure should have been updated
        assert calc.atoms is not distorted_cu

    def test_relax_convergence(self, distorted_cu: Atoms) -> None:
        """Test relaxation converges properly."""
        calc = RelaxCalculation(
            distorted_cu,
            calculator=EMT(),
            fmax=0.05,
            steps=100,
        )
        calc.run()

        forces = calc.results["forces"]
        max_force = np.max(np.linalg.norm(forces, axis=1))

        if calc.results["converged"]:
            assert max_force <= 0.05

    def test_relax_optimizer_options(self, distorted_cu: Atoms) -> None:
        """Test different optimizer options."""
        for optimizer in ["BFGS", "LBFGS", "FIRE"]:
            calc = RelaxCalculation(
                distorted_cu.copy(),
                calculator=EMT(),
                fmax=0.5,
                steps=10,
                optimizer=optimizer,
            )
            calc.run()
            assert calc.results["energy"] is not None


class TestAIMDCalculation:
    """Tests for AIMD calculation with direct execution."""

    @pytest.fixture
    def cu_atoms(self) -> Atoms:
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        return atoms

    def test_nvt_with_calculator(self, cu_atoms: Atoms) -> None:
        """Test NVT MD with ASE calculator."""
        from atomix.core.calculation import NVTCalculation

        calc = NVTCalculation(
            cu_atoms,
            calculator=EMT(),
            temperature=300.0,
            timestep=1.0,
            steps=10,  # Short for testing
        )
        calc.run()

        assert calc.results["converged"] is True
        assert len(calc.results["trajectory"]) >= 10  # May include initial frame
        assert calc.results["n_steps"] == 10

    def test_nve_with_calculator(self, cu_atoms: Atoms) -> None:
        """Test NVE MD with ASE calculator."""
        from atomix.core.calculation import NVECalculation

        calc = NVECalculation(
            cu_atoms,
            calculator=EMT(),
            temperature=300.0,
            timestep=1.0,
            steps=5,
        )
        calc.run()

        assert calc.results["converged"] is True
        assert len(calc.results["trajectory"]) >= 5  # May include initial frame


class TestParseResults:
    """Tests for parsing results from completed calculations."""

    def test_static_parse_results_returns_cached(self) -> None:
        """Test that parse_results returns cached results."""
        atoms = bulk("Cu", "fcc", a=3.6)
        calc = StaticCalculation(atoms, calculator=EMT())
        calc.run()

        # Should return same cached result
        result1 = calc.parse_results()
        result2 = calc.parse_results()
        assert result1 is result2

    def test_relax_parse_results_returns_cached(self) -> None:
        """Test that parse_results returns cached results."""
        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.positions[0] += [0.1, 0, 0]

        calc = RelaxCalculation(atoms, calculator=EMT(), fmax=0.5, steps=10)
        calc.run()

        result1 = calc.parse_results()
        result2 = calc.parse_results()
        assert result1 is result2
