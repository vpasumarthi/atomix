"""Tests for VASPCalculator output parsing, focused on the OUTCAR fallback.

When vasprun.xml is missing or unparseable (e.g. long AIMD runs where only
OUTCAR/XDATCAR are retained from the cluster), convergence must be inferred
from the OUTCAR. pymatgen's ``Outcar`` has no ``converged`` attribute, so
completion is read from ``run_stats``.

Regression guard: the mocked Outcar uses ``spec`` that *omits* ``converged``.
If a future change reaches for ``outcar.converged`` again, that raises
AttributeError, gets swallowed by the surrounding try/except, and these
assertions fail — which is exactly what we want.
"""

from unittest.mock import MagicMock, patch

from atomix.calculators.vasp import VASPCalculator


def _fake_outcar(run_stats, final_energy=-123.45):
    # spec excludes "converged" deliberately (it does not exist on pymatgen's
    # Outcar); accessing it on this mock raises AttributeError.
    outcar = MagicMock(spec=["run_stats", "final_energy"])
    outcar.run_stats = run_stats
    outcar.final_energy = final_energy
    return outcar


class TestOutcarFallback:
    @patch("atomix.calculators.vasp.Outcar")
    def test_is_converged_true_when_run_completed(self, mock_outcar_cls, tmp_path):
        (tmp_path / "OUTCAR").write_text("")  # no vasprun.xml present
        mock_outcar_cls.return_value = _fake_outcar({"Total CPU time used (sec)": 12.3})

        assert VASPCalculator(directory=tmp_path).is_converged() is True

    @patch("atomix.calculators.vasp.Outcar")
    def test_is_converged_false_when_run_incomplete(self, mock_outcar_cls, tmp_path):
        (tmp_path / "OUTCAR").write_text("")
        mock_outcar_cls.return_value = _fake_outcar({})  # empty run_stats

        assert VASPCalculator(directory=tmp_path).is_converged() is False

    @patch("atomix.calculators.vasp.Outcar")
    def test_read_outputs_marks_converged_from_outcar(self, mock_outcar_cls, tmp_path):
        (tmp_path / "OUTCAR").write_text("")
        mock_outcar_cls.return_value = _fake_outcar({"Total CPU time used (sec)": 12.3})

        results = VASPCalculator(directory=tmp_path).read_outputs()

        assert results["converged"] is True
        # the old bug surfaced as a swallowed "...has no attribute 'converged'"
        # warning; there should be no OUTCAR-parse warning now.
        assert not any("converged" in w for w in results["warnings"])
        assert results["energy"] == -123.45
