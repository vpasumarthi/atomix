"""Tests for the supported segmented-VASP inspection command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from atomix.cli.main import cli


def _xdatcar(position: tuple[float, float, float]) -> str:
    return "\n".join(
        [
            "CLI segmented trajectory",
            "1.0",
            "10.0 0.0 0.0",
            "0.0 10.0 0.0",
            "0.0 0.0 10.0",
            "H",
            "1",
            "Direct configuration=      1",
            "{:.8f} {:.8f} {:.8f}".format(*position),
            "",
        ]
    )


def _segment(root: Path, name: str, position: tuple[float, float, float]) -> None:
    segment = root / name
    segment.mkdir()
    (segment / "XDATCAR").write_text(_xdatcar(position), encoding="utf-8")
    (segment / "OUTCAR").write_text(
        " POTIM = 1.0 time-step for ionic-motion\n"
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    (segment / "REPORT").write_text("REPORT\n", encoding="utf-8")


def test_inspect_vasp_human_summary(tmp_path: Path) -> None:
    _segment(tmp_path, "seg01", (0.1, 0.0, 0.0))
    _segment(tmp_path, "seg02", (0.2, 0.0, 0.0))

    result = CliRunner().invoke(cli, ["inspect-vasp", str(tmp_path)])

    assert result.exit_code == 0
    assert "Segments: 2 (1, 2)" in result.output
    assert "Frames: 2 (XDATCAR: 2)" in result.output
    assert "Cumulative time: 1 to 2 fs" in result.output
    assert "Diagnostics: 0 warnings, 0 errors" in result.output


def test_inspect_vasp_json_includes_provenance(tmp_path: Path) -> None:
    _segment(tmp_path, "seg01", (0.1, 0.0, 0.0))

    result = CliRunner().invoke(cli, ["inspect-vasp", str(tmp_path), "--json"])

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["segment_count"] == 1
    assert summary["segments"][0]["number"] == 1
    assert summary["frame_sources"] == {"XDATCAR": 1}
    assert summary["time_range_fs"] == [1.0, 1.0]


def test_inspect_vasp_exits_nonzero_for_ambiguous_segments(tmp_path: Path) -> None:
    _segment(tmp_path, "seg1", (0.1, 0.0, 0.0))
    _segment(tmp_path, "seg01", (0.2, 0.0, 0.0))

    result = CliRunner().invoke(cli, ["inspect-vasp", str(tmp_path), "--json"])

    assert result.exit_code == 1
    summary = json.loads(result.output)
    assert summary["error_count"] == 1
    assert summary["diagnostics"][0]["code"] == "duplicate_segment_number"
