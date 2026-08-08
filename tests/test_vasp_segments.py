"""Deterministic tests for segmented VASP output reading."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from atomix.calculators import VASPSegmentReader
from atomix.calculators import vasp_segments as vasp_segments_module


def _xdatcar(
    positions: list[tuple[float, float, float]],
    configuration_numbers: list[int] | None = None,
) -> str:
    numbers = configuration_numbers or list(range(1, len(positions) + 1))
    header = [
        "segmented test trajectory",
        "1.0",
        "10.0 0.0 0.0",
        "0.0 10.0 0.0",
        "0.0 0.0 10.0",
        "H",
        "1",
    ]
    lines = list(header)
    for number, position in zip(numbers, positions, strict=True):
        lines.append(f"Direct configuration= {number:6d}")
        lines.append("{:.8f} {:.8f} {:.8f}".format(*position))
    return "\n".join(lines) + "\n"


def _complete_outcar(label: str) -> str:
    return f"{label}\nGeneral timing and accounting informations for this job:\n"


def _write_segment(
    root: Path,
    name: str,
    positions: list[tuple[float, float, float]],
    *,
    configuration_numbers: list[int] | None = None,
) -> Path:
    segment = root / name
    segment.mkdir()
    (segment / "XDATCAR").write_text(_xdatcar(positions, configuration_numbers), encoding="utf-8")
    (segment / "OUTCAR").write_text(_complete_outcar(name), encoding="utf-8")
    (segment / "REPORT").write_text(f"REPORT for {name}\n", encoding="utf-8")
    return segment


@pytest.fixture
def complete_segmented_calculation(tmp_path: Path) -> Path:
    """Two complete segments with a repeated boundary and root outputs."""
    (tmp_path / "INCAR").write_text("POTIM = 2.0\n", encoding="utf-8")
    (tmp_path / "XDATCAR").write_text(_xdatcar([(0.9, 0.0, 0.0)]), encoding="utf-8")
    (tmp_path / "OUTCAR").write_text(_complete_outcar("root"), encoding="utf-8")
    (tmp_path / "REPORT").write_text("root REPORT\n", encoding="utf-8")
    _write_segment(tmp_path, "seg01", [(0.1, 0.0, 0.0), (0.2, 0.0, 0.0)])
    _write_segment(tmp_path, "seg02", [(0.2, 0.0, 0.0), (0.3, 0.0, 0.0)])
    return tmp_path


def test_complete_segments_preserve_provenance_and_ignore_root_data(
    complete_segmented_calculation: Path,
) -> None:
    before = {
        path.relative_to(complete_segmented_calculation): path.read_bytes()
        for path in complete_segmented_calculation.rglob("*")
        if path.is_file()
    }

    result = VASPSegmentReader(complete_segmented_calculation).read()

    assert [segment.number for segment in result.segments] == [1, 2]
    assert len(result.frames) == 3
    assert [
        (
            frame.segment_number,
            frame.local_frame_index,
            frame.global_frame_index,
            frame.source_directory.name,
            frame.source_kind,
        )
        for frame in result.frames
    ] == [
        (1, 0, 0, "seg01", "XDATCAR"),
        (1, 1, 1, "seg01", "XDATCAR"),
        (2, 1, 2, "seg02", "XDATCAR"),
    ]
    assert [frame.configuration_number for frame in result.frames] == [1, 2, 2]
    assert [frame.timestep_fs for frame in result.frames] == [2.0, 2.0, 2.0]
    assert [frame.local_time_fs for frame in result.frames] == [2.0, 4.0, 4.0]
    assert [frame.time_fs for frame in result.frames] == [2.0, 4.0, 6.0]
    assert [record.source_directory.name for record in result.outcar_data] == [
        "seg01",
        "seg02",
    ]
    assert [record.source_directory.name for record in result.report_data] == [
        "seg01",
        "seg02",
    ]
    assert [record.is_complete for record in result.outcar_data] == [True, True]
    assert [record.is_complete for record in result.report_data] == [None, None]
    assert all(record.line_count > 0 for record in result.outcar_data + result.report_data)
    assert all(record.size_bytes > 0 for record in result.outcar_data + result.report_data)
    assert {warning.code for warning in result.warnings} >= {
        "duplicate_boundary_frame",
        "root_data_ignored",
    }
    assert not result.has_errors

    after = {
        path.relative_to(complete_segmented_calculation): path.read_bytes()
        for path in complete_segmented_calculation.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_mixed_padding_and_100_sort_numerically(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("POTIM = 1.0\n", encoding="utf-8")
    _write_segment(tmp_path, "seg100", [(0.3, 0.0, 0.0)])
    _write_segment(tmp_path, "seg1", [(0.1, 0.0, 0.0)])
    _write_segment(tmp_path, "seg02", [(0.2, 0.0, 0.0)])

    result = VASPSegmentReader(tmp_path).read()

    assert [segment.number for segment in result.segments] == [1, 2, 100]
    assert [frame.segment_number for frame in result.frames] == [1, 2, 100]
    assert result.frames[-1].time_fs is None
    assert any(warning.code == "segment_sequence_gap" for warning in result.warnings)


def test_root_outputs_are_used_only_when_no_segment_directories_exist(
    tmp_path: Path,
) -> None:
    (tmp_path / "INCAR").write_text("POTIM = 1.0\n", encoding="utf-8")
    (tmp_path / "XDATCAR").write_text(_xdatcar([(0.1, 0.0, 0.0)]), encoding="utf-8")
    (tmp_path / "OUTCAR").write_text(_complete_outcar("root"), encoding="utf-8")
    (tmp_path / "REPORT").write_text("root REPORT\n", encoding="utf-8")

    result = VASPSegmentReader(tmp_path).read()

    assert [(segment.number, segment.directory) for segment in result.segments] == [
        (None, tmp_path)
    ]
    assert result.frames[0].segment_number is None
    assert result.frames[0].source_directory == tmp_path
    assert not any(warning.code == "root_data_ignored" for warning in result.warnings)


@pytest.mark.parametrize(
    "boundary_tolerance",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_boundary_tolerance_must_be_finite_and_positive(
    boundary_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        VASPSegmentReader(boundary_tolerance=boundary_tolerance)


def test_duplicate_segment_numbers_are_reported_and_excluded(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("POTIM = 1.0\n", encoding="utf-8")
    _write_segment(tmp_path, "seg1", [(0.1, 0.0, 0.0)])
    _write_segment(tmp_path, "seg01", [(0.2, 0.0, 0.0)])
    _write_segment(tmp_path, "seg2", [(0.3, 0.0, 0.0)])

    result = VASPSegmentReader(tmp_path).read()

    assert [segment.number for segment in result.segments] == [2]
    assert [frame.segment_number for frame in result.frames] == [2]
    assert result.frames[0].local_time_fs == 1.0
    assert result.frames[0].time_fs is None
    duplicate = next(
        warning for warning in result.warnings if warning.code == "duplicate_segment_number"
    )
    assert duplicate.level == "error"
    assert duplicate.segment_number == 1
    assert {path.name for path in duplicate.related_paths} == {"seg1", "seg01"}
    assert "all conflicting directories were excluded" in duplicate.message
    assert any(warning.code == "segment_sequence_gap" for warning in result.warnings)
    assert result.has_errors


def test_timestep_metadata_falls_back_to_outcar(tmp_path: Path) -> None:
    segment = _write_segment(tmp_path, "seg01", [(0.1, 0.0, 0.0)])
    (tmp_path / "INCAR").write_text("POTIM = 9.0\n", encoding="utf-8")
    (segment / "OUTCAR").write_text(
        " POTIM = 1.5 time-step for ionic-motion\n"
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )

    result = VASPSegmentReader(tmp_path).read()

    assert result.frames[0].timestep_fs == 1.5
    assert result.frames[0].local_time_fs == 1.5
    assert not any(warning.code == "time_metadata_unavailable" for warning in result.warnings)


def test_outcar_trajectory_is_used_when_xdatcar_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = tmp_path / "seg01"
    segment.mkdir()
    (segment / "OUTCAR").write_text(
        " POTIM = 1.5 time-step for ionic-motion\n"
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    (segment / "REPORT").write_text("REPORT\n", encoding="utf-8")
    fallback_frames = [
        Atoms("H", positions=[(1.0, 0.0, 0.0)], cell=(10.0, 10.0, 10.0), pbc=True),
        Atoms("H", positions=[(2.0, 0.0, 0.0)], cell=(10.0, 10.0, 10.0), pbc=True),
    ]

    def fake_ase_read(*args: object, **kwargs: object) -> list[Atoms]:
        assert kwargs["format"] == "vasp-out"
        return fallback_frames

    monkeypatch.setattr(vasp_segments_module, "ase_read", fake_ase_read)

    result = VASPSegmentReader(tmp_path).read()

    assert [frame.source_kind for frame in result.frames] == ["OUTCAR", "OUTCAR"]
    assert [frame.source_file for frame in result.frames] == [
        segment / "OUTCAR",
        segment / "OUTCAR",
    ]
    assert [frame.local_frame_index for frame in result.frames] == [0, 1]
    assert [frame.configuration_number for frame in result.frames] == [None, None]
    assert [frame.local_time_fs for frame in result.frames] == [1.5, 3.0]
    assert [frame.time_fs for frame in result.frames] == [1.5, 3.0]
    assert any(warning.code == "missing_file" for warning in result.warnings)


def test_partial_segments_return_structured_warnings_and_complete_frames(
    tmp_path: Path,
) -> None:
    (tmp_path / "INCAR").write_text("POTIM = 0.5\n", encoding="utf-8")
    first = tmp_path / "seg01"
    first.mkdir()
    truncated_xdatcar = _xdatcar([(0.1, 0.0, 0.0)])
    truncated_xdatcar += "Direct configuration=      2\n"
    (first / "XDATCAR").write_text(truncated_xdatcar, encoding="utf-8")
    (first / "OUTCAR").write_text("unfinished OUTCAR", encoding="utf-8")
    (first / "REPORT").write_text("", encoding="utf-8")

    second = tmp_path / "seg02"
    second.mkdir()
    (second / "OUTCAR").write_text("", encoding="utf-8")

    third = _write_segment(tmp_path, "seg03", [(0.1, 0.0, 0.0)])

    result = VASPSegmentReader(tmp_path).read()

    assert [frame.segment_number for frame in result.frames] == [1, 3]
    assert result.frames[1].source_directory == third
    assert result.frames[1].local_time_fs == 0.5
    assert result.frames[1].time_fs is None
    warnings = {(warning.code, warning.path) for warning in result.warnings}
    assert ("truncated_file", first / "XDATCAR") in warnings
    assert ("truncated_file", first / "OUTCAR") in warnings
    assert ("incomplete_file", first / "OUTCAR") in warnings
    assert ("empty_file", first / "REPORT") in warnings
    assert ("missing_file", second / "XDATCAR") in warnings
    assert ("empty_file", second / "OUTCAR") in warnings
    assert ("missing_file", second / "REPORT") in warnings


def test_more_than_100_segments_are_read_in_numeric_order(tmp_path: Path) -> None:
    (tmp_path / "INCAR").write_text("POTIM = 1.0\n", encoding="utf-8")
    for number in range(1, 102):
        name = f"seg{number:02d}"
        _write_segment(tmp_path, name, [(number / 1000.0, 0.0, 0.0)])

    result = VASPSegmentReader(tmp_path).read()

    expected = list(range(1, 102))
    assert [segment.number for segment in result.segments] == expected
    assert [frame.segment_number for frame in result.frames] == expected
    assert [frame.global_frame_index for frame in result.frames] == list(range(101))
    assert result.segments[-2].directory.name == "seg100"
    assert result.segments[-1].directory.name == "seg101"
    assert len(result.outcar_data) == 101
    assert len(result.report_data) == 101
