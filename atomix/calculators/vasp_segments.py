"""Read segmented VASP calculations without modifying calculation files."""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read as ase_read

FileKind = Literal["OUTCAR", "REPORT"]
FrameSource = Literal["XDATCAR", "OUTCAR"]
WarningLevel = Literal["warning", "error"]


@dataclass(frozen=True)
class VASPReadWarning:
    """Structured diagnostic emitted while discovering or reading segments."""

    code: str
    message: str
    level: WarningLevel = "warning"
    path: Path | None = None
    segment_number: int | None = None
    related_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class VASPSegment:
    """A uniquely numbered VASP segment directory."""

    number: int | None
    directory: Path


@dataclass
class VASPFrame:
    """A trajectory frame with its segment and time provenance.

    ``local_time_fs`` is derived from the XDATCAR configuration number (or the
    one-based OUTCAR frame index) and ``POTIM``. ``time_fs`` additionally
    includes the durations of preceding segments and is unavailable after a
    segment gap or once an earlier segment lacks the required metadata. The
    femtosecond interpretation of ``POTIM`` applies to molecular dynamics.
    """

    atoms: Atoms
    source_directory: Path
    source_file: Path
    source_kind: FrameSource
    segment_number: int | None
    local_frame_index: int
    global_frame_index: int
    configuration_number: int | None = None
    timestep_fs: float | None = None
    local_time_fs: float | None = None
    time_fs: float | None = None


@dataclass(frozen=True)
class VASPTextData:
    """Raw text and provenance for one OUTCAR or REPORT file."""

    kind: FileKind
    source_directory: Path
    source_file: Path
    segment_number: int | None
    content: str
    line_count: int
    size_bytes: int
    is_complete: bool | None
    is_truncated: bool


@dataclass
class VASPSegmentReadResult:
    """Combined data read from a segmented VASP calculation."""

    segments: list[VASPSegment] = field(default_factory=list)
    frames: list[VASPFrame] = field(default_factory=list)
    outcar_data: list[VASPTextData] = field(default_factory=list)
    report_data: list[VASPTextData] = field(default_factory=list)
    warnings: list[VASPReadWarning] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Whether reading produced an error-level diagnostic."""
        return any(warning.level == "error" for warning in self.warnings)


@dataclass(frozen=True)
class _XDATCARFrame:
    atoms: Atoms
    local_index: int
    configuration_number: int | None


class VASPSegmentReader:
    """Read VASP outputs distributed across ``seg<number>`` directories.

    Segment names are identified by their integer suffix, so names with mixed
    padding are sorted numerically. If two directories map to the same integer
    (for example ``seg1`` and ``seg01``), neither is read and an error-level
    diagnostic identifies both paths.

    When any segment directories are present, root-directory ``XDATCAR``,
    ``OUTCAR``, and ``REPORT`` files are ignored. If no segment directories are
    present, the root directory is treated as a single unnumbered source.

    Parameters
    ----------
    directory : Path | str
        Calculation root containing segment directories.
    boundary_tolerance : float
        Cartesian tolerance in Angstrom used to identify a repeated frame at
        the boundary between consecutive segments.
    """

    _SEGMENT_RE = re.compile(r"^seg(?P<number>\d+)$")
    _CONFIG_RE = re.compile(r"Direct\s+configuration\s*=\s*(?P<number>\d+)")
    _POTIM_RE = re.compile(
        r"^\s*POTIM\s*=\s*(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)",
        re.IGNORECASE,
    )
    _ROOT_DATA_FILES = ("XDATCAR", "OUTCAR", "REPORT")
    _OUTCAR_COMPLETION_MARKER = "General timing and accounting"

    def __init__(
        self,
        directory: Path | str = ".",
        boundary_tolerance: float = 1e-6,
    ) -> None:
        if not math.isfinite(boundary_tolerance) or boundary_tolerance <= 0:
            raise ValueError("boundary_tolerance must be finite and positive")
        self.directory = Path(directory)
        self.boundary_tolerance = boundary_tolerance

    def read(self) -> VASPSegmentReadResult:
        """Discover and read all unambiguous segments.

        Returns
        -------
        VASPSegmentReadResult
            Canonical trajectory frames, raw OUTCAR/REPORT text, and all
            structured diagnostics. The method performs no filesystem writes.
        """
        result = VASPSegmentReadResult()
        segments, discovery_warnings = self._discover_segments()
        result.segments = segments
        result.warnings.extend(discovery_warnings)

        cumulative_time_fs: float | None = 0.0
        previous_frame: Atoms | None = None
        previous_segment_number: int | None = None

        if segments and segments[0].number is not None and segments[0].number > 1:
            first_segment = segments[0]
            cumulative_time_fs = None
            result.warnings.append(
                VASPReadWarning(
                    code="segment_sequence_gap",
                    message=(
                        f"Segment sequence starts at {first_segment.number}; "
                        "cumulative time is unavailable because preceding segments "
                        "may be missing."
                    ),
                    path=first_segment.directory,
                    segment_number=first_segment.number,
                )
            )

        for segment in segments:
            if (
                previous_frame is not None
                and previous_segment_number is not None
                and segment.number is not None
                and segment.number != previous_segment_number + 1
            ):
                result.warnings.append(
                    VASPReadWarning(
                        code="segment_sequence_gap",
                        message=(
                            f"Segment sequence jumps from {previous_segment_number} "
                            f"to {segment.number}; boundary de-duplication and "
                            "cumulative time were reset."
                        ),
                        path=segment.directory,
                        segment_number=segment.number,
                    )
                )
                previous_frame = None
                cumulative_time_fs = None

            outcar = self._read_text_file(segment, "OUTCAR", result.warnings)
            if outcar is not None:
                result.outcar_data.append(outcar)

            report = self._read_text_file(segment, "REPORT", result.warnings)
            if report is not None:
                result.report_data.append(report)

            timestep_fs = self._read_timestep(segment, outcar, result.warnings)

            parsed_frames = self._read_xdatcar(segment, result.warnings)
            source_kind: FrameSource = "XDATCAR"
            source_file = segment.directory / "XDATCAR"

            if not parsed_frames and outcar is not None and outcar.content:
                parsed_frames = self._read_outcar_frames(segment, result.warnings)
                source_kind = "OUTCAR"
                source_file = segment.directory / "OUTCAR"

            local_steps = [
                frame.configuration_number
                if frame.configuration_number is not None
                else frame.local_index + 1
                for frame in parsed_frames
            ]
            segment_time_offset_fs = cumulative_time_fs

            for index, parsed in enumerate(parsed_frames):
                if index == 0 and previous_frame is not None:
                    if self._frames_match(previous_frame, parsed.atoms):
                        result.warnings.append(
                            VASPReadWarning(
                                code="duplicate_boundary_frame",
                                message=(
                                    "Skipped the first frame because it duplicates the "
                                    "last retained frame from the preceding segment."
                                ),
                                path=source_file,
                                segment_number=segment.number,
                            )
                        )
                        first_step = (
                            parsed.configuration_number
                            if parsed.configuration_number is not None
                            else parsed.local_index + 1
                        )
                        if segment_time_offset_fs is not None and timestep_fs is not None:
                            segment_time_offset_fs -= first_step * timestep_fs
                        continue

                local_step = (
                    parsed.configuration_number
                    if parsed.configuration_number is not None
                    else parsed.local_index + 1
                )
                local_time_fs = local_step * timestep_fs if timestep_fs is not None else None
                time_fs = None
                if segment_time_offset_fs is not None and local_time_fs is not None:
                    time_fs = segment_time_offset_fs + local_time_fs

                result.frames.append(
                    VASPFrame(
                        atoms=parsed.atoms,
                        source_directory=segment.directory,
                        source_file=source_file,
                        source_kind=source_kind,
                        segment_number=segment.number,
                        local_frame_index=parsed.local_index,
                        global_frame_index=len(result.frames),
                        configuration_number=parsed.configuration_number,
                        timestep_fs=timestep_fs,
                        local_time_fs=local_time_fs,
                        time_fs=time_fs,
                    )
                )
                previous_frame = parsed.atoms

            if parsed_frames:
                previous_segment_number = segment.number
                if segment_time_offset_fs is None or timestep_fs is None:
                    cumulative_time_fs = None
                else:
                    cumulative_time_fs = segment_time_offset_fs + max(local_steps) * timestep_fs
            else:
                previous_frame = None
                previous_segment_number = None
                cumulative_time_fs = None

        return result

    def _discover_segments(self) -> tuple[list[VASPSegment], list[VASPReadWarning]]:
        warnings: list[VASPReadWarning] = []

        if not self.directory.exists():
            warnings.append(
                VASPReadWarning(
                    code="base_directory_missing",
                    message=f"Calculation directory does not exist: {self.directory}",
                    level="error",
                    path=self.directory,
                )
            )
            return [], warnings

        if not self.directory.is_dir():
            warnings.append(
                VASPReadWarning(
                    code="base_directory_not_directory",
                    message=f"Calculation path is not a directory: {self.directory}",
                    level="error",
                    path=self.directory,
                )
            )
            return [], warnings

        try:
            children = sorted(self.directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            warnings.append(
                VASPReadWarning(
                    code="base_directory_unreadable",
                    message=f"Could not inspect calculation directory: {exc}",
                    level="error",
                    path=self.directory,
                )
            )
            return [], warnings

        numbered_paths: dict[int, list[Path]] = {}
        for child in children:
            match = self._SEGMENT_RE.fullmatch(child.name)
            if match is None or not child.is_dir():
                continue
            number = int(match.group("number"))
            numbered_paths.setdefault(number, []).append(child)

        if not numbered_paths:
            return [VASPSegment(number=None, directory=self.directory)], warnings

        segments: list[VASPSegment] = []
        for number in sorted(numbered_paths):
            paths = numbered_paths[number]
            if len(paths) > 1:
                names = ", ".join(path.name for path in paths)
                warnings.append(
                    VASPReadWarning(
                        code="duplicate_segment_number",
                        message=(
                            f"Segment number {number} is ambiguous ({names}); "
                            "all conflicting directories were excluded."
                        ),
                        level="error",
                        segment_number=number,
                        related_paths=tuple(paths),
                    )
                )
                continue
            segments.append(VASPSegment(number=number, directory=paths[0]))

        root_paths = tuple(
            self.directory / name
            for name in self._ROOT_DATA_FILES
            if (self.directory / name).exists()
        )
        if root_paths:
            warnings.append(
                VASPReadWarning(
                    code="root_data_ignored",
                    message=(
                        "Ignored root-directory VASP output files because segment "
                        "directories are present."
                    ),
                    path=self.directory,
                    related_paths=root_paths,
                )
            )

        return segments, warnings

    def _read_timestep(
        self,
        segment: VASPSegment,
        outcar: VASPTextData | None,
        warnings: list[VASPReadWarning],
    ) -> float | None:
        # OUTCAR records the value VASP actually used, so it is more
        # authoritative than an input file that may have changed after a run.
        if outcar is not None:
            for line in outcar.content.splitlines():
                match = self._POTIM_RE.match(line)
                if match is not None:
                    value = match.group("value").replace("D", "E").replace("d", "e")
                    try:
                        return float(value)
                    except ValueError as exc:
                        warnings.append(
                            VASPReadWarning(
                                code="time_metadata_parse_error",
                                message=f"Could not read POTIM from OUTCAR: {exc}",
                                path=outcar.source_file,
                                segment_number=segment.number,
                            )
                        )

        candidates = [segment.directory / "INCAR"]
        root_incar = self.directory / "INCAR"
        if root_incar not in candidates:
            candidates.append(root_incar)

        for path in candidates:
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    continue
                with path.open(errors="replace") as handle:
                    for line in handle:
                        clean_line = line.split("!", 1)[0].split("#", 1)[0]
                        match = self._POTIM_RE.match(clean_line)
                        if match is not None:
                            value = match.group("value").replace("D", "E").replace("d", "e")
                            return float(value)
            except (OSError, ValueError) as exc:
                warnings.append(
                    VASPReadWarning(
                        code="time_metadata_parse_error",
                        message=f"Could not read POTIM from {path.name}: {exc}",
                        path=path,
                        segment_number=segment.number,
                    )
                )

        warnings.append(
            VASPReadWarning(
                code="time_metadata_unavailable",
                message=("POTIM was not available in OUTCAR, the segment INCAR, or root INCAR."),
                path=segment.directory,
                segment_number=segment.number,
            )
        )
        return None

    def _read_text_file(
        self,
        segment: VASPSegment,
        kind: FileKind,
        warnings: list[VASPReadWarning],
    ) -> VASPTextData | None:
        path = segment.directory / kind
        if not path.exists():
            warnings.append(
                VASPReadWarning(
                    code="missing_file",
                    message=f"{kind} is missing.",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return None
        if not path.is_file():
            warnings.append(
                VASPReadWarning(
                    code="invalid_file_type",
                    message=f"Expected {kind} to be a regular file.",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return None

        try:
            content = path.read_text(errors="replace")
            size_bytes = path.stat().st_size
        except OSError as exc:
            warnings.append(
                VASPReadWarning(
                    code="file_read_error",
                    message=f"Could not read {kind}: {exc}",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return None

        if not content:
            warnings.append(
                VASPReadWarning(
                    code="empty_file",
                    message=f"{kind} is empty.",
                    path=path,
                    segment_number=segment.number,
                )
            )

        is_truncated = bool(content) and not content.endswith(("\n", "\r"))
        if is_truncated:
            warnings.append(
                VASPReadWarning(
                    code="truncated_file",
                    message=f"{kind} does not end with a complete text line.",
                    path=path,
                    segment_number=segment.number,
                )
            )

        is_complete: bool | None = None
        if kind == "OUTCAR":
            is_complete = self._OUTCAR_COMPLETION_MARKER in content
            if content and not is_complete:
                warnings.append(
                    VASPReadWarning(
                        code="incomplete_file",
                        message=(
                            "OUTCAR has no final timing marker and may represent an "
                            "incomplete or interrupted run."
                        ),
                        path=path,
                        segment_number=segment.number,
                    )
                )

        return VASPTextData(
            kind=kind,
            source_directory=segment.directory,
            source_file=path,
            segment_number=segment.number,
            content=content,
            line_count=len(content.splitlines()),
            size_bytes=size_bytes,
            is_complete=is_complete,
            is_truncated=is_truncated,
        )

    def _read_xdatcar(
        self,
        segment: VASPSegment,
        warnings: list[VASPReadWarning],
    ) -> list[_XDATCARFrame]:
        path = segment.directory / "XDATCAR"
        if not path.exists():
            warnings.append(
                VASPReadWarning(
                    code="missing_file",
                    message="XDATCAR is missing; OUTCAR will be tried as a trajectory fallback.",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return []
        if not path.is_file():
            warnings.append(
                VASPReadWarning(
                    code="invalid_file_type",
                    message="Expected XDATCAR to be a regular file.",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return []

        try:
            content = path.read_text(errors="replace")
        except OSError as exc:
            warnings.append(
                VASPReadWarning(
                    code="file_read_error",
                    message=f"Could not read XDATCAR: {exc}",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return []

        if not content:
            warnings.append(
                VASPReadWarning(
                    code="empty_file",
                    message="XDATCAR is empty; OUTCAR will be tried as a trajectory fallback.",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return []

        prepared, metadata, incomplete_count = self._prepare_xdatcar(content)
        if incomplete_count:
            warnings.append(
                VASPReadWarning(
                    code="truncated_file",
                    message=(
                        f"XDATCAR contains {incomplete_count} incomplete frame(s); "
                        "complete frames were retained."
                    ),
                    path=path,
                    segment_number=segment.number,
                )
            )

        if not prepared or not metadata:
            warnings.append(
                VASPReadWarning(
                    code="invalid_file",
                    message="XDATCAR contains no complete, parseable frames.",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return []

        try:
            parsed = ase_read(io.StringIO(prepared), index=":", format="vasp-xdatcar")
            atoms_list = parsed if isinstance(parsed, list) else [parsed]
        except Exception as exc:
            warnings.append(
                VASPReadWarning(
                    code="parse_error",
                    message=f"Could not parse complete XDATCAR frames: {exc}",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return []

        if len(atoms_list) != len(metadata):
            warnings.append(
                VASPReadWarning(
                    code="frame_count_mismatch",
                    message=(
                        f"XDATCAR metadata described {len(metadata)} frames but ASE "
                        f"returned {len(atoms_list)}; only aligned frames were retained."
                    ),
                    path=path,
                    segment_number=segment.number,
                )
            )

        count = min(len(atoms_list), len(metadata))
        return [
            _XDATCARFrame(
                atoms=atoms_list[index],
                local_index=metadata[index][0],
                configuration_number=metadata[index][1],
            )
            for index in range(count)
        ]

    def _prepare_xdatcar(
        self,
        content: str,
    ) -> tuple[str, list[tuple[int, int | None]], int]:
        lines = content.splitlines()
        header_indices = [
            index for index, line in enumerate(lines) if "Direct configuration" in line
        ]
        if not header_indices:
            return "", [], 1

        first_header = header_indices[0]
        atom_count = self._find_xdatcar_atom_count(lines, first_header)
        if atom_count is None or atom_count <= 0:
            return "", [], len(header_indices)

        selected_lines = list(lines[:first_header])
        metadata: list[tuple[int, int | None]] = []
        incomplete_count = 0

        for local_index, header_index in enumerate(header_indices):
            coordinate_lines = lines[header_index + 1 : header_index + 1 + atom_count]
            if len(coordinate_lines) != atom_count or not all(
                self._valid_coordinate_line(line) for line in coordinate_lines
            ):
                incomplete_count += 1
                continue

            selected_lines.append(lines[header_index])
            selected_lines.extend(coordinate_lines)
            match = self._CONFIG_RE.search(lines[header_index])
            configuration_number = int(match.group("number")) if match is not None else None
            metadata.append((local_index, configuration_number))

        if not metadata:
            return "", [], incomplete_count
        return "\n".join(selected_lines) + "\n", metadata, incomplete_count

    @staticmethod
    def _find_xdatcar_atom_count(lines: list[str], first_header: int) -> int | None:
        for index in range(first_header - 1, 4, -1):
            fields = lines[index].split()
            if fields and all(field.isdigit() for field in fields):
                return sum(int(field) for field in fields)
        return None

    @staticmethod
    def _valid_coordinate_line(line: str) -> bool:
        fields = line.split()
        if len(fields) < 3:
            return False
        try:
            for value in fields[:3]:
                float(value)
        except ValueError:
            return False
        return True

    def _read_outcar_frames(
        self,
        segment: VASPSegment,
        warnings: list[VASPReadWarning],
    ) -> list[_XDATCARFrame]:
        path = segment.directory / "OUTCAR"
        try:
            parsed = ase_read(str(path), index=":", format="vasp-out")
            atoms_list = parsed if isinstance(parsed, list) else [parsed]
        except Exception as exc:
            warnings.append(
                VASPReadWarning(
                    code="parse_error",
                    message=f"Could not parse OUTCAR trajectory fallback: {exc}",
                    path=path,
                    segment_number=segment.number,
                )
            )
            return []

        return [
            _XDATCARFrame(atoms=atoms, local_index=index, configuration_number=None)
            for index, atoms in enumerate(atoms_list)
        ]

    def _frames_match(self, first: Atoms, second: Atoms) -> bool:
        if first.get_chemical_symbols() != second.get_chemical_symbols():
            return False
        if not np.array_equal(first.get_pbc(), second.get_pbc()):
            return False
        if not np.allclose(
            first.cell.array,
            second.cell.array,
            rtol=0.0,
            atol=self.boundary_tolerance,
        ):
            return False

        differences = first.positions - second.positions
        try:
            _, distances = find_mic(differences, first.cell, first.get_pbc())
        except (ValueError, np.linalg.LinAlgError):
            distances = np.linalg.norm(differences, axis=1)
        return bool(np.all(distances <= self.boundary_tolerance))
