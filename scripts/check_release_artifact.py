#!/usr/bin/env python3
"""Fail when an Atomix wheel omits runtime data or includes legacy sources."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REQUIRED_FILES = {
    "atomix/ai/docs/analysis.md",
    "atomix/ai/docs/vasp.md",
    "atomix/ai/docs/workflows.md",
}
FORBIDDEN_PREFIXES = ("atomix-pypi-release/",)


def source_version() -> str:
    """Read the single package version without importing the source tree."""
    init_path = Path(__file__).resolve().parents[1] / "atomix" / "__init__.py"
    for line in init_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__ = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError(f"Could not find __version__ in {init_path}")


def check_wheel(wheel: Path) -> list[str]:
    """Return artifact errors for one wheel."""
    errors: list[str] = []
    expected_version = source_version()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED_FILES - names)
        if missing:
            errors.append(f"missing runtime files: {', '.join(missing)}")

        forbidden = sorted(
            name for name in names if any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
        )
        if forbidden:
            errors.append(f"legacy package files included: {', '.join(forbidden)}")

        metadata_files = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_files) != 1:
            errors.append(f"expected one METADATA file, found {len(metadata_files)}")
        else:
            metadata = archive.read(metadata_files[0]).decode("utf-8")
            if f"Version: {expected_version}\n" not in metadata:
                errors.append(f"wheel metadata does not report version {expected_version}")

    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_release_artifact.py WHEEL", file=sys.stderr)
        return 2

    wheel = Path(argv[1])
    errors = check_wheel(wheel)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Artifact check passed: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
