#!/usr/bin/env python3
"""Fail when an Atomix release artifact omits data or includes legacy sources."""

from __future__ import annotations

import sys
import tarfile
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


def check_sdist(sdist: Path) -> list[str]:
    """Return artifact errors for one source distribution."""
    errors: list[str] = []
    expected_version = source_version()
    with tarfile.open(sdist, "r:gz") as archive:
        raw_names = [member.name for member in archive.getmembers()]
        roots = {name.split("/", 1)[0] for name in raw_names if name}
        if len(roots) != 1:
            return [f"expected one source-distribution root, found {len(roots)}"]

        root = roots.pop()
        names = {
            name[len(root) + 1 :]
            for name in raw_names
            if name.startswith(f"{root}/") and len(name) > len(root) + 1
        }
        missing = sorted(REQUIRED_FILES - names)
        if missing:
            errors.append(f"missing runtime files: {', '.join(missing)}")

        forbidden = sorted(
            name for name in names if any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
        )
        if forbidden:
            errors.append(f"legacy package files included: {', '.join(forbidden)}")

        pkg_info_name = f"{root}/PKG-INFO"
        try:
            pkg_info = archive.extractfile(pkg_info_name)
        except KeyError:
            pkg_info = None
        if pkg_info is None:
            errors.append("source distribution has no root PKG-INFO")
        else:
            metadata = pkg_info.read().decode("utf-8")
            if f"Version: {expected_version}\n" not in metadata:
                errors.append(f"source metadata does not report version {expected_version}")

    return errors


def check_artifact(artifact: Path) -> list[str]:
    """Dispatch an artifact to its format-specific checks."""
    if artifact.suffix == ".whl":
        return check_wheel(artifact)
    if artifact.name.endswith(".tar.gz"):
        return check_sdist(artifact)
    return [f"unsupported release artifact: {artifact.name}"]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_release_artifact.py ARTIFACT [ARTIFACT ...]", file=sys.stderr)
        return 2

    failed = False
    for artifact_arg in argv[1:]:
        artifact = Path(artifact_arg)
        errors = check_artifact(artifact)
        if errors:
            failed = True
            for error in errors:
                print(f"ERROR [{artifact.name}]: {error}", file=sys.stderr)
        else:
            print(f"Artifact check passed: {artifact.name}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
