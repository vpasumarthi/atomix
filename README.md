# atomix

[![CI](https://github.com/vpasumarthi/atomix/actions/workflows/ci.yml/badge.svg)](https://github.com/vpasumarthi/atomix/actions/workflows/ci.yml)

Atomistic Modeling Interface for eXploration — a natural language driven toolkit for ab initio / DFT / atomistic modeling workflows.

## Installation

Atomix is currently installed from a source checkout. The segmented VASP
reader requires the scientific dependencies:

```bash
conda activate atomix
pip install -e ".[science]"
```

## Status

Early development. The supported surface is deliberately narrow: read-only
inspection of VASP calculations stored as `seg01`, `seg02`, ... directories.
Other commands and modules remain experimental scaffolding.

### Command line

```bash
atomix inspect-vasp path/to/calculation
atomix inspect-vasp path/to/calculation --json
atomix inspect-vasp path/to/calculation --diagnostics
```

The command exits nonzero when the reader finds an error-level ambiguity, such
as both `seg1` and `seg01` representing the same segment number.

### Python

```python
from atomix.calculators import VASPSegmentReader

result = VASPSegmentReader("path/to/calculation").read()

for frame in result.frames:
    print(frame.segment_number, frame.time_fs, frame.source_file)

for warning in result.warnings:
    print(warning.level, warning.code, warning.path)
```

The reader sorts segments numerically, prefers `XDATCAR` trajectories with an
`OUTCAR` fallback, removes repeated boundary frames, and retains raw `OUTCAR`
and `REPORT` records with source metadata. It never modifies calculation files.

## Development

```bash
pip install -e ".[science,dev]"
pytest tests/
ruff check .
```
