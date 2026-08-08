# atomix

[![CI](https://github.com/vpasumarthi/atomix/actions/workflows/ci.yml/badge.svg)](https://github.com/vpasumarthi/atomix/actions/workflows/ci.yml)

Atomistic Modeling Interface for eXploration — a natural language driven toolkit for ab initio / DFT / atomistic modeling workflows.

## Installation

```bash
conda activate atomix
pip install -e .
```

## Status

Early development. Most features are stubs. A read-only segmented VASP reader
is available for calculations stored as `seg01`, `seg02`, ... directories.
It requires the scientific dependencies: `pip install -e ".[science]"`.

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
