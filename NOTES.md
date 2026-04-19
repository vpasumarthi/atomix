# Development Notes

## Current State

Early scaffolding stage. Initial work toward Phases 1-4 (NL → VASP setup, calculation management, analysis pipeline, MLIP integration) is in the source tree but unstable, untested at scale, and subject to API churn through v0.2.0. The published PyPI release (`pip install atomix`) is a minimal placeholder, not the full toolkit. Real first usable release targeted for v0.2.0.

## Licensing (TBD)

**Status**: Undecided, removed license declaration for now.

**Preference**: Open-core / hybrid model

| Component | License | Rationale |
|-----------|---------|-----------|
| `atomix/core/` | Open (MIT/Apache) | Basic utilities, structure handling |
| `atomix/calculators/` | Open | Standard calculator interfaces |
| `atomix/sites/` | Open | Site identification utilities |
| `atomix/analysis/` | TBD | May split basic vs advanced |
| `atomix/ai/` | Proprietary | NL generation, premium feature |

**Decision needed before**: First public release or external sharing.

**References**:
- GitLab CE/EE model
- Elastic open + X-Pack proprietary

## Computation Backend

Consider atomate2 + jobflow-remote instead of building custom job management. Handles HPC submission, error recovery, provenance tracking out of the box. Key features for atomix:
- MDMaker / MultiMDMaker for AIMD (auto-splits long runs into walltime-safe chunks)
- AdsorptionMaker for surface adsorption calculations
- Custodian for auto-fixing VASP errors
- MongoDB provenance database (queryable results)
- Docs: https://materialsproject.github.io/atomate2/

## Cleanup

**Before next release (v0.2.0): converge to single pyproject; remove `atomix-pypi-release/`.**

The slim variant was created 2026-04-19 as a name-claim shortcut for the v0.1.1 PyPI release (PEP 541 reclaim, issue #9152). Two pyprojects now drift — every metadata change risks updating only one side.

Convergence steps:
1. Refactor main repo so heavy deps (ase, pymatgen, mace-torch, etc.) are imported lazily inside the functions that need them, not at module top level.
2. Move heavy deps from `[project.dependencies]` to `[project.optional-dependencies]` extras (already partially structured: `mlip`, `llm`, `dev`, `all`).
3. Verify `python -m build` from repo root produces a wheel that installs cleanly with only core deps and runs `atomix --help` / `atomix info` equivalent to the slim variant.
4. Test-install in a fresh venv, then in `[all]` mode, confirm both work.
5. Delete `atomix-pypi-release/` and update README install instructions to reference the extras.

Do this as a discrete prep step, not bundled with v0.2.0 feature work.
