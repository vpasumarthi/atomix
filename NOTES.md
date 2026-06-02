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
1. [DONE 2026-06-02] Heavy deps are now imported lazily. CLI commands already deferred imports into command bodies; subpackage `__init__.py` files now use PEP 562 `__getattr__` so `import atomix.core` etc. no longer pull ase/numpy/pymatgen. Also fixed `generate --dry-run` to not import ase on the no-structure path.
2. [DONE 2026-06-02] Heavy deps moved out of `[project.dependencies]` (now `click` only) into a new `science` extra (ase, numpy, scipy, pymatgen, pyyaml, requests); `all = [science,mlip,llm,dev]`.
3. [DONE 2026-06-02] `python -m build` wheel installs with click-only deps; `atomix --help`/`info`/`--version` and `generate --dry-run` all work on the minimal install; `info` command ported to the root CLI; `__version__` synced to 0.1.1.
4. [PARTIAL 2026-06-02] Fresh-venv minimal install verified (only atomix+click present; all subpackages import; heavy-class access fails cleanly). Full test suite (110 passed) runs in the editable `atomix` conda env. A literal `pip install .[all]` in a clean venv has not been re-exercised.
5. [TODO] Delete `atomix-pypi-release/` and update README install instructions to reference the extras. BLOCKED on the license decision (deferred 2026-06-02): the only LICENSE file currently lives inside the slim dir, so the license must be recorded at repo root before the slim dir can be removed.

Do this as a discrete prep step, not bundled with v0.2.0 feature work.
