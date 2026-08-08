# atomix Roadmap

This roadmap keeps atomix easy to resume. It turns the project vision into milestones, epics, and issue-sized tasks with priorities, sizes, dependencies, and compute needs.

## Product Goal

Build a natural-language-assisted toolkit for atomistic modeling workflows that lets a computational materials researcher describe what they want to simulate, then reliably generate, run, inspect, and analyze VASP-first workflows with a path toward MLIP-accelerated screening.

The useful product is not a chatbot demo. The useful product is a trustworthy workflow layer: structure context, calculation setup, job submission, result parsing, and analysis, with natural language as an interface on top of reproducible scientific primitives.

## Current Status

- Early scaffolding exists for CLI commands, VASP input generation, calculation classes, workflow orchestration, site tools, analysis helpers, MLIP wrappers, and AI generation.
- CI and tests exist.
- README states most features are still stubs.
- A supported `inspect-vasp` CLI and Python API read segmented VASP outputs without modifying them.
- The historical `atomix-pypi-release/` placeholder is preserved but excluded from package discovery.
- Licensing is undecided before public release or external sharing.
- API churn is expected through v0.2.0.

## Priority Scale

- `P0`: required before v0.2.0 is trustworthy or releasable.
- `P1`: core MVP functionality for real local use.
- `P2`: expansion after core workflows are stable.
- `P3`: later research/product ideas.

## Size Scale

- `S`: 30-90 minutes, one focused change or audit.
- `M`: half day, multiple files or a small feature.
- `L`: one to two days, clear substeps.
- `XL`: too big for direct implementation; split before starting.

## Compute Tags

- `compute:none`: design/docs/API decisions.
- `compute:light`: local tests, lint, package build, mock workflows.
- `compute:network`: PyPI/GitHub/LLM-provider work.
- `compute:cluster`: Slurm/HPC integration or remote smoke tests.
- `compute:gpu`: MLIP screening or training/inference that needs GPU.
- `compute:agent`: good candidate for a separate agent chat.

## Milestones

### M0 - v0.2.0 Release Foundation

Outcome: atomix has one authoritative package configuration, predictable installs, clear licensing direction, and a clean release boundary.

Acceptance:
- [x] Main repo can build a wheel from the root.
- [x] Minimal install works without heavy optional dependencies.
- [x] `atomix --help` and a basic info/help command work from a fresh environment.
- [ ] Optional extras for MLIP, LLM, and dev install cleanly. (Restructured: heavy deps moved to a new `science` extra; a clean-venv `pip install .[all]` has not yet been re-exercised.)
- [x] `atomix-pypi-release/` is explicitly marked historical and excluded from current builds.
- [ ] License decision is recorded before public release. (Deferred 2026-06-02.)
- [x] `pytest tests/` passes.
- [x] Built wheel contents and a fresh wheel installation are checked in CI.

### M1 - VASP Setup MVP

Outcome: a user can create a small VASP calculation setup from a structure and structured or natural-language input, then inspect the generated files before running anything.

Acceptance:
- [ ] `atomix generate ... --dry-run` gives useful validation output.
- [ ] Generation with a provided structure writes POSCAR, INCAR, KPOINTS, and job script where appropriate.
- [ ] VASP defaults are documented and test-covered.
- [ ] Missing structure, missing POTCAR assumptions, and unsupported requests fail clearly.
- [ ] No real cluster submission is required for this milestone.

### M2 - Calculation Management MVP

Outcome: atomix can manage a calculation directory enough to submit, status-check, and parse simple results.

Acceptance:
- [ ] `atomix submit --dry-run` produces scheduler scripts for Slurm without submitting.
- [ ] `atomix status` recognizes not-started, running/incomplete, completed, and failed directories from file evidence.
- [ ] `atomix analyze` returns useful summary/energy/forces output.
- [ ] File-based mode and direct ASE/MLIP mode have clear boundaries.

### M3 - Catalysis Workflow MVP

Outcome: atomix can support common surface science workflows beyond generic calculation setup.

Acceptance:
- [ ] Surface slab/site helpers cover top/bridge/hollow style enumeration for a simple metal slab.
- [ ] Adsorption energy workflow is documented and test-covered with mock or lightweight calculators.
- [ ] Example workflow exists for a small surface adsorption case.

### M4 - MLIP Screening And Active Learning

Outcome: atomix can run MLIP-first screening workflows and export reference data for later validation/fine-tuning.

Acceptance:
- [ ] MLIP calculator wrappers are lazy-loaded and optional.
- [ ] `screen` and `screen-sites` commands work with mocked calculators in tests and documented real backends.
- [ ] Training-data export from DFT output directories has a tested schema.
- [ ] Active-learning helper boundaries are defined.

### M5 - Natural Language Layer

Outcome: natural language becomes a reliable interface over the workflow primitives, not a separate fragile system.

Acceptance:
- [ ] Prompt/context docs are versioned and testable.
- [ ] Generated calculation parameters are validated before writing files.
- [ ] Provider failures degrade with clear errors.
- [ ] Example prompts map to deterministic expected parameter schemas in tests.

## Epics

### Epic: Packaging And Release

Goal: make atomix installable, testable, and releasable from one source tree.

Candidate tasks:
- Keep `atomix-pypi-release/` as a read-only historical snapshot excluded from package discovery (`done`).
- Move heavy dependencies to optional extras and lazy imports where needed (`P0`, `size:M`, `type:maintenance`, `area:package`, `compute:light`).
- Maintain package build and fresh-venv install smoke tests (`done`).
- Record license decision for v0.2.0 (`P0`, `size:S`, `type:decision`, `area:license`, `compute:none`).

### Epic: CLI And User Workflows

Goal: make the command-line surface coherent and useful before deeper features spread.

Candidate tasks:
- Audit current CLI commands and document stable vs experimental commands (`P0`, `size:S`, `type:audit`, `area:cli`, `compute:none`).
- Add or improve `atomix info` / environment diagnostics (`P1`, `size:S`, `type:feature`, `area:cli`, `compute:light`).
- Create a minimal end-to-end example for structure -> VASP inputs -> status/analyze (`P1`, `size:M`, `type:feature`, `area:cli`, `compute:light`).

### Epic: VASP Input Generation

Goal: generate conservative, inspectable VASP inputs using established libraries.

Candidate tasks:
- Define supported calculation types and default parameters for v0.2.0 (`P1`, `size:S`, `type:decision`, `area:vasp`, `compute:none`).
- Add tests for static, relax, and AIMD input generation (`P1`, `size:M`, `type:test`, `area:vasp`, `compute:light`).
- Document POTCAR handling and what atomix does not automate (`P1`, `size:S`, `type:docs`, `area:vasp`, `compute:none`).

### Epic: Workflow And Job Management

Goal: support direct ASE/MLIP workflows and file-based DFT workflows without mixing their assumptions.

Candidate tasks:
- Clarify direct mode vs file mode in docs and tests (`P1`, `size:S`, `type:docs`, `area:workflow`, `compute:light`).
- Add Slurm script generation tests for common VASP settings (`P1`, `size:M`, `type:test`, `area:workflow`, `compute:light`).
- Evaluate atomate2/jobflow-remote as a backend instead of expanding custom job management (`P1`, `size:M`, `type:research`, `area:workflow`, `compute:none`).

### Epic: Analysis And Catalysis

Goal: provide useful scientific primitives for surface/catalysis workflows.

Candidate tasks:
- Add a documented adsorption-energy example using mock/lightweight calculators (`P1`, `size:M`, `type:feature`, `area:analysis`, `compute:light`).
- Tighten energy/forces/trajectory analyzer outputs and JSON schemas (`P1`, `size:M`, `type:feature`, `area:analysis`, `compute:light`).
- Expand surface-site enumeration tests for simple facets (`P2`, `size:M`, `type:test`, `area:sites`, `compute:light`).

### Epic: MLIP Workflows

Goal: make MLIPs a validated acceleration layer, not a vague claim.

Candidate tasks:
- Keep MLIP dependencies optional and import-safe (`P0`, `size:M`, `type:maintenance`, `area:mlip`, `compute:light`).
- Add a lightweight MLIP screening example with mocked backend and documented real backend (`P1`, `size:M`, `type:feature`, `area:mlip`, `compute:light`).
- Define training-data export schema for VASP outputs (`P1`, `size:M`, `type:feature`, `area:mlip`, `compute:light`).
- Later: integrate with a real Pt-water/MLIP validation workflow only after the standalone MLIP manuscript path is clearer (`P3`, `size:L`, `type:research`, `area:mlip`, `compute:gpu`).

### Epic: Natural Language Layer

Goal: expose reliable workflow primitives through natural language only after the primitives have stable schemas.

Candidate tasks:
- Define output schema for NL-generated calculation setup (`P1`, `size:M`, `type:feature`, `area:ai`, `compute:none`).
- Add validation layer between LLM output and file writing (`P1`, `size:M`, `type:feature`, `area:ai`, `compute:light`).
- Add deterministic tests using fixture LLM responses (`P1`, `size:M`, `type:test`, `area:ai`, `compute:light`).
- Later: support provider selection and local docs context more robustly (`P2`, `size:M`, `type:feature`, `area:ai`, `compute:network`).

## Priority Backlog

| Priority | Task | Size | Compute | Depends On | Notes |
|---|---|---|---|---|---|
| P0 | Preserve but exclude historical `atomix-pypi-release/` | L | light/agent | none | Done; artifact checker prevents regression. |
| P0 | Move heavy dependencies behind optional extras/lazy imports | M | light | package convergence plan | Needed for minimal install. |
| P0 | Add package build and fresh-venv smoke tests | M | light | dependency cleanup | Done; verifies the unreleased wheel install path. |
| P0 | Record license decision | S | none | user decision | Needed before public release/external sharing. |
| P0 | CLI surface audit: stable vs experimental | S | none/agent | none | Done; only `inspect-vasp` is documented as supported. |
| P1 | Minimal end-to-end VASP setup example | M | light | CLI audit | First useful demo. |
| P1 | Tests for static/relax/AIMD input generation | M | light | VASP defaults decision | Core trust-building. |
| P1 | Direct mode vs file mode docs/tests | S | light | none | Clarifies workflow model. |
| P1 | atomate2/jobflow-remote backend evaluation | M | none/agent | none | Decide before growing custom scheduler code. |
| P1 | NL output schema and validation layer | M | light | VASP setup schemas | Keeps AI layer safe. |
| P2 | Surface adsorption example | M | light | analysis schema | Useful catalysis demo. |
| P2 | Real MLIP screening example | M | gpu/network | optional deps stable | Later, after package cleanup. |

## Ready-To-Delegate Tasks

### atomate2/jobflow-remote Evaluation

Labels: `priority:P1`, `size:M`, `type:research`, `area:workflow`, `compute:agent`

Goal: decide whether atomix should wrap atomate2/jobflow-remote instead of building custom job-management layers.

Acceptance:
- [ ] Compare current atomix job/workflow code to atomate2/jobflow-remote capabilities.
- [ ] Identify overlap, gaps, and migration risk.
- [ ] Recommend whether v0.2.0 should depend on, wrap, or defer atomate2/jobflow.

## Later

- QE support after VASP path is stable.
- CP2K support beyond placeholder interfaces.
- Natural-language multi-step workflow generation.
- Cluster-specific profiles for Gautschi, NERSC, and other HPC systems.
- Active-learning loops tied to real MLIP fine-tuning workflows.
- Public documentation site.

## Issue Labels

- Priorities: `priority:P0`, `priority:P1`, `priority:P2`, `priority:P3`
- Sizes: `size:S`, `size:M`, `size:L`, `size:XL`
- Types: `type:feature`, `type:bug`, `type:docs`, `type:test`, `type:audit`, `type:infra`, `type:maintenance`, `type:research`, `type:decision`
- Areas: `area:package`, `area:cli`, `area:vasp`, `area:workflow`, `area:analysis`, `area:sites`, `area:mlip`, `area:ai`, `area:ci`, `area:license`
- Compute: `compute:none`, `compute:light`, `compute:network`, `compute:cluster`, `compute:gpu`, `compute:agent`
- Status: `status:blocked`, `status:ready`, `status:needs-decision`

## Recommended Next Task

Exercise `atomix inspect-vasp` on additional real segmented calculations and
refine only the summary fields that prove useful. Keep publication blocked
until the product boundary and license are deliberately chosen.
