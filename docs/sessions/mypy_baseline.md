---
summary: "Triaged mypy baseline from the v0.x quality pass. 2026-06-24 pickup note: Ruff is clean after 684690b; source-only mypy baseline is 43 errors with --follow-imports=skip."
read_when:
  - "starting a mypy / type-annotation cleanup task"
  - "touching structure-reading return types (Atoms vs list[Atoms])"
  - "running mypy repo-wide (note the tifffile follow-imports gotcha below)"
---

# mypy baseline (triaged)

Generated during the mechanical quality pass on **2026-06-03**; refreshed the
same day after two fixes landed.

## Pickup Note — 2026-06-24

Context: Ruff cleanup was completed and pushed in commit `684690b`
(`style: satisfy ruff strict zip checks`). `ruff check . --no-cache` is now
green and `pytest tests/` passes with 113 tests in the `atomix` conda env.

Latest mypy spot checks:

| Scope | Command | Result |
|-------|---------|--------|
| source + tests | `python -m mypy --show-error-codes atomix tests` | 70 errors, 17 files |
| source only, documented baseline mode | `python -m mypy --show-error-codes --follow-imports=skip atomix` | 43 errors, 11 files |

Interpretation: this is still essentially the known mypy baseline, not a
regression from the Ruff cleanup. The +1 relative to the 2026-06-03 documented
42-error baseline is the `yaml` missing-stubs warning surfacing in
`atomix/core/config.py`.

Recommended next pickup:

1. Do a source-only mechanical pass first, using
   `--follow-imports=skip atomix`, and avoid API-contract changes in that pass.
2. Low-risk targets: annotate the `vasp.py` validation result container, add
   the simple numeric accumulator annotations, cast/annotate NumPy/ASE/LLM
   `Any` returns, fix the `adsorption.py` `list[int]`/`list[float]` return
   mismatch, rename the confusing `selected` list in `screening.py`, and rename
   the early `sites` accumulator in `surface.py`.
3. Leave these for a separate decision pass: `BaseCalculation.setup()` return
   contract, `JobSubmitter.submit()` base/subclass signatures, and the
   `Atoms | list[Atoms]` structure-reading boundary in CLI/trajectory code.
4. After source-only cleanup, decide whether tests should be included in the
   mypy gate or tracked separately. The source+tests run currently adds
   test-only typing debt, especially in `tests/test_mlip.py`.

## Update — 2026-06-03 (after fixes)

Two of the headline items from the original triage are now **resolved**:

- ✅ **py3.9 → 3.10+ decision applied** (commit `1dcc4d0`). `requires-python`,
  mypy `python_version`, ruff `target-version`, and the classifier were bumped
  to 3.10. This cleared **all 103 `syntax` (`X | Y` union) errors** with no
  source edits — the code already used PEP 604 unions everywhere, so the bump
  just makes the metadata honest.
- ✅ **`Outcar.converged` bug fixed + tested** (commit `486539e`). Both call
  sites in `vasp.py` now use `bool(outcar.run_stats)` for the OUTCAR-fallback
  completion signal; `tests/test_vasp.py` covers the path with a mock `spec`
  that omits `converged`, so the regression can't return silently.

**New count: 165 → 42 errors.** Nothing else from the list below has been fixed.

### Baseline runs (for reproducibility)

| When | mypy | command | result |
|------|------|---------|--------|
| original | 1.19.1 | `mypy atomix` (full follow) | 165 errors, 16 files |
| current | 2.1.0 | `mypy atomix --follow-imports=skip` | 42 errors, 10 files |

> ⚠️ **Repo-wide gotcha:** under mypy 2.1.0, a plain `mypy atomix` now follows
> a transitive dependency (`tifffile`, via pymatgen) and aborts on its
> Python-3.12 `type` statement before checking atomix. Use
> `--follow-imports=skip`, or add a `follow_imports`/`exclude` rule (or pin
> mypy) in `[tool.mypy]` for a clean repo-wide run. Tracked as an open item.
>
> Because the current run uses `--follow-imports=skip` (lib types become `Any`)
> on a newer mypy, some sub-counts below differ from the original full-follow
> run (notably `arg-type`/`union-attr` dropped as the structure-union no longer
> resolves through `Any` lib boundaries). The **per-module triage stays a valid
> map of the categories of remaining work**; re-run for exact line numbers.

## Error-code summary (current — 42)

| Count | Code | Bucket | Notes |
|------:|------|--------|-------|
| ~~103~~ → 0 | `syntax` | ✅ **done** | Cleared by the py3.10 bump (`1dcc4d0`). |
| 14 | `attr-defined` | easy (1 fix) | All 14 are the `"object" has no attribute "append"` container in `vasp.py`; annotate it at init and they clear. (The 2 `Outcar.converged` are ✅ fixed.) |
| 11 | `no-any-return` | easy | `warn_return_any` on untyped lib returns (ASE/pymatgen/numpy). Cast or annotate. |
| 7 | `override` | **decision** | LSP violations: `setup()` return type, `submit()` signature vs base class. |
| 5 | `var-annotated` | easy | Add an explicit annotation on an accumulator/initial value. |
| 2 | `assignment` | investigate | In `screening.py` — looks like a real bool/list mixup. |
| 1 | `no-redef` | easy/investigate | `sites` redefined in `surface.py`. |
| 1 | `dict-item` | easy | `list[int]` vs `list[float]` — make the literals floats. |
| 1 | `arg-type` | **decision** | Structure-union flowing into an `Atoms`-only API. |

---

## Per-module checklist

Legend: 🟢 likely-easy (unambiguous mechanical fix) · 🟡 needs-decision (API/design) · 🔴 investigate (possible real bug) · ✅ resolved

> The `syntax` (×N) lines below are all ✅ resolved by the py3.10 bump and are
> kept only for traceability. Line numbers predate the `vasp.py` fix edits.

### atomix/cli/main.py — was 26 (now ~11)
- [ ] 🟡 `read_structure()` returns `Atoms | list[Atoms]`; downstream APIs want `Atoms`. Drives the `union-attr`/`arg-type` cluster (`get_chemical_formula` etc.). Decide whether to narrow the return type, raise on multi-frame, or branch on the list case. One design fix resolves the cluster.
- [ ] 🟢 `:335` `var-annotated` — `max_force` needs an explicit annotation (e.g. `max_force: float = 0.0`).
- [x] ✅ syntax ×15 — cleared by py3.10 bump.

### atomix/core/jobs.py — was 18 (now 2)
- [ ] 🟡 `:188, :353` `override` — `SlurmSubmitter.submit()` / `LocalSubmitter.submit()` signatures differ from base `JobSubmitter.submit(self, **kwargs)`. Decide a stable base signature.
- [x] ✅ syntax ×16 — cleared by py3.10 bump.

### atomix/calculators/vasp.py — was 20 (now 14)
- [ ] 🟢 (14×) `"object" has no attribute "append"` (~`:408–:445`, shifted by the fix) — a container is inferred as `object`. Annotate at init (e.g. `errors: list[str] = []`) and all 14 clear.
- [x] ✅ 🔴 `Outcar.converged` (was `:243, :304`) — **fixed** in `486539e`; now `bool(outcar.run_stats)`, covered by `tests/test_vasp.py`.
- [x] ✅ syntax ×3 — cleared by py3.10 bump.
- [ ] 🟡 (if it resurfaces under full-follow) `Poscar()` got `Structure | Molecule`; expects `Structure | IStructure`. Guard against `Molecule` or narrow.

### atomix/core/calculation.py — was 19 (now ~5)
- [ ] 🟡 `:134, :216, :350, :456, :606` `override` — subclass `setup()` returns `dict[str, Path]` but `BaseCalculation.setup()` is annotated `-> None`. Fix the base-class return annotation; one edit resolves all 5.
- [ ] 🔴 `:568` `arg-type` — `BFGS(NEB(...))` passes a `NEB` where `Atoms` is expected. ASE's `BFGS` does accept a `NEB` optimizable; likely a stub limitation, confirm.
- [x] ✅ syntax ×13 — cleared by py3.10 bump.

### atomix/core/screening.py — was 19 (now 2)
- [ ] 🔴 `~:219, :227, :229` `assignment` — `results` appears to be initialized as a `bool` then assigned a `list[ScreeningResult]`, then `enumerate`d. Reads like a real init bug; investigate.
- [x] ✅ syntax ×16 — cleared by py3.10 bump.

### atomix/analysis/trajectory.py — was 13 (now ~8)
- [ ] 🟢 `:101, :102, :276, :277` `var-annotated` — `n1, n2, ss_res, ss_tot` need annotations (e.g. `: float = 0.0`).
- [ ] 🟢 `:147, :155, :177, :211` `no-any-return` — numpy returns `Any`; cast or precise types.
- [ ] 🟡 `:28` `assignment` — `Atoms | list[Atoms]` assigned to a `list[Atoms]` var; structure-union root cause.
- [x] ✅ syntax ×4 — cleared by py3.10 bump.

### atomix/calculators/mlip.py — was 10 (now 1)
- [ ] 🟢 `:213` `no-any-return` — returns `Any` where `Calculator` declared; cast or annotate.
- [x] ✅ syntax ×9 — cleared by py3.10 bump.

### atomix/core/active_learning.py — was 10 (now 3)
- [ ] 🟢 `:468, :482, :506` `no-any-return` — numpy `Any` returns vs declared `ndarray`; cast.
- [x] ✅ syntax ×7 — cleared by py3.10 bump.

### atomix/ai/generator.py — was 9 (now 3)
- [ ] 🟢 `:99, :107` `no-any-return` (declared `str`), `:143` (declared `dict[str, Any]`) — JSON/lib returns `Any`; cast or annotate.
- [x] ✅ syntax ×6 — cleared by py3.10 bump.

### atomix/sites/surface.py — was 6 (now ~4)
- [ ] 🟢/🔴 `:95` `no-redef` — `sites` redefined (first at `:72`). Usually a rename; confirm it isn't shadowing by mistake.
- [ ] 🟢 `:109, :114, :259` `no-any-return` — numpy/ASE `Any` returns; cast.
- [x] ✅ syntax ×2 — cleared by py3.10 bump.

### atomix/analysis/adsorption.py — was 5 (now 1)
- [ ] 🟢 `:131` `dict-item` — dict value typed `list[float]` but built from `list[int]`; make the literals floats.
- [x] ✅ syntax ×4 — cleared by py3.10 bump.

### atomix/core/config.py — was 4 (now 0 under skip)
- [ ] 🟢 `:6` `import-untyped` (full-follow only) — `yaml` has no stubs. Add `types-PyYAML` to the `dev` extra, or a per-module `ignore`.
- [x] ✅ syntax ×3 — cleared by py3.10 bump.

### atomix/analysis/energy.py, atomix/calculators/cp2k.py, atomix/core/workflow.py
- [x] ✅ syntax-only modules — all cleared by py3.10 bump.

### atomix/sites/bulk.py — 1
- [ ] 🟢 `:71` `no-any-return` — ASE `Any` return vs declared `Atoms`; cast.

---

## Suggested order of attack (when the fix pass happens)

1. ✅ **Decide py3.9 vs py3.10+** — done (3.10+, `1dcc4d0`); cleared 103.
2. **Easy batch** (~32): `var-annotated` (5), `no-any-return` (11), `dict-item` (1), `vasp.py` append-container annotation (14 via 1 fix), `no-redef` (1). Mechanical, low-risk.
3. **Decisions** (~8): structure-read union type (cli/main.py + trajectory.py), `setup()`/`submit()` override signatures.
4. **Investigate** (~3): ~~`Outcar.converged`~~ ✅ done; `screening.py` bool/list init; `BFGS(NEB)`.
5. **Config:** add `follow_imports`/`exclude` (or pin mypy) so `mypy atomix` doesn't abort on the `tifffile` dependency.
