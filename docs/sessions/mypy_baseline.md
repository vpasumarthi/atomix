---
summary: "Triaged baseline of mypy errors (165 total) from the v0.x quality pass. Checklist grouped by module, tagged likely-easy vs needs-decision. No fixes applied yet."
read_when:
  - "starting a mypy / type-annotation cleanup task"
  - "deciding whether to keep Python 3.9 support"
  - "touching structure-reading return types (Atoms vs list[Atoms])"
---

# mypy baseline (triaged)

Generated during the mechanical quality pass on **2026-06-03**.

- **Command:** `mypy atomix` (uses `[tool.mypy]` in `pyproject.toml`: `python_version = "3.9"`, `warn_return_any = true`, `warn_unused_configs = true`, `ignore_missing_imports = true`)
- **Result:** `Found 165 errors in 16 files (checked 24 source files)`
- **mypy version:** 1.19.1
- **Status:** NOT fixed. This is a triage checklist only — several items need design decisions.

## Error-code summary

| Count | Code | Bucket | Notes |
|------:|------|--------|-------|
| 103 | `syntax` | **decision (1 global fix)** | `X \| Y` union syntax requires py3.10; config targets py3.9. One decision clears all 103. |
| 16 | `attr-defined` | mixed | 14 are one annotation fix in `vasp.py`; 2 are a likely real bug (`Outcar.converged`). |
| 15 | `no-any-return` | easy | `warn_return_any` firing on untyped lib returns (ASE/pymatgen/numpy). Cast or annotate. |
| 10 | `arg-type` | **decision** | Stems from `Atoms \| list[Atoms]` structure-read type flowing into `Atoms`-only APIs. |
| 7 | `override` | **decision** | LSP violations: `setup()` return type, `submit()` signature vs base class. |
| 5 | `var-annotated` | easy | Add an explicit annotation on an accumulator/initial value. |
| 3 | `union-attr` | **decision** | Same root cause as `arg-type` (structure union). |
| 3 | `assignment` | investigate | 2 in `screening.py` look like a real bool/list mixup. |
| 1 | `no-redef` | easy/investigate | `sites` redefined in `surface.py`. |
| 1 | `import-untyped` | easy | Add `types-PyYAML` to dev deps (or per-module ignore). |
| 1 | `dict-item` | easy | `list[int]` vs `list[float]` — make the literals floats. |

## The big lever: PEP 604 unions vs Python 3.9 target

103 of 165 errors are `X | Y syntax for unions requires Python 3.10`. The code uses modern `int | None` annotations everywhere, but `pyproject.toml` pins `requires-python = ">=3.9"` and the mypy config sets `python_version = "3.9"`. At runtime this is currently harmless (annotations aren't evaluated in these positions on the 3.11 dev env), but it is a real portability claim mismatch.

**This is a single decision, not 103 fixes.** Pick one:

1. **Drop 3.9, target 3.10+** — bump `requires-python` and `python_version` (and the `py39` ruff target + 3.9 classifiers). Clears all 103 with no source edits. Simplest if 3.9 support isn't a hard requirement.
2. **Keep 3.9, add `from __future__ import annotations`** to every module that uses `|` unions. Clears all 103 and keeps 3.9 runtime-safe. Mechanical but touches ~16 files.
3. **Keep 3.9, rewrite to `Optional`/`Union`** — most invasive, not recommended.

Recommendation: decide #1 vs #2 first; it removes ~62% of the noise and makes the remaining ~62 errors readable. Tracked once below rather than per-line (line numbers in the raw list are exhaustive).

---

## Per-module checklist

Legend: 🟢 likely-easy (unambiguous mechanical fix) · 🟡 needs-decision (API/design) · 🔴 investigate (possible real bug)

### atomix/cli/main.py — 26 (15 syntax, 7 arg-type, 3 union-attr, 1 var-annotated)
- [ ] 🟡 `read_structure()` returns `Atoms | list[Atoms]`; downstream APIs want `Atoms`. This drives **all** of: union-attr `:64, :662, :1064` (`get_chemical_formula`), arg-type `:86, :113, :666, :707, :812, :1067, :1106`. Decide whether to narrow the return type, raise on multi-frame, or branch on the list case. One design fix resolves 10 errors here.
- [ ] 🟢 `:335` `var-annotated` — `max_force` needs an explicit annotation (e.g. `max_force: float = 0.0`).
- [ ] (syntax ×15 — covered by the global py3.9 decision)

### atomix/core/jobs.py — 18 (16 syntax, 2 override)
- [ ] 🟡 `:188, :353` `override` — `SlurmSubmitter.submit()` / `LocalSubmitter.submit()` signatures differ from base `JobSubmitter.submit(self, **kwargs)`. Decide a stable base signature (keep `**kwargs`, or make the base abstract with the concrete params).
- [ ] (syntax ×16 — global py3.9 decision)

### atomix/calculators/vasp.py — 20 (16 attr-defined, 3 syntax, 1 arg-type)
- [ ] 🟢 `:408–:445` (14×) `"object" has no attribute "append"` — a container is inferred as `object`. Annotate it at its initialization (e.g. `errors: list[str] = []`) and all 14 clear.
- [ ] 🔴 `:243, :304` `"Outcar" has no attribute "converged"` — pymatgen's `Outcar` has no `converged` attribute. Likely a real bug (intended `Vasprun.converged`?). Verify against current pymatgen API.
- [ ] 🟡 `:83` `arg-type` — `Poscar()` got `Structure | Molecule`; expects `Structure | IStructure`. Guard against `Molecule` or narrow the type.
- [ ] (syntax ×3 — global py3.9 decision)

### atomix/core/calculation.py — 19 (13 syntax, 5 override, 1 arg-type)
- [ ] 🟡 `:134, :216, :350, :456, :606` `override` — subclass `setup()` returns `dict[str, Path]` but `BaseCalculation.setup()` is annotated `-> None`. Fix the base class return annotation to match (likely `-> dict[str, Path]`). One base-class edit resolves all 5.
- [ ] 🔴 `:568` `arg-type` — `BFGS(NEB(...))` passes a `NEB` where `Atoms` is expected. ASE's `BFGS` does accept a `NEB` optimizable; likely a stub limitation, but confirm — could be a real misuse.
- [ ] (syntax ×13 — global py3.9 decision)

### atomix/core/screening.py — 19 (16 syntax, 2 assignment, 1 arg-type)
- [ ] 🔴 `:219, :227, :229` — `results` appears to be initialized as a `bool` then assigned a `list[ScreeningResult]`, then `enumerate`d. Reads like a real init bug (wrong default). Investigate the variable around `:219`.
- [ ] (syntax ×16 — global py3.9 decision)

### atomix/analysis/trajectory.py — 13 (4 syntax, 4 no-any-return, 4 var-annotated, 1 assignment)
- [ ] 🟢 `:101, :102, :276, :277` `var-annotated` — `n1, n2, ss_res, ss_tot` need annotations (accumulators; e.g. `: float = 0.0`).
- [ ] 🟢 `:147, :155, :177, :211` `no-any-return` — numpy returns `Any`; add `cast` or precise return types.
- [ ] 🟡 `:28` `assignment` — `Atoms | list[Atoms]` assigned to a `list[Atoms]` var; same structure-union root cause as cli/main.py.
- [ ] (syntax ×4 — global py3.9 decision)

### atomix/calculators/mlip.py — 10 (9 syntax, 1 no-any-return)
- [ ] 🟢 `:213` `no-any-return` — returns `Any` where `Calculator` declared; cast or annotate.
- [ ] (syntax ×9 — global py3.9 decision)

### atomix/core/active_learning.py — 10 (7 syntax, 3 no-any-return)
- [ ] 🟢 `:468, :482, :506` `no-any-return` — numpy `Any` returns vs declared `ndarray`; cast.
- [ ] (syntax ×7 — global py3.9 decision)

### atomix/ai/generator.py — 9 (6 syntax, 3 no-any-return)
- [ ] 🟢 `:99, :107` `no-any-return` (declared `str`), `:143` (declared `dict[str, Any]`) — JSON/lib returns `Any`; cast or annotate.
- [ ] (syntax ×6 — global py3.9 decision)

### atomix/sites/surface.py — 6 (3 no-any-return, 2 syntax, 1 no-redef)
- [ ] 🟢/🔴 `:95` `no-redef` — `sites` redefined (first at `:72`). Usually a rename; confirm it isn't shadowing an earlier value by mistake.
- [ ] 🟢 `:109, :114, :259` `no-any-return` — numpy/ASE `Any` returns; cast or annotate.
- [ ] (syntax ×2 — global py3.9 decision)

### atomix/analysis/adsorption.py — 5 (4 syntax, 1 dict-item)
- [ ] 🟢 `:131` `dict-item` — dict value typed `list[float]` but built from `list[int]`; make the literals floats (`[0.0, ...]`).
- [ ] (syntax ×4 — global py3.9 decision)

### atomix/core/config.py — 4 (3 syntax, 1 import-untyped)
- [ ] 🟢 `:6` `import-untyped` — `yaml` has no stubs. Add `types-PyYAML` to the `dev` extra (cleanest), or a per-module `ignore`.
- [ ] (syntax ×3 — global py3.9 decision)

### atomix/analysis/energy.py — 3 (3 syntax)
- [ ] (syntax ×3 — global py3.9 decision)

### atomix/calculators/cp2k.py — 1 (1 syntax)
- [ ] (syntax ×1 — global py3.9 decision)

### atomix/core/workflow.py — 1 (1 syntax)
- [ ] (syntax ×1 — global py3.9 decision)

### atomix/sites/bulk.py — 1 (1 no-any-return)
- [ ] 🟢 `:71` `no-any-return` — ASE `Any` return vs declared `Atoms`; cast.

---

## Suggested order of attack (when the fix pass happens)

1. **Decide py3.9 vs py3.10+** and apply (option #1 or #2 above) → clears 103.
2. **Easy batch** (~26): `var-annotated` (5), `no-any-return` (15), `import-untyped` (1), `dict-item` (1), `vasp.py` append-container annotation (14 errors via 1 fix), `no-redef` (1). Mechanical, low-risk.
3. **Decisions** (~10): structure-read union type (cli/main.py + trajectory.py), `setup()`/`submit()` override signatures.
4. **Investigate** (~5): `Outcar.converged`, `screening.py` bool/list init, `BFGS(NEB)`.
</content>
</invoke>
