# Development Notes

## Current State

Early scaffolding stage. Read-only segmented VASP inspection is the first
supported capability. VASP setup, calculation management, analysis, MLIP, and
natural-language scaffolding remain experimental and subject to API churn
through the `0.x` series. PyPI `0.2.0` is the first usable release; `0.1.1`
is only the historical name-claim placeholder.

## Release policy

The current top-level source declares no license. This does not technically
block a PyPI release; it means no general reuse or redistribution permission is
granted. Revisit licensing when the product boundary and likely proprietary
extensions are clearer. The historical PyPI placeholder remains a separate
MIT-licensed `0.1.1` artifact.

## Computation Backend

Consider atomate2 + jobflow-remote instead of building custom job management. Handles HPC submission, error recovery, provenance tracking out of the box. Key features for atomix:
- MDMaker / MultiMDMaker for AIMD (auto-splits long runs into walltime-safe chunks)
- AdsorptionMaker for surface adsorption calculations
- Custodian for auto-fixing VASP errors
- MongoDB provenance database (queryable results)
- Docs: https://materialsproject.github.io/atomate2/

## Reference: echemdb architecture (reviewed 2026-08-03)

Preprint: "echemdb: An Interfacial Electrochemistry Dataset of Fingerprint Cyclic Voltammograms for Single Crystal Electrodes", https://doi.org/10.26434/chemrxiv.15006518/v1 (posted 2026-07-24). Software credit is Hermann, Hörmann, Rüth, and Engstfeld. Reuter and Jacob are funding and review only, so do not read this as a Reuter-group architecture.

Closest shipped example of the warehouse pattern in an adjacent domain. Worth copying the layering, not the storage choice.

**How they split it.** Five repos, each separately installable, each with a Zenodo DOI:

| Layer | Repo |
|---|---|
| Schema | `metadata-schema` |
| Extraction | `svgdigitizer` |
| Store + API | `unitpackage` |
| Data | `electrochemistry-data` |
| Surface | `website` (MkDocs, generated in CI from the store) |

Schema and store ship as products independent of the data. The website is generated, never hand-maintained.

**Take:**
- Zenodo DOI per release. Citable without a journal, cheap now, and the one credibility mechanism available before adoption exists.
- Generated docs surface driven by the store, not maintained separately.
- Schema as its own versioned artifact rather than something embedded in the store code.

**Reject:** their storage choice. They use file-based frictionless Data Packages (CSV plus JSON metadata) with a Python API on top. Portable, git-diffable, no infrastructure, but no query engine. The load-bearing atomix use case is a query ("what Pt(111) calculations have I run at 400 eV across all projects"), which argues for DuckDB/parquet or SQLite instead. Note this is a real alternative being declined, not an option that was never considered.

**Positioning.** echemdb recovers *experimental* data from published literature. The preprint notes existing databases either extract summary values or store computationally generated data, and neither is their scope. The store for a researcher's own generated computational results is still empty, matching the earlier NovoMCP finding from a different direction.

**Cost lesson.** About 350 entries from about 90 publications, manually digitized, 14 authors, with LLMs added as a preliminary metadata cross-check to speed curation. Curation-as-product is expensive. Atomix writes at calculation time from metadata already in the INCAR, so it avoids that cost entirely. This is the structural argument for the atomix approach.

**Possible capability, if the store lands first:** compare a computed result against the echemdb reference for a given surface and electrolyte. One-sentence invocation, external data source, writes to the local store. Gated on the licensing question above.

## Publishing atomix

Recorded so the reasoning is not re-derived. Blocked on v0.2.0 either way.

**Decide what the paper is for first. The two goals have different answers.**

| Goal | Sequencing |
|---|---|
| Build a user base | Product first, paper documents it later |
| Credential for job applications | Preprint as soon as v0.2.0 is real, adoption irrelevant |

The second goal is the live one during the job search. A ChemRxiv or arXiv preprint plus a working PyPI package is a portfolio artifact on its own.

**Why the paper does not drive adoption here.** Method software (MACE, NequIP) is different: the method is the contribution, so the paper creates the users. Infrastructure and wrapper software (ASE, pymatgen, phonopy) gets papers that document and give a citation for software people already use. ASE and phonopy both had years of users before their canonical papers. Atomix is currently the second kind.

**Venues.** JOSS, SoftwareX, or J. Cheminformatics. JOSS fits and accepts solo-authored submissions, but screens against thin API clients and minor utility packages. A natural-language wrapper with nothing behind it is that profile, so JOSS needs real functionality first. A preprint has no such gate. Note the echemdb preprint uses Nature Scientific Data's template (Background & Summary, Usage Notes, Data/Code Availability), which is a data-descriptor format and not the model to copy for software.

**Solo authorship is not the risk.** Phonopy and spglib were effectively one author for years. Credibility comes from adoption, docs, tests, and the problem being real.

**The real risk is precedent.** PyCD was well engineered, tested, documented, and had a PhD and papers behind it, and still saw adoption by one student inside the group. A paper does not manufacture users.

**Cheaper distribution than a paper**, available at v0.2.0: aimd-tutorial as a showcase (already planned), Psi-k, and direct network posts.

## Cleanup

**Before the next release: keep one authoritative package configuration and
verify the built artifacts.**

The slim variant was created 2026-04-19 as a name-claim shortcut for the v0.1.1 PyPI release (PEP 541 reclaim, issue #9152). Two pyprojects now drift — every metadata change risks updating only one side.

Convergence steps:
1. [DONE 2026-06-02] Heavy deps are now imported lazily. CLI commands already deferred imports into command bodies; subpackage `__init__.py` files now use PEP 562 `__getattr__` so `import atomix.core` etc. no longer pull ase/numpy/pymatgen. Also fixed `generate --dry-run` to not import ase on the no-structure path.
2. [DONE 2026-06-02] Heavy deps moved out of `[project.dependencies]` (now `click` only) into a new `science` extra (ase, numpy, scipy, pymatgen, pyyaml, requests); `all = [science,mlip,llm,dev]`.
3. [DONE 2026-06-02] `python -m build` wheel installs with click-only deps; `atomix --help`/`info`/`--version` and `generate --dry-run` all work on the minimal install; `info` command ported to the root CLI; `__version__` synced to 0.1.1.
4. [PARTIAL 2026-06-02] Fresh-venv minimal install verified (only atomix+click present; all subpackages import; heavy-class access fails cleanly). Full test suite (110 passed) runs in the editable `atomix` conda env. A literal `pip install .[all]` in a clean venv has not been re-exercised.
5. [DONE 2026-08-08] Preserve `atomix-pypi-release/` as the historical MIT
   `0.1.1` source, mark it non-authoritative, and exclude it from current
   package discovery. README installation instructions now reference extras.
6. [DONE 2026-08-08] Add a built-wheel content check, fresh wheel-install CI
   smoke test, packaged AI reference documents, and the supported
   `inspect-vasp` command. The source version is `0.2.0`.

Do this as a discrete prep step, not bundled with v0.2.0 feature work.
