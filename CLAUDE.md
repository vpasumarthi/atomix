# CLAUDE.md - atomix

> **atomix**: Atomistic Modeling Interface for eXploration
> A natural-language-driven toolkit for ab initio / DFT / atomistic modeling workflows

## Project Vision

Reduce friction in computational materials research by enabling high-level ideation and execution through natural language interfaces, while maintaining rigorous scientific workflows underneath.

**Core philosophy**: The researcher should focus on *what* to simulate and *why*, not *how* to set up input files.

## Target User

- Computational catalysis researcher
- Primary tool: VASP (with extensibility to CP2K, QE)
- Uses ASE for structure manipulation
- Wants to integrate MLIPs (MACE, NequIP) for accelerated screening
- Values reproducibility and proper scientific workflow

## Core Features (Priority Order)

### Phase 1: Natural Language → Simulation Setup
```
User: "Relax Cu(111) 3x3 slab with 4 layers, bottom 2 fixed, PBE+D3, 400eV cutoff"
      ↓
atomix generates: POSCAR, INCAR, KPOINTS, POTCAR paths, job script
```

**Components needed:**
1. System prompt encoding VASP conventions and best practices
2. Local documentation (markdown) describing common patterns
3. Structure context parser (read existing POSCAR/CIF/traj)
4. Template library for common calculation types

### Phase 2: Calculation Management
- OOP architecture for calculation types (Static, Relax, AIMD, NEB, etc.)
- Job submission abstraction (SLURM, PBS, local)
- Status tracking and restart handling
- Output parsing and validation

### Phase 3: Analysis Pipeline
```
User: "Calculate O adsorption energy on all unique sites"
      ↓
atomix: runs workflow, parses outputs, returns formatted results
```

### Phase 4: MLIP Integration
- Drop-in calculator replacement (same workflow, swap DFT↔MLIP)
- Screening workflows (MLIP fast scan → DFT validation)
- Active learning data generation

## Architecture Principles

### 1. Calculator Agnostic
```python
# Same workflow interface regardless of backend
calc = VASPCalculator(...)      # DFT
calc = MACECalculator(...)      # MLIP
calc = CP2KCalculator(...)      # Alternative DFT

workflow.run(atoms, calc)       # Identical interface
```

### 2. OOP Class Hierarchy
```
BaseCalculation
├── StaticCalculation
├── RelaxCalculation
├── AIMDCalculation
│   ├── NVTCalculation
│   ├── NPTCalculation
│   └── NVECalculation
├── NEBCalculation
└── FrequencyCalculation

BaseAnalyzer
├── EnergyAnalyzer
├── TrajectoryAnalyzer
├── AdsorptionAnalyzer
└── DiffusionAnalyzer

BaseSite (for catalysis)
├── SurfaceSite
├── BulkSite
└── DefectSite
```

### 3. Natural Language Layer
- NOT fine-tuned models (too expensive, hard to maintain)
- System prompt + local docs (RAG without vector store)
- Structure-aware context injection
- Validation of generated scripts before execution

### 4. Delegate Low-Level I/O
atomix focuses on NL interface + workflow orchestration. Delegate to established libraries:
- **pymatgen**: VASP I/O (Incar, Poscar, Kpoints, Vasprun, Outcar)
- **ASE**: Structure manipulation, format conversion
- Don't reimplement what these libraries do well

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Structure handling | ASE | Industry standard, broad format support |
| DFT I/O | pymatgen | Robust VASP input/output parsing |
| Error recovery | custodian (optional) | Automatic job fixing/restarting |
| MLIP | MACE (mace-torch) | Pretrained foundation models available |
| LLM API | OpenAI / Anthropic / local (ollama) | Flexible provider |
| CLI | Click or Typer | Modern Python CLI |
| Config | YAML | Human-readable, ASE-compatible |

## File Structure (Proposed)

```
atomix/
├── atomix/
│   ├── __init__.py
│   ├── core/
│   │   ├── calculation.py      # Base calculation classes
│   │   ├── workflow.py         # Workflow orchestration
│   │   └── config.py           # Configuration handling
│   ├── calculators/
│   │   ├── vasp.py             # VASP-specific logic
│   │   ├── cp2k.py             # CP2K support
│   │   └── mlip.py             # MLIP wrappers
│   ├── analysis/
│   │   ├── energy.py           # Energy analysis
│   │   ├── trajectory.py       # MD trajectory analysis
│   │   └── adsorption.py       # Adsorption energy workflows
│   ├── sites/
│   │   ├── surface.py          # Surface site identification
│   │   └── bulk.py             # Bulk defect sites
│   ├── ai/
│   │   ├── generator.py        # NL → code generation
│   │   ├── prompts.py          # System prompts
│   │   └── docs/               # Local documentation for RAG
│   │       ├── vasp.md
│   │       ├── workflows.md
│   │       └── analysis.md
│   └── cli/
│       └── main.py             # Command-line interface
├── examples/
│   ├── cu111_relaxation/
│   └── o_adsorption/
├── tests/
├── docs/
├── CLAUDE.md                   # This file (for Claude Code)
├── pyproject.toml
└── README.md
```

## VASP Conventions (Domain Knowledge)

### Standard INCAR Settings by Calculation Type

**Relaxation:**
- IBRION = 2 (CG) or 1 (quasi-Newton)
- NSW = 100-500
- EDIFF = 1E-5, EDIFFG = -0.02
- ISIF = 2 (ions only) or 3 (cell + ions)

**AIMD (NVT):**
- IBRION = 0, SMASS = 0-3 (Nose-Hoover)
- NSW = steps, POTIM = 1-2 fs
- TEBEG = T_start, TEEND = T_end

**Static:**
- NSW = 0, IBRION = -1

**Common for catalysis:**
- IVDW = 11 (D3) or 12 (D3BJ) for dispersion
- LREAL = Auto for large cells
- NCORE = sqrt(total_cores) typically

### KPOINTS Conventions
- Relaxation: Gamma-centered, density ~30-50 per Å⁻¹
- AIMD: Gamma-only often sufficient for large supercells
- Band structure: Automatic with proper path

### POTCAR Selection
- Use PBE_54 or PBE_52 POTCARs
- Standard vs _pv vs _sv based on system
- Document POTCAR versions for reproducibility

## Catalysis-Specific Features

### Adsorption Energy Workflow
```
E_ads = E(slab+adsorbate) - E(slab) - E(adsorbate_gas)
```
- Automatic reference calculations
- Site enumeration (top, bridge, hollow)
- Coverage handling

### Surface Slab Generation
- Miller index specification
- Layer control with selective dynamics
- Vacuum and dipole corrections (LDIPOL, IDIPOL)

### Reaction Pathway
- NEB/CI-NEB setup
- Dimer method for transition states
- Frequency calculations for verification

## Development Guidelines

### Code Style
- Type hints throughout
- Docstrings (NumPy style)
- No unnecessary complexity
- Prefer composition over deep inheritance

### Testing
- Unit tests for core functions
- Integration tests with mock calculators
- Example workflows as tests

### Documentation
- Inline docs sufficient for AI context
- Examples over extensive prose
- Keep docs in sync with code

## What NOT to Build

- Custom DFT engine (use existing codes)
- Custom VASP I/O parsing (use pymatgen)
- Custom structure file writers (use ASE/pymatgen)
- Complex GUI (CLI + NL interface is enough)
- Database system (use existing: ASE db, pymatgen)
- Workflow scheduler (use existing: FireWorks, AiiDA if needed)

## Current Status

### Completed
- **Phase 1**: Natural Language → Simulation Setup ✓
  - NLGenerator with Anthropic/OpenAI support
  - System prompt with VASP conventions
  - CLI: `atomix generate`

- **Phase 2**: Calculation Management ✓
  - VASPCalculator with pymatgen I/O
  - Job submission (SLURM, PBS, local)
  - Output parsing and validation
  - Restart handling for incomplete calculations
  - CLI: `submit`, `status`, `analyze`, `validate`, `restart`

### Next: Phase 3 - Analysis Pipeline
Focus areas:
1. Adsorption energy workflows (E_ads calculation)
2. Site enumeration (top, bridge, hollow)
3. Reference calculations (gas-phase adsorbates)
4. Batch analysis across directories

Key files to implement/extend:
- `atomix/analysis/adsorption.py` - AdsorptionAnalyzer class
- `atomix/sites/surface.py` - SurfaceSite identification
- CLI command: `atomix adsorption`

## Getting Started Commands

```bash
# Project is set up - activate environment
conda activate atomix

# Run tests
pytest tests/

# Try CLI
atomix --help
atomix generate "Static calculation of bulk Cu" --dry-run
```

## References & Inspiration

- **gg (graph-gcbh)**: OOP modifier/sites pattern, NL→code architecture (conceptual inspiration, no code copying)
- **ASE**: Atoms/Calculator interface
- **pymatgen**: VASP input/output handling
- **custodian**: Job management patterns

---

*This document is read automatically by Claude Code to understand the project context.*
