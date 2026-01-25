# VASP Reference Documentation

## INCAR Parameters Quick Reference

### Electronic Relaxation

| Parameter | Values | Description |
|-----------|--------|-------------|
| ENCUT | 300-600 | Plane-wave cutoff (eV) |
| EDIFF | 1E-4 to 1E-6 | Electronic convergence |
| PREC | Low/Normal/Accurate | Precision level |
| ALGO | Normal/Fast/VeryFast | Electronic minimization |
| ISMEAR | -5/0/1/2 | Smearing method |
| SIGMA | 0.05-0.2 | Smearing width (eV) |

### Ionic Relaxation

| Parameter | Values | Description |
|-----------|--------|-------------|
| IBRION | -1/0/1/2/3 | Relaxation algorithm |
| NSW | 0-1000 | Max ionic steps |
| EDIFFG | -0.01 to -0.05 | Force convergence (eV/Å) |
| ISIF | 0-7 | Stress/strain relaxation |
| POTIM | 0.1-2.0 | Time step or step width |

### IBRION Values

- -1: No ionic update (static)
- 0: Molecular dynamics
- 1: Quasi-Newton (RMM-DIIS)
- 2: Conjugate gradient
- 3: Damped MD

### ISIF Values

- 0: Ions only, no cell shape/volume
- 2: Ions only (default for IBRION=1,2)
- 3: Ions + cell shape + volume
- 4: Ions + cell shape, fixed volume

### Dispersion Corrections

| IVDW | Method |
|------|--------|
| 10 | DFT-D2 |
| 11 | DFT-D3 (zero damping) |
| 12 | DFT-D3-BJ (Becke-Johnson) |

## Common Workflows

### Surface Slab Relaxation

```
IBRION = 2
NSW = 100
EDIFFG = -0.02
ISIF = 2
LDIPOL = .TRUE.
IDIPOL = 3
```

### NEB Transition State

```
IBRION = 3  # or 1
POTIM = 0
ICHAIN = 0
IMAGES = 5-9
SPRING = -5
LCLIMB = .TRUE.  # CI-NEB
```

### Frequency Calculation

```
IBRION = 5  # or 6 for finite differences
NSW = 1
NFREE = 2
POTIM = 0.015
```
