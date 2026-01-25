# Analysis Methods Reference

## Energy Analysis

### Formation Energy

```
E_f = E_total - Σ(n_i × μ_i)
```

Where n_i is count of element i and μ_i is chemical potential (often elemental reference).

### Adsorption Energy

```
E_ads = E(surface+adsorbate) - E(surface) - E(adsorbate_gas)
```

Negative = exothermic (favorable adsorption)

### Reaction Energy

```
ΔE = Σ E(products) - Σ E(reactants)
```

## Trajectory Analysis

### Radial Distribution Function (RDF)

g(r) describes probability of finding atom at distance r from reference.

- First peak position → bond length
- Peak height → coordination
- Long-range → 1 (bulk liquid/gas)

### Mean Squared Displacement (MSD)

```
MSD(t) = <|r(t) - r(0)|²>
```

- Linear regime → diffusion
- D = MSD / (2 × d × t) where d = dimensionality

### Coordination Number

```
CN = 4π ∫₀^r_cut r² ρ g(r) dr
```

## Thermodynamic Properties

### Vibrational Free Energy

```
F_vib = Σ [ℏω/2 + k_B T ln(1 - exp(-ℏω/k_B T))]
```

### Surface Free Energy

```
γ(T, p) = (1/2A) [G_slab - N × g_bulk]
```

For slabs with adsorbates:
```
γ(T, μ) = (1/A) [G_slab+ads - N_surf × g_bulk - N_ads × μ_ads]
```

## Output Files

### VASP Outputs

| File | Contents |
|------|----------|
| OUTCAR | Full calculation log |
| CONTCAR | Final structure |
| OSZICAR | Electronic steps |
| vasprun.xml | All data (parseable) |
| XDATCAR | MD trajectory |
| DOSCAR | Density of states |
| EIGENVAL | Band energies |

### Parsing with ASE

```python
from ase.io import read

# Final structure
atoms = read('CONTCAR')

# Trajectory
traj = read('XDATCAR', index=':')

# With calculator results
atoms = read('vasprun.xml')
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```
