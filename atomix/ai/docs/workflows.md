# Common Atomistic Workflows

## Catalysis Workflows

### Adsorption Energy Calculation

1. Relax clean surface slab
2. Identify adsorption sites (top, bridge, hollow)
3. Place adsorbate at each site
4. Relax slab+adsorbate systems
5. Calculate: E_ads = E(slab+ads) - E(slab) - E(gas)

### Reaction Pathway (NEB)

1. Relax initial state
2. Relax final state
3. Generate interpolated images
4. Run NEB or CI-NEB
5. Verify transition state with frequency calculation

### Surface Phase Diagram

1. Calculate surface energies at various coverages
2. Include gas phase references (T, p dependence)
3. Plot surface free energy vs. chemical potential

## Structure Preparation

### Slab Generation

```python
from ase.build import fcc111

# Create Cu(111) slab
slab = fcc111('Cu', size=(3, 3, 4), vacuum=10.0)

# Fix bottom layers
from ase.constraints import FixAtoms
c = FixAtoms(indices=[i for i in range(len(slab))
                      if slab[i].position[2] < slab.cell[2,2]/2])
slab.set_constraint(c)
```

### Common Miller Indices

| Surface | Type | Coordination |
|---------|------|--------------|
| (111) | Close-packed | 9 (fcc), 8 (bcc) |
| (100) | Square | 8 (fcc), 4 (bcc) |
| (110) | Rectangular | 7 (fcc), 6 (bcc) |

## AIMD Workflows

### Equilibration Protocol

1. NVT at target T for 1-2 ps
2. Check temperature stability
3. Production run (5-20 ps)
4. Post-process: RDF, MSD, coordination

### Active Learning Loop

1. Run MLIP MD at elevated T
2. Extract diverse configurations
3. Single-point DFT on selected frames
4. Retrain potential
5. Repeat until converged
