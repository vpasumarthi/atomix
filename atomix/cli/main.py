"""Main CLI entry point for atomix."""

from pathlib import Path

import click


@click.group()
@click.version_option()
def cli() -> None:
    """atomix: Atomistic Modeling Interface for eXploration.

    Natural language driven toolkit for DFT and atomistic modeling.
    """
    pass


@cli.command()
@click.argument("prompt")
@click.option("--structure", "-s", type=click.Path(exists=True), help="Input structure file")
@click.option("--output", "-o", type=click.Path(), default=".", help="Output directory")
@click.option("--provider", "-p", default="anthropic", help="LLM provider")
@click.option("--model", "-m", default=None, help="Model name (provider-specific)")
@click.option("--dry-run", is_flag=True, help="Show what would be generated without calling LLM")
def generate(
    prompt: str,
    structure: str | None,
    output: str,
    provider: str,
    model: str | None,
    dry_run: bool,
) -> None:
    """Generate simulation inputs from natural language.

    PROMPT is the natural language description of the desired simulation.

    Example:
        atomix generate "Relax Cu(111) 3x3 slab, 4 layers, PBE+D3"
    """
    from ase.io import read as ase_read

    from atomix.ai.generator import NLGenerator
    from atomix.calculators.vasp import VASPCalculator

    click.echo(f"Generating simulation setup for: {prompt}")

    # Load structure if provided
    atoms = None
    if structure:
        atoms = ase_read(structure)
        click.echo(f"Loaded structure: {len(atoms)} atoms, {atoms.get_chemical_formula()}")

    if dry_run:
        click.echo("\n[Dry run - would call LLM with above prompt]")
        click.echo(f"Provider: {provider}")
        click.echo(f"Output directory: {output}")
        return

    # Initialize generator
    gen = NLGenerator(provider=provider, model=model) if model else NLGenerator(provider=provider)

    # Load documentation for RAG context
    docs_path = Path(__file__).parent.parent / "ai" / "docs"
    if docs_path.exists():
        gen.load_docs(docs_path)

    # Generate parameters from natural language
    try:
        result = gen.generate(prompt, atoms=atoms)
    except Exception as e:
        click.echo(f"Error calling LLM: {e}", err=True)
        raise click.Abort()

    # Extract parameters
    incar = result.get("incar", {})
    kpoints = result.get("kpoints", {"type": "gamma", "grid": [1, 1, 1]})
    calc_type = result.get("calc_type", "static")
    warnings = result.get("warnings", [])

    # Check if we need to generate structure
    struct_info = result.get("structure", {})
    if struct_info.get("action") == "generate" and atoms is None:
        click.echo(
            f"Note: Structure generation requested: {struct_info.get('description', 'N/A')}",
            err=True,
        )
        click.echo("Please provide a structure file with -s/--structure", err=True)
        raise click.Abort()

    if atoms is None:
        click.echo("Error: No structure provided and none could be generated", err=True)
        raise click.Abort()

    # Write VASP inputs
    calculator = VASPCalculator(output, **incar)
    files = calculator.write_inputs(atoms, calc_type, kpoints=kpoints)

    # Report results
    click.echo(f"\nSetup complete in {output}/")
    click.echo(f"  Calculation type: {calc_type}")
    for name, path in files.items():
        click.echo(f"  - {name}")

    if warnings:
        click.echo("\nWarnings:")
        for w in warnings:
            click.echo(f"  - {w}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--scheduler", "-S", default="slurm", help="Scheduler: slurm, pbs, local")
@click.option("--partition", "-p", default=None, help="SLURM partition or PBS queue")
@click.option("--account", "-A", default=None, help="Account/project name")
@click.option("--nodes", "-n", default=1, help="Number of nodes")
@click.option("--ntasks", "-t", default=32, help="Tasks per node")
@click.option("--time", "-T", default="24:00:00", help="Wall time")
@click.option("--vasp-command", default="vasp_std", help="VASP executable")
@click.option("--modules", "-m", multiple=True, help="Modules to load")
@click.option("--dry-run", is_flag=True, help="Generate script but don't submit")
def submit(
    directory: str,
    scheduler: str,
    partition: str | None,
    account: str | None,
    nodes: int,
    ntasks: int,
    time: str,
    vasp_command: str,
    modules: tuple[str, ...],
    dry_run: bool,
) -> None:
    """Submit calculation to job scheduler.

    DIRECTORY is the path containing VASP input files.

    Example:
        atomix submit ./cu_relax -S slurm -p normal -n 2 -t 64
    """
    from atomix.core.jobs import get_submitter

    click.echo(f"Submitting job from: {directory}")

    # Build submitter options
    options: dict = {
        "nodes": nodes,
        "ntasks_per_node": ntasks,
        "time": time,
    }
    if partition:
        if scheduler == "slurm":
            options["partition"] = partition
        else:
            options["queue"] = partition
    if account:
        options["account"] = account

    submitter = get_submitter(scheduler, directory, **options)

    # Generate script
    if hasattr(submitter, "generate_script"):
        script = submitter.generate_script(
            job_name=Path(directory).name or "vasp_job",
            vasp_command=vasp_command,
            modules=list(modules) if modules else None,
        )
        click.echo("\nGenerated script:")
        click.echo("-" * 40)
        click.echo(script)
        click.echo("-" * 40)

    if dry_run:
        click.echo("\n[Dry run - script not submitted]")
        return

    # Submit
    try:
        job_id = submitter.submit(
            job_name=Path(directory).name or "vasp_job",
            vasp_command=vasp_command,
            modules=list(modules) if modules else None,
        )
        click.echo(f"\nJob submitted: {job_id}")
    except Exception as e:
        click.echo(f"Submission failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--job-id", "-j", default=None, help="Job ID to check (for scheduler status)")
@click.option("--scheduler", "-S", default="slurm", help="Scheduler: slurm, pbs, local")
def status(directory: str, job_id: str | None, scheduler: str) -> None:
    """Check status of calculations in directory.

    Without --job-id, checks for VASP output files.
    With --job-id, queries the job scheduler.
    """
    from atomix.core.jobs import get_submitter

    click.echo(f"Checking status in: {directory}")

    dir_path = Path(directory)

    # Check for VASP files
    poscar = dir_path / "POSCAR"
    incar = dir_path / "INCAR"
    outcar = dir_path / "OUTCAR"
    oszicar = dir_path / "OSZICAR"

    if not poscar.exists():
        click.echo("  No POSCAR found - not a VASP calculation directory")
        return

    click.echo(f"  POSCAR: {'found' if poscar.exists() else 'missing'}")
    click.echo(f"  INCAR:  {'found' if incar.exists() else 'missing'}")

    # Check OUTCAR for completion
    if outcar.exists():
        content = outcar.read_text()
        if "reached required accuracy" in content or "General timing" in content:
            click.echo("  Status: COMPLETED")
        elif "Error" in content or "VERY BAD NEWS" in content:
            click.echo("  Status: FAILED")
        else:
            click.echo("  Status: RUNNING or INCOMPLETE")

        # Extract some info from OSZICAR if available
        if oszicar.exists():
            lines = oszicar.read_text().strip().split("\n")
            if lines:
                last_line = lines[-1]
                click.echo(f"  Last step: {last_line[:60]}...")
    else:
        click.echo("  OUTCAR: not found")
        click.echo("  Status: NOT STARTED or QUEUED")

    # Check scheduler status if job_id provided
    if job_id:
        submitter = get_submitter(scheduler, directory)
        sched_status = submitter.status(job_id)
        click.echo(f"  Scheduler ({scheduler}): {sched_status}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--type", "-t", "calc_type", default="summary", help="Analysis type: summary, energy, forces, trajectory")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def analyze(directory: str, calc_type: str, as_json: bool) -> None:
    """Analyze calculation outputs.

    Analysis types:
      summary    - Overview of calculation results (default)
      energy     - Energy values and convergence
      forces     - Force analysis and max force
      trajectory - Relaxation/MD trajectory info
    """
    import json

    from atomix.calculators.vasp import VASPCalculator

    calc = VASPCalculator(directory)

    # Check if outputs exist
    outcar = Path(directory) / "OUTCAR"
    if not outcar.exists():
        click.echo(f"No OUTCAR found in {directory}", err=True)
        raise click.Abort()

    # Parse outputs
    try:
        results = calc.read_outputs()
    except Exception as e:
        click.echo(f"Error parsing outputs: {e}", err=True)
        raise click.Abort()

    if as_json:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if hasattr(value, "tolist"):
                json_results[key] = value.tolist()
            elif key == "atoms" and value is not None:
                json_results[key] = {
                    "formula": value.get_chemical_formula(),
                    "n_atoms": len(value),
                }
            elif key == "trajectory":
                json_results[key] = [
                    {"formula": a.get_chemical_formula(), "n_atoms": len(a)}
                    for a in value
                ]
            else:
                json_results[key] = value
        click.echo(json.dumps(json_results, indent=2))
        return

    # Display based on analysis type
    if calc_type == "summary":
        click.echo(f"\n=== Calculation Summary: {directory} ===\n")
        click.echo(f"  Converged: {'Yes' if results['converged'] else 'No'}")
        if results["energy"] is not None:
            click.echo(f"  Energy: {results['energy']:.6f} eV")
        click.echo(f"  Ionic steps: {results['n_steps']}")

        if results["atoms"] is not None:
            atoms = results["atoms"]
            click.echo(f"  Final structure: {len(atoms)} atoms, {atoms.get_chemical_formula()}")

        if results["forces"] is not None:
            import numpy as np
            max_force = np.max(np.abs(results["forces"]))
            click.echo(f"  Max force: {max_force:.4f} eV/Å")

        if results["errors"]:
            click.echo("\n  Errors:")
            for err in results["errors"]:
                click.echo(f"    - {err}")

        if results["warnings"]:
            click.echo("\n  Warnings:")
            for warn in results["warnings"]:
                click.echo(f"    - {warn}")

    elif calc_type == "energy":
        click.echo(f"\n=== Energy Analysis: {directory} ===\n")
        if results["energy"] is not None:
            click.echo(f"  Total energy: {results['energy']:.6f} eV")
            if results["atoms"] is not None:
                e_per_atom = results["energy"] / len(results["atoms"])
                click.echo(f"  Energy/atom: {e_per_atom:.6f} eV")
        else:
            click.echo("  No energy found in outputs")

    elif calc_type == "forces":
        click.echo(f"\n=== Force Analysis: {directory} ===\n")
        if results["forces"] is not None:
            import numpy as np
            forces = results["forces"]
            force_mags = np.linalg.norm(forces, axis=1)
            click.echo(f"  Number of atoms: {len(forces)}")
            click.echo(f"  Max force: {np.max(force_mags):.6f} eV/Å")
            click.echo(f"  Mean force: {np.mean(force_mags):.6f} eV/Å")
            click.echo(f"  RMS force: {np.sqrt(np.mean(force_mags**2)):.6f} eV/Å")

            # Show atoms with largest forces
            max_idx = np.argmax(force_mags)
            click.echo(f"\n  Largest force on atom {max_idx}: {force_mags[max_idx]:.6f} eV/Å")
        else:
            click.echo("  No forces found in outputs")

    elif calc_type == "trajectory":
        click.echo(f"\n=== Trajectory Analysis: {directory} ===\n")
        traj = results.get("trajectory", [])
        if traj:
            click.echo(f"  Number of frames: {len(traj)}")
            # Show energy progression if available
            energies = []
            for atoms in traj:
                if atoms.calc is not None:
                    try:
                        energies.append(atoms.get_potential_energy())
                    except Exception:
                        pass
            if energies:
                click.echo(f"  Initial energy: {energies[0]:.6f} eV")
                click.echo(f"  Final energy: {energies[-1]:.6f} eV")
                click.echo(f"  Energy change: {energies[-1] - energies[0]:.6f} eV")
        else:
            click.echo("  No trajectory found (try reading XDATCAR or vasprun.xml)")

    else:
        click.echo(f"Unknown analysis type: {calc_type}", err=True)
        click.echo("Available types: summary, energy, forces, trajectory")


@cli.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def validate(directory: str, as_json: bool) -> None:
    """Validate calculation outputs and check for errors.

    Checks for common VASP errors and provides recommendations.
    """
    import json

    from atomix.calculators.vasp import VASPCalculator

    calc = VASPCalculator(directory)
    result = calc.validate_outputs()

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"\n=== Validation: {directory} ===\n")
    click.echo(f"  Status: {result['status'].upper()}")
    click.echo(f"  Valid: {'Yes' if result['valid'] else 'No'}")
    click.echo(f"  Can restart: {'Yes' if result['can_restart'] else 'No'}")

    if "n_ionic_steps" in result:
        click.echo(f"  Ionic steps completed: {result['n_ionic_steps']}")

    if result["errors"]:
        click.echo("\n  Errors:")
        for err in result["errors"]:
            click.echo(f"    - {err}")

    if result["warnings"]:
        click.echo("\n  Recommendations:")
        for warn in result["warnings"]:
            click.echo(f"    - {warn}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--type", "-t", "calc_type", default="relax", help="Calculation type for INCAR")
def restart(directory: str, calc_type: str) -> None:
    """Set up calculation for restart from incomplete run.

    Copies CONTCAR to POSCAR and sets ISTART/ICHARG based on
    available WAVECAR/CHGCAR files.
    """
    from atomix.calculators.vasp import VASPCalculator

    calc = VASPCalculator(directory)

    # First validate
    validation = calc.validate_outputs()
    if validation["status"] == "completed" and validation["valid"]:
        click.echo("Calculation already completed successfully.")
        return

    if not validation["can_restart"]:
        click.echo("Cannot restart: no valid CONTCAR found.", err=True)
        raise click.Abort()

    try:
        files = calc.setup_restart(calc_type)
        click.echo(f"Restart setup complete in {directory}/")
        for name, path in files.items():
            click.echo(f"  - Updated {name}")

        # Show what restart settings were applied
        incar_path = Path(directory) / "INCAR"
        if incar_path.exists():
            content = incar_path.read_text()
            if "ISTART" in content:
                click.echo("\n  WAVECAR restart enabled (ISTART=1)")
            if "ICHARG" in content:
                click.echo("  CHGCAR restart enabled (ICHARG=1)")

    except FileNotFoundError as e:
        click.echo(f"Restart failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="atomix.yaml", help="Config file path")
def init(output: str) -> None:
    """Initialize atomix configuration file."""
    from atomix.core.config import Config

    config = Config()
    config.save(output)
    click.echo(f"Created configuration file: {output}")


@cli.command()
@click.argument("directories", nargs=-1, type=click.Path(exists=True))
@click.option("--slab", "-s", type=click.Path(exists=True), help="Slab calculation directory")
@click.option("--slab-energy", "-E", type=float, help="Slab energy (eV) if known")
@click.option("--gas-ref", "-g", type=click.Path(exists=True), help="Gas reference calculation directory")
@click.option("--gas-energy", "-G", type=float, help="Gas reference energy (eV) if known")
@click.option("--adsorbate", "-a", default="O", help="Adsorbate species name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def adsorption(
    directories: tuple[str, ...],
    slab: str | None,
    slab_energy: float | None,
    gas_ref: str | None,
    gas_energy: float | None,
    adsorbate: str,
    as_json: bool,
) -> None:
    """Calculate adsorption energies.

    E_ads = E(slab+adsorbate) - E(slab) - E(gas_reference)

    Provide calculation directories for slab+adsorbate systems.
    Reference energies can be specified directly or read from directories.

    Examples:
        # Single calculation with explicit references
        atomix adsorption ./o_on_cu -E -123.45 -G -4.93

        # Batch calculations
        atomix adsorption ./site_* -s ./clean_slab -g ./o_gas -a O

        # Multiple sites with JSON output
        atomix adsorption ./top ./bridge ./hollow -s ./slab -G -4.93 --json
    """
    import json

    from atomix.analysis.adsorption import AdsorptionAnalyzer
    from atomix.calculators.vasp import VASPCalculator

    # Get slab energy
    if slab_energy is None:
        if slab is None:
            click.echo("Error: Must provide either --slab directory or --slab-energy", err=True)
            raise click.Abort()
        calc = VASPCalculator(slab)
        try:
            slab_energy = calc.get_energy()
            if slab_energy is None:
                click.echo(f"Error: Could not read energy from {slab}", err=True)
                raise click.Abort()
        except Exception as e:
            click.echo(f"Error reading slab energy: {e}", err=True)
            raise click.Abort()

    # Get gas reference energy
    if gas_energy is None:
        if gas_ref is None:
            click.echo("Error: Must provide either --gas-ref directory or --gas-energy", err=True)
            raise click.Abort()
        calc = VASPCalculator(gas_ref)
        try:
            gas_energy = calc.get_energy()
            if gas_energy is None:
                click.echo(f"Error: Could not read energy from {gas_ref}", err=True)
                raise click.Abort()
        except Exception as e:
            click.echo(f"Error reading gas reference energy: {e}", err=True)
            raise click.Abort()

    # Create analyzer
    analyzer = AdsorptionAnalyzer(
        slab_energy=slab_energy,
        gas_references={adsorbate: gas_energy},
    )

    if not directories:
        click.echo("No calculation directories provided", err=True)
        raise click.Abort()

    # Calculate adsorption energies
    results = analyzer.batch_adsorption_energies(
        list(directories),
        adsorbate=adsorbate,
    )

    if as_json:
        click.echo(json.dumps(results, indent=2))
        return

    # Display as table
    click.echo(f"\n=== Adsorption Energies ({adsorbate}) ===\n")
    click.echo(f"  Slab energy: {slab_energy:.6f} eV")
    click.echo(f"  Gas reference ({adsorbate}): {gas_energy:.6f} eV")
    click.echo()
    click.echo(f"  {'Directory':<30} {'Energy (eV)':<14} {'E_ads (eV)':<12} {'Status'}")
    click.echo("  " + "-" * 70)

    for r in results:
        name = r["name"][:28] if len(r["name"]) > 28 else r["name"]
        if r["energy"] is not None:
            energy_str = f"{r['energy']:.4f}"
            e_ads_str = f"{r['e_ads']:.4f}" if r["e_ads"] is not None else "N/A"
            status = "OK" if r["converged"] else "not converged"
        else:
            energy_str = "N/A"
            e_ads_str = "N/A"
            status = r.get("error", "error")[:20]

        click.echo(f"  {name:<30} {energy_str:<14} {e_ads_str:<12} {status}")

    # Summary statistics
    valid_e_ads = [r["e_ads"] for r in results if r["e_ads"] is not None]
    if valid_e_ads:
        click.echo()
        click.echo(f"  Min E_ads: {min(valid_e_ads):.4f} eV")
        click.echo(f"  Max E_ads: {max(valid_e_ads):.4f} eV")
        if len(valid_e_ads) > 1:
            import numpy as np
            click.echo(f"  Mean E_ads: {np.mean(valid_e_ads):.4f} eV")


@cli.command()
@click.argument("structure", type=click.Path(exists=True))
@click.option("--height", "-h", default=1.5, help="Height above surface for sites (Angstrom)")
@click.option("--symprec", "-s", default=0.1, help="Symmetry precision for unique sites (Angstrom)")
@click.option("--adsorbate", "-a", default=None, help="Adsorbate to place (e.g., O, CO, OH)")
@click.option("--output", "-o", type=click.Path(), help="Output directory for structures")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def sites(
    structure: str,
    height: float,
    symprec: float,
    adsorbate: str | None,
    output: str | None,
    as_json: bool,
) -> None:
    """Find adsorption sites on a surface slab.

    Identifies unique top, bridge, and hollow (fcc/hcp) sites
    using Delaunay triangulation of surface atoms.

    Examples:
        # Find sites on a slab
        atomix sites ./POSCAR

        # Find sites and generate structures with O adsorbate
        atomix sites ./slab.cif -a O -o ./site_structures/

        # Output site info as JSON
        atomix sites ./POSCAR --json
    """
    import json
    from pathlib import Path

    from ase.io import read as ase_read
    from ase.io import write as ase_write

    from atomix.sites.surface import add_adsorbate_at_site, find_surface_sites

    # Load structure
    try:
        slab = ase_read(structure)
    except Exception as e:
        click.echo(f"Error reading structure: {e}", err=True)
        raise click.Abort()

    click.echo(f"Loaded: {len(slab)} atoms, {slab.get_chemical_formula()}")

    # Find sites
    try:
        found_sites = find_surface_sites(slab, height=height, symprec=symprec)
    except Exception as e:
        click.echo(f"Error finding sites: {e}", err=True)
        raise click.Abort()

    if as_json:
        site_data = []
        for i, site in enumerate(found_sites):
            site_data.append({
                "index": i,
                "type": site.site_type,
                "position": site.position.tolist(),
                "atoms_indices": site.atoms_indices,
            })
        click.echo(json.dumps(site_data, indent=2))
        return

    # Count by type
    type_counts: dict[str, int] = {}
    for site in found_sites:
        type_counts[site.site_type] = type_counts.get(site.site_type, 0) + 1

    click.echo(f"\nFound {len(found_sites)} unique adsorption sites:")
    for site_type, count in sorted(type_counts.items()):
        click.echo(f"  {site_type}: {count}")

    click.echo(f"\n  {'#':<4} {'Type':<8} {'Position (x, y, z)'}")
    click.echo("  " + "-" * 50)
    for i, site in enumerate(found_sites):
        pos_str = f"({site.position[0]:.3f}, {site.position[1]:.3f}, {site.position[2]:.3f})"
        click.echo(f"  {i:<4} {site.site_type:<8} {pos_str}")

    # Generate structures with adsorbate if requested
    if adsorbate and output:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)

        click.echo(f"\nGenerating structures with {adsorbate}:")
        for i, site in enumerate(found_sites):
            struct_with_ads = add_adsorbate_at_site(slab, adsorbate, site)
            filename = f"{site.site_type}_{i:02d}.vasp"
            filepath = out_path / filename
            ase_write(filepath, struct_with_ads, format="vasp")
            click.echo(f"  - {filename}")

        click.echo(f"\nStructures saved to {output}/")


@cli.command()
@click.argument("structures", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--calculator", "-c", default="mace", help="MLIP calculator: mace, nequip")
@click.option("--model", "-m", default="medium", help="Model name or path")
@click.option("--device", "-d", default="cpu", help="Device: cpu, cuda, mps")
@click.option("--top-n", "-n", default=10, help="Number of top structures to select")
@click.option("--energy-window", "-w", type=float, help="Energy window from min (eV)")
@click.option("--output", "-o", type=click.Path(), help="Output directory for selected structures")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def screen(
    structures: tuple[str, ...],
    calculator: str,
    model: str,
    device: str,
    top_n: int,
    energy_window: float | None,
    output: str | None,
    as_json: bool,
) -> None:
    """Screen structures with MLIP for fast energy ranking.

    Rapidly evaluate candidate structures using machine learning potentials
    to identify promising configurations for DFT validation.

    Examples:
        # Screen structures with MACE foundation model
        atomix screen ./struct_*.vasp -c mace -m medium -n 5

        # Screen with energy window selection
        atomix screen ./candidates/ -w 0.5 --json

        # Save selected structures
        atomix screen ./site_*.cif -n 10 -o ./selected/
    """
    import json
    from pathlib import Path

    from ase.io import read as ase_read
    from ase.io import write as ase_write

    from atomix.calculators.mlip import get_mlip_calculator
    from atomix.core.screening import ScreeningConfig, ScreeningWorkflow

    # Load structures
    atoms_list = []
    file_names = []
    for struct_path in structures:
        path = Path(struct_path)
        if path.is_dir():
            # Try to find structure files in directory
            for ext in ["POSCAR", "CONTCAR", "*.vasp", "*.cif", "*.xyz"]:
                files = list(path.glob(ext))
                for f in files:
                    try:
                        atoms = ase_read(str(f))
                        atoms_list.append(atoms)
                        file_names.append(str(f))
                    except Exception:
                        pass
        else:
            try:
                atoms = ase_read(str(path))
                atoms_list.append(atoms)
                file_names.append(str(path))
            except Exception as e:
                click.echo(f"Warning: Could not read {path}: {e}", err=True)

    if not atoms_list:
        click.echo("No valid structures found", err=True)
        raise click.Abort()

    click.echo(f"Loaded {len(atoms_list)} structures")

    # Initialize calculator
    try:
        if Path(model).exists():
            # Custom model path
            mlip = get_mlip_calculator(calculator, model_path=model, device=device)
        else:
            # Foundation model name
            mlip = get_mlip_calculator(calculator, model=model, device=device)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("Install MLIP dependencies: pip install atomix[mlip]", err=True)
        raise click.Abort()

    # Configure screening
    config = ScreeningConfig(
        top_n=top_n,
        energy_window=energy_window,
    )

    # Run screening
    click.echo(f"Screening with {calculator} ({model}) on {device}...")
    workflow = ScreeningWorkflow(mlip, config)
    metadata = [{"filename": f} for f in file_names]
    results = workflow.screen(atoms_list, metadata)

    if as_json:
        json_results = []
        for r in results:
            json_results.append({
                "rank": r.rank,
                "filename": r.metadata.get("filename", ""),
                "energy": r.mlip_energy,
                "selected": r.selected_for_dft,
                "n_atoms": len(r.atoms),
                "formula": r.atoms.get_chemical_formula(),
            })
        click.echo(json.dumps(json_results, indent=2))
        return

    # Display results
    selected = workflow.get_selected()
    click.echo(f"\n=== Screening Results ({len(results)} structures) ===\n")
    click.echo(f"  {'Rank':<6} {'Energy (eV)':<14} {'Formula':<20} {'File'}")
    click.echo("  " + "-" * 70)

    for r in results[:20]:  # Show top 20
        marker = "*" if r.selected_for_dft else " "
        fname = Path(r.metadata.get("filename", "")).name[:25]
        formula = r.atoms.get_chemical_formula()[:18]
        energy_str = f"{r.mlip_energy:.4f}" if r.mlip_energy else "N/A"
        click.echo(f"{marker} {r.rank:<5} {energy_str:<14} {formula:<20} {fname}")

    if len(results) > 20:
        click.echo(f"  ... and {len(results) - 20} more structures")

    click.echo(f"\n  Selected for DFT validation: {len(selected)} structures")

    # Save selected structures if output specified
    if output and selected:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)

        click.echo(f"\nSaving selected structures to {output}/")
        for r in selected:
            orig_name = Path(r.metadata.get("filename", f"struct_{r.rank}")).stem
            out_file = out_path / f"{r.rank:03d}_{orig_name}.vasp"
            ase_write(str(out_file), r.atoms, format="vasp")

        click.echo(f"  Saved {len(selected)} structures")


@cli.command("train-data")
@click.argument("directories", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path(), help="Output file path")
@click.option("--format", "-f", "fmt", default="extxyz", help="Format: extxyz, db")
@click.option("--energy-key", default="REF_energy", help="Key for energy in extended XYZ")
@click.option("--forces-key", default="REF_forces", help="Key for forces in extended XYZ")
@click.option("--source", "-s", default="vasp", help="Source label for training data")
@click.option("--split", type=float, help="Validation split fraction (e.g., 0.1)")
@click.option("--recursive", "-r", is_flag=True, help="Search directories recursively")
def train_data(
    directories: tuple[str, ...],
    output: str,
    fmt: str,
    energy_key: str,
    forces_key: str,
    source: str,
    split: float | None,
    recursive: bool,
) -> None:
    """Export DFT calculations as MLIP training data.

    Reads completed VASP calculations and exports energies and forces
    in formats suitable for MLIP training (MACE, NequIP, etc.).

    Examples:
        # Export single calculation
        atomix train-data ./relaxation -o train.xyz

        # Export multiple calculations with custom keys
        atomix train-data ./calc_* -o data.xyz --energy-key energy --forces-key forces

        # Export with train/val split
        atomix train-data ./calcs/ -r -o train.xyz --split 0.1

        # Export to ASE database
        atomix train-data ./md_run -o train.db -f db
    """
    from pathlib import Path

    from atomix.calculators.vasp import VASPCalculator
    from atomix.core.active_learning import TrainingDataExporter, TrainingPoint

    # Find all calculation directories
    calc_dirs = []
    for d in directories:
        path = Path(d)
        if path.is_dir():
            # Check if this is a calculation directory
            if (path / "OUTCAR").exists() or (path / "vasprun.xml").exists():
                calc_dirs.append(path)

            # Search recursively if requested
            if recursive:
                for subdir in path.rglob("*"):
                    if subdir.is_dir():
                        if (subdir / "OUTCAR").exists() or (subdir / "vasprun.xml").exists():
                            calc_dirs.append(subdir)

    if not calc_dirs:
        click.echo("No valid calculation directories found", err=True)
        click.echo("Directories must contain OUTCAR or vasprun.xml")
        raise click.Abort()

    click.echo(f"Found {len(calc_dirs)} calculation directories")

    # Extract training data
    exporter = TrainingDataExporter()
    success_count = 0
    error_count = 0

    for calc_dir in calc_dirs:
        try:
            calc = VASPCalculator(calc_dir)
            results = calc.read_outputs()

            if results["energy"] is None:
                click.echo(f"  Skipping {calc_dir}: no energy found", err=True)
                error_count += 1
                continue

            if results["forces"] is None:
                click.echo(f"  Skipping {calc_dir}: no forces found", err=True)
                error_count += 1
                continue

            # Use final structure or trajectory frames
            if results.get("trajectory"):
                # For MD/relaxation, can export trajectory frames
                traj = results["trajectory"]
                # Just use final frame for now (could add --all-frames option)
                atoms = traj[-1] if traj else results["atoms"]
            else:
                atoms = results["atoms"]

            if atoms is None:
                click.echo(f"  Skipping {calc_dir}: no structure found", err=True)
                error_count += 1
                continue

            point = TrainingPoint(
                atoms=atoms,
                energy=results["energy"],
                forces=results["forces"],
                stress=results.get("stress"),
                source=source,
                metadata={"directory": str(calc_dir)},
            )
            exporter.add_point(point)
            success_count += 1

        except Exception as e:
            click.echo(f"  Error processing {calc_dir}: {e}", err=True)
            error_count += 1

    if len(exporter) == 0:
        click.echo("No valid training data extracted", err=True)
        raise click.Abort()

    click.echo(f"\nExtracted {success_count} training points ({error_count} errors)")

    # Handle train/val split
    out_path = Path(output)
    if split is not None and 0 < split < 1:
        train_exp, val_exp = exporter.split_train_val(val_fraction=split)

        # Determine output paths
        stem = out_path.stem
        suffix = out_path.suffix or (".xyz" if fmt == "extxyz" else ".db")
        train_path = out_path.parent / f"{stem}_train{suffix}"
        val_path = out_path.parent / f"{stem}_val{suffix}"

        if fmt == "extxyz":
            train_exp.to_extxyz(train_path, energy_key=energy_key, forces_key=forces_key)
            val_exp.to_extxyz(val_path, energy_key=energy_key, forces_key=forces_key)
        else:
            train_exp.to_ase_db(train_path)
            val_exp.to_ase_db(val_path)

        click.echo(f"\nSaved training data:")
        click.echo(f"  Train: {train_path} ({len(train_exp)} points)")
        click.echo(f"  Val:   {val_path} ({len(val_exp)} points)")
    else:
        # Single output file
        if fmt == "extxyz":
            exporter.to_extxyz(out_path, energy_key=energy_key, forces_key=forces_key)
        else:
            exporter.to_ase_db(out_path)

        click.echo(f"\nSaved {len(exporter)} training points to {out_path}")


@cli.command("screen-sites")
@click.argument("structure", type=click.Path(exists=True))
@click.option("--adsorbate", "-a", required=True, help="Adsorbate species (e.g., O, CO)")
@click.option("--calculator", "-c", default="mace", help="MLIP calculator: mace, nequip")
@click.option("--model", "-m", default="medium", help="Model name or path")
@click.option("--device", "-d", default="cpu", help="Device: cpu, cuda, mps")
@click.option("--height", "-h", default=2.0, help="Adsorbate height above site (Angstrom)")
@click.option("--top-n", "-n", default=5, help="Number of top sites to report")
@click.option("--output", "-o", type=click.Path(), help="Output directory for top structures")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def screen_sites(
    structure: str,
    adsorbate: str,
    calculator: str,
    model: str,
    device: str,
    height: float,
    top_n: int,
    output: str | None,
    as_json: bool,
) -> None:
    """Screen adsorption sites with MLIP.

    Finds adsorption sites on a surface slab and screens them
    with MLIP to rank by adsorption energy.

    Examples:
        # Screen O adsorption sites on Cu slab
        atomix screen-sites ./cu_slab.vasp -a O -c mace -m medium

        # Save top 3 configurations
        atomix screen-sites ./slab.cif -a CO -n 3 -o ./top_sites/
    """
    import json
    from pathlib import Path

    from ase import Atoms
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    from atomix.calculators.mlip import get_mlip_calculator
    from atomix.core.screening import AdsorptionScreening, ScreeningConfig
    from atomix.sites.surface import find_surface_sites

    # Load slab
    try:
        slab = ase_read(structure)
    except Exception as e:
        click.echo(f"Error reading structure: {e}", err=True)
        raise click.Abort()

    click.echo(f"Loaded slab: {len(slab)} atoms, {slab.get_chemical_formula()}")

    # Find adsorption sites
    sites = find_surface_sites(slab, height=height)
    if not sites:
        click.echo("No adsorption sites found on surface", err=True)
        raise click.Abort()

    click.echo(f"Found {len(sites)} adsorption sites")

    # Create adsorbate
    if len(adsorbate) <= 2 and adsorbate.isalpha():
        # Single atom or diatomic
        ads_atoms = Atoms(adsorbate)
    else:
        # Try as molecule formula
        from ase.build import molecule
        try:
            ads_atoms = molecule(adsorbate)
        except Exception:
            ads_atoms = Atoms(adsorbate)

    # Initialize MLIP
    try:
        if Path(model).exists():
            mlip = get_mlip_calculator(calculator, model_path=model, device=device)
        else:
            mlip = get_mlip_calculator(calculator, model=model, device=device)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

    # Run screening
    click.echo(f"Screening sites with {calculator} ({model})...")
    config = ScreeningConfig(top_n=top_n)
    screening = AdsorptionScreening(mlip, config)

    site_positions = [tuple(s.position) for s in sites]
    site_labels = [f"{s.site_type}_{i}" for i, s in enumerate(sites)]

    results = screening.screen_sites(
        slab, ads_atoms, site_positions,
        height=height, site_labels=site_labels
    )

    if as_json:
        json_results = []
        for r in results:
            json_results.append({
                "rank": r.rank,
                "site_label": r.metadata.get("site_label", ""),
                "energy": r.mlip_energy,
                "position": list(r.metadata.get("site", [])),
                "selected": r.selected_for_dft,
            })
        click.echo(json.dumps(json_results, indent=2))
        return

    # Display results
    click.echo(f"\n=== Site Screening Results ===\n")
    click.echo(f"  {'Rank':<6} {'Energy (eV)':<14} {'Site':<15} {'Position'}")
    click.echo("  " + "-" * 60)

    for r in results[:top_n + 5]:  # Show a few more than top_n
        marker = "*" if r.selected_for_dft else " "
        site_label = r.metadata.get("site_label", "")[:14]
        energy_str = f"{r.mlip_energy:.4f}" if r.mlip_energy else "N/A"
        pos = r.metadata.get("site", (0, 0, 0))
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        click.echo(f"{marker} {r.rank:<5} {energy_str:<14} {site_label:<15} {pos_str}")

    click.echo(f"\n  * = Selected (top {top_n})")

    # Save top structures if output specified
    if output:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)

        selected = [r for r in results if r.selected_for_dft]
        click.echo(f"\nSaving top {len(selected)} structures to {output}/")

        for r in selected:
            site_label = r.metadata.get("site_label", f"site_{r.rank}")
            out_file = out_path / f"{r.rank:02d}_{site_label}.vasp"
            ase_write(str(out_file), r.atoms, format="vasp")

        click.echo(f"  Saved {len(selected)} structures")


if __name__ == "__main__":
    cli()
