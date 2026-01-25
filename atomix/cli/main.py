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


if __name__ == "__main__":
    cli()
