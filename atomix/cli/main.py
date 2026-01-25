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
@click.option("--type", "-t", "calc_type", default="energy", help="Analysis type")
def analyze(directory: str, calc_type: str) -> None:
    """Analyze calculation outputs."""
    click.echo(f"Analyzing {calc_type} in: {directory}")
    click.echo("(Not yet implemented)")


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
