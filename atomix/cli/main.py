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
def status(directory: str) -> None:
    """Check status of calculations in directory."""
    click.echo(f"Checking status in: {directory}")
    click.echo("(Not yet implemented)")


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
