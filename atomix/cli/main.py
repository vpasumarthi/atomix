"""Main CLI entry point for atomix."""

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
def generate(prompt: str, structure: str | None, output: str, provider: str) -> None:
    """Generate simulation inputs from natural language.

    PROMPT is the natural language description of the desired simulation.

    Example:
        atomix generate "Relax Cu(111) 3x3 slab, 4 layers, PBE+D3"
    """
    click.echo(f"Generating simulation setup for: {prompt}")
    click.echo("(Not yet implemented)")


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
