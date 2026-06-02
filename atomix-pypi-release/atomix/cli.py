"""Command-line interface for atomix."""

import click
from atomix import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """
    atomix: Atomistic Modeling Interface for eXploration

    A natural-language-driven toolkit for ab initio / DFT / atomistic modeling.

    This is an early alpha release (v0.1.1). Full features coming in v0.2.0+.
    """
    pass


@main.command()
def info():
    """Show atomix information."""
    click.echo(f"atomix version {__version__}")
    click.echo("\nStatus: Alpha - Under active development")
    click.echo("\nTarget applications:")
    click.echo("  • VASP DFT calculations")
    click.echo("  • Machine learning interatomic potentials (MACE, NequIP)")
    click.echo("  • Catalysis and surface chemistry workflows")
    click.echo("  • Atomistic simulation automation")
    click.echo("\nFull features coming in v0.2.0+")


if __name__ == "__main__":
    main()
