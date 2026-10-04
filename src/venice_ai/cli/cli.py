#!/usr/bin/env python3
"""
Venice AI CLI - Main entry point

This module provides the main CLI interface for Venice AI.
"""

import sys
from pathlib import Path

import click

from .commands import account as account_commands
from .commands import api_keys as api_keys_commands
from .commands import audio as audio_commands
from .commands import characters as characters_commands
from .commands import chat as chat_commands
from .commands import decisions as decisions_commands
from .commands import embeddings as embeddings_commands
from .commands import image as image_commands
from .commands import skills as skills_commands
from .commands import video as video_commands
from .commands.health import health_command
from .commands.lint import lint_command
from .commands.models.group import models as models_group
from .config import load_config, set_active_config_path
from .utils.console import console, enable_plain_mode, print_version_info

# Top-level command grouping for the curated ``venice-py --help`` layout.
# The CLI ships ~13 subcommands; lumping them in one alphabetical block was
# making the help output a wall of names, so we group them by intent.
_COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Generate", ("chat", "image", "audio", "video", "embeddings", "decisions")),
    ("Discover", ("models", "characters", "health")),
    ("Account", ("account", "api-keys")),
    ("Develop", ("lint", "skills", "configure", "completion")),
)


class _GroupedCLI(click.Group):
    """Click ``Group`` whose ``--help`` renders the grouped layout above."""

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        seen: set[str] = set()
        for section, names in _COMMAND_GROUPS:
            rows: list[tuple[str, str]] = []
            for name in names:
                cmd = self.get_command(ctx, name)
                if cmd is None or cmd.hidden:
                    continue
                seen.add(name)
                rows.append((name, cmd.get_short_help_str(limit=72)))
            if rows:
                with formatter.section(f"Commands ({section})"):
                    formatter.write_dl(rows)
        # Catch-all for any command we haven't listed (e.g. plugins).
        leftover: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            if name in seen:
                continue
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            leftover.append((name, cmd.get_short_help_str(limit=72)))
        if leftover:
            with formatter.section("Commands"):
                formatter.write_dl(leftover)


@click.group(cls=_GroupedCLI, invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--config", type=click.Path(), help="Path to config file")
@click.option("--plain", is_flag=True, help="Plain text output (no colors, panels, or emojis)")
@click.pass_context
def cli(ctx: click.Context, version: bool, config: str | None, plain: bool) -> None:
    """
    venice-py - Your Venice AI assistant in the terminal.

    Unofficial and community-maintained. This is not Venice's official CLI,
    which is a separate Node program invoked as ``venice``.

    Featured commands: ``venice-py models resolve`` (find a model by capability),
    ``venice-py lint`` (catch v1 / OpenAI-style usage in your code).
    """
    # Enable plain mode first if requested (before any output)
    if plain:
        enable_plain_mode()

    # Handle version flag early — no config loading needed
    if version:
        print_version_info()
        ctx.exit()
        return

    if ctx.invoked_subcommand is None:
        # No subcommand provided, show help
        click.echo(ctx.get_help())

    # Load configuration
    config_path = Path(config) if config else None
    # Record the chosen config path so downstream commands resolve the API key
    # and base URL from the user-specified file rather than the default.
    set_active_config_path(config_path)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    ctx.obj["console"] = console
    ctx.obj["plain"] = plain


# Register command groups
cli.add_command(account_commands.account)
cli.add_command(api_keys_commands.api_keys)
cli.add_command(audio_commands.audio)
cli.add_command(characters_commands.characters)
cli.add_command(chat_commands.chat)
cli.add_command(decisions_commands.decisions)
cli.add_command(embeddings_commands.embeddings)
cli.add_command(image_commands.image)
cli.add_command(video_commands.video)
cli.add_command(models_group)
cli.add_command(lint_command)
cli.add_command(skills_commands.skills)
cli.add_command(health_command)


@cli.command()
@click.pass_context
def configure(ctx: click.Context) -> None:
    """Configure Venice AI CLI settings"""
    from .commands.configure import configure_cli

    configure_cli(ctx.obj)


@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@click.pass_context
def completion(ctx: click.Context, shell: str) -> None:
    """Generate shell completion script.

    Examples:

        venice-py completion bash >> ~/.bashrc

        venice-py completion zsh >> ~/.zshrc

        venice-py completion fish > ~/.config/fish/completions/venice-py.fish
    """
    from click.shell_completion import BashComplete, FishComplete, ZshComplete

    prog_name = "venice-py"
    complete_var = f"_{prog_name.replace('-', '_').upper()}_COMPLETE"

    shell_map = {
        "bash": BashComplete,
        "zsh": ZshComplete,
        "fish": FishComplete,
    }

    complete_cls = shell_map[shell]
    complete_obj = complete_cls(cli, ctx.params, prog_name, complete_var)
    click.echo(complete_obj.source())


def main():
    """Main entry point for the CLI"""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
