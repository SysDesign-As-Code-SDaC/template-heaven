"""
Main CLI entry point for Template Heaven.

This module provides the main command-line interface using Click framework
with support for interactive commands and beautiful terminal output.
"""

import click
from pathlib import Path
from typing import Optional

from ..__version__ import __version__
from ..config.settings import Config
from ..core.template_manager import TemplateManager
from ..utils.logger import setup_logging, get_logger
from .commands.init import init_command
from .commands.list import list_command
from .commands.config import config_command
from .wizard import Wizard

logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="Template Heaven")
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--config-dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help='Custom configuration directory'
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_dir: Optional[Path]) -> None:
    """
    Template Heaven - Interactive template management.
    
    A comprehensive tool for discovering, customizing, and initializing
    projects from templates across 24+ technology stacks.
    
    Examples:
        templateheaven init                    # Launch interactive wizard
        templateheaven list --stack frontend  # List frontend templates
        templateheaven config set author "John Doe"  # Set configuration
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set up logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    # Initialize configuration
    try:
        config = Config(config_dir) if config_dir else Config()
        ctx.obj['config'] = config
        
        # Initialize template manager
        template_manager = TemplateManager(config)
        ctx.obj['template_manager'] = template_manager
        
        logger.debug("CLI initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize CLI: {e}")
        raise click.ClickException(f"Initialization failed: {e}")


@cli.command()
@click.option(
    '--template', '-t',
    help='Template name to use'
)
@click.option(
    '--name', '-n',
    help='Project name'
)
@click.option(
    '--stack', '-s',
    help='Stack category'
)
@click.option(
    '--directory', '-d',
    type=click.Path(path_type=Path),
    default='.',
    help='Output directory (default: current directory)'
)
@click.option(
    '--author',
    help='Project author'
)
@click.option(
    '--license',
    help='Project license'
)
@click.option(
    '--package-manager',
    type=click.Choice(['npm', 'yarn', 'pnpm', 'pip', 'poetry', 'cargo', 'go']),
    help='Package manager to use'
)
@click.option(
    '--no-wizard',
    is_flag=True,
    help='Skip interactive wizard and use command line options only'
)
@click.pass_context
def init(
    ctx: click.Context,
    template: Optional[str],
    name: Optional[str],
    stack: Optional[str],
    directory: Path,
    author: Optional[str],
    license: Optional[str],
    package_manager: Optional[str],
    no_wizard: bool
) -> None:
    """
    Initialize a new project from a template.
    
    This command can be used in two modes:
    
    1. Interactive wizard (default): Launches an interactive wizard to guide
       you through template selection and project configuration.
    
    2. Command-line mode: Use command-line options to specify all parameters.
       Use --no-wizard to skip the interactive mode.
    
    Examples:
        templateheaven init
        templateheaven init --template react-vite --name my-app
        templateheaven init --stack frontend --name my-project --no-wizard
    """
    config = ctx.obj['config']
    template_manager = ctx.obj['template_manager']
    
    try:
        if no_wizard and template and name:
            # Command-line mode
            init_command(
                template_manager=template_manager,
                config=config,
                template_name=template,
                project_name=name,
                stack=stack,
                output_dir=directory,
                author=author,
                license=license,
                package_manager=package_manager
            )
        else:
            # Interactive wizard mode
            wizard = Wizard(template_manager, config)
            wizard.run(output_dir=directory)
            
    except Exception as e:
        logger.error(f"Failed to initialize project: {e}")
        raise click.ClickException(f"Initialization failed: {e}")


@cli.command()
@click.option(
    '--stack', '-s',
    help='Filter by stack category'
)
@click.option(
    '--tags',
    help='Filter by tags (comma-separated)'
)
@click.option(
    '--search',
    help='Search in name, description, and tags'
)
@click.option(
    '--format',
    type=click.Choice(['table', 'json', 'yaml']),
    default='table',
    help='Output format'
)
@click.option(
    '--limit',
    type=int,
    default=50,
    help='Maximum number of results'
)
@click.pass_context
def list(
    ctx: click.Context,
    stack: Optional[str],
    tags: Optional[str],
    search: Optional[str],
    format: str,
    limit: int
) -> None:
    """
    List available templates.
    
    Examples:
        templateheaven list
        templateheaven list --stack frontend
        templateheaven list --tags react,typescript
        templateheaven list --search "machine learning"
    """
    template_manager = ctx.obj['template_manager']
    
    try:
        # Parse tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
        
        # Get templates
        templates = template_manager.list_templates(
            stack=stack,
            tags=tag_list,
            search=search
        )
        
        # Limit results
        if limit > 0:
            templates = templates[:limit]
        
        # Display results
        list_command(templates, format=format)
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise click.ClickException(f"Failed to list templates: {e}")


@cli.command()
@click.argument('template_name')
@click.pass_context
def info(ctx: click.Context, template_name: str) -> None:
    """
    Show detailed information about a template.
    
    Examples:
        templateheaven info react-vite
        templateheaven info fastapi
    """
    template_manager = ctx.obj['template_manager']
    
    try:
        template = template_manager.get_template(template_name)
        if not template:
            raise click.ClickException(f"Template not found: {template_name}")
        
        # Display template information
        _display_template_info(template)
        
    except Exception as e:
        logger.error(f"Failed to get template info: {e}")
        raise click.ClickException(f"Failed to get template info: {e}")


@cli.command()
@click.option(
    '--query', '-q',
    help='Search query'
)
@click.option(
    '--limit',
    type=int,
    default=10,
    help='Maximum number of results'
)
@click.option(
    '--min-score',
    type=float,
    default=0.1,
    help='Minimum relevance score'
)
@click.pass_context
def search(
    ctx: click.Context,
    query: Optional[str],
    limit: int,
    min_score: float
) -> None:
    """
    Search templates with relevance scoring.
    
    Examples:
        templateheaven search "react typescript"
        templateheaven search --query "machine learning" --limit 5
    """
    template_manager = ctx.obj['template_manager']
    
    if not query:
        query = click.prompt("Enter search query")
    
    try:
        results = template_manager.search_templates(
            query=query,
            limit=limit,
            min_score=min_score
        )
        
        if not results:
            click.echo("No templates found matching your query.")
            return
        
        # Display search results
        _display_search_results(results)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise click.ClickException(f"Search failed: {e}")


@cli.command()
@click.option(
    '--key',
    help='Configuration key to get'
)
@click.option(
    '--value',
    help='Configuration value to set'
)
@click.option(
    '--unset',
    help='Configuration key to unset'
)
@click.option(
    '--list-all',
    is_flag=True,
    help='List all configuration values'
)
@click.option(
    '--reset',
    is_flag=True,
    help='Reset configuration to defaults'
)
@click.pass_context
def config(
    ctx: click.Context,
    key: Optional[str],
    value: Optional[str],
    unset: Optional[str],
    list_all: bool,
    reset: bool
) -> None:
    """
    Manage configuration settings.
    
    Examples:
        templateheaven config --list-all
        templateheaven config --key default_author --value "John Doe"
        templateheaven config --unset github_token
        templateheaven config --reset
    """
    config = ctx.obj['config']
    
    try:
        config_command(
            config=config,
            key=key,
            value=value,
            unset=unset,
            list_all=list_all,
            reset=reset
        )
        
    except Exception as e:
        logger.error(f"Configuration operation failed: {e}")
        raise click.ClickException(f"Configuration operation failed: {e}")


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """
    Show template statistics and cache information.
    """
    template_manager = ctx.obj['template_manager']
    config = ctx.obj['config']
    
    try:
        # Get template statistics
        template_stats = template_manager.get_template_stats()
        cache_stats = template_manager.get_cache_stats()
        config_info = config.get_config_info()
        
        # Display statistics
        _display_stats(template_stats, cache_stats, config_info)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise click.ClickException(f"Failed to get statistics: {e}")


def _display_template_info(template) -> None:
    """Display detailed template information."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    
    # Template header
    header = Text(f"{template.get_display_name()}", style="bold blue")
    console.print(Panel(header, title="Template Information"))
    
    # Basic information
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Name", template.name)
    info_table.add_row("Stack", template.stack.value)
    info_table.add_row("Description", template.description)
    info_table.add_row("Version", template.version or "N/A")
    info_table.add_row("Author", template.author or "N/A")
    info_table.add_row("License", template.license or "N/A")
    
    if template.upstream_url:
        info_table.add_row("Upstream URL", template.upstream_url)
    
    console.print(info_table)
    
    # Tags
    if template.tags:
        tags_text = ", ".join(template.tags)
        console.print(f"\n[cyan]Tags:[/cyan] {tags_text}")
    
    # Features
    if template.features:
        console.print(f"\n[cyan]Features:[/cyan]")
        for feature in template.features:
            console.print(f"  • {feature}")
    
    # Dependencies
    if template.dependencies:
        console.print(f"\n[cyan]Dependencies:[/cyan]")
        for dep, version in template.dependencies.items():
            console.print(f"  • {dep}{version}")


def _display_search_results(results) -> None:
    """Display search results."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    
    table = Table(title="Search Results")
    table.add_column("Template", style="cyan")
    table.add_column("Stack", style="green")
    table.add_column("Description", style="white")
    table.add_column("Score", style="yellow")
    table.add_column("Match", style="blue")
    
    for result in results:
        table.add_row(
            result.template.name,
            result.template.stack.value,
            result.template.description[:50] + "..." if len(result.template.description) > 50 else result.template.description,
            f"{result.score:.2f}",
            result.match_reason or "N/A"
        )
    
    console.print(table)


def _display_stats(template_stats, cache_stats, config_info) -> None:
    """Display statistics."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    
    console = Console()
    
    # Template statistics
    template_table = Table(title="Template Statistics")
    template_table.add_column("Metric", style="cyan")
    template_table.add_column("Value", style="white")
    
    template_table.add_row("Total Templates", str(template_stats['total_templates']))
    template_table.add_row("Stacks", str(len(template_stats['stacks'])))
    template_table.add_row("Unique Tags", str(len(template_stats['tags'])))
    template_table.add_row("Dependencies", str(len(template_stats['dependencies'])))
    
    console.print(Panel(template_table, title="Template Statistics"))
    
    # Cache statistics
    cache_table = Table(title="Cache Statistics")
    cache_table.add_column("Metric", style="cyan")
    cache_table.add_column("Value", style="white")
    
    cache_table.add_row("Total Entries", str(cache_stats['total_entries']))
    cache_table.add_row("Active Entries", str(cache_stats['active_entries']))
    cache_table.add_row("Expired Entries", str(cache_stats['expired_entries']))
    cache_table.add_row("Cache Size", f"{cache_stats['total_size_bytes']} bytes")
    
    console.print(Panel(cache_table, title="Cache Statistics"))
    
    # Configuration info
    config_table = Table(title="Configuration")
    config_table.add_column("Property", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Config File", config_info['config_file'])
    config_table.add_row("Cache Directory", config_info['cache_dir'])
    config_table.add_row("First Run", str(config_info['is_first_run']))
    
    console.print(Panel(config_table, title="Configuration"))


if __name__ == '__main__':
    cli()
