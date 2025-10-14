"""
Config command implementation for Template Heaven CLI.

This module provides the command-line implementation for configuration management.
"""

from typing import Optional

from ...config.settings import Config
from ...utils.logger import get_logger

logger = get_logger(__name__)


def config_command(
    config: Config,
    key: Optional[str] = None,
    value: Optional[str] = None,
    unset: Optional[str] = None,
    list_all: bool = False,
    reset: bool = False
) -> None:
    """
    Handle configuration operations.
    
    Args:
        config: Configuration instance
        key: Configuration key to get/set
        value: Configuration value to set
        unset: Configuration key to unset
        list_all: Whether to list all configuration
        reset: Whether to reset configuration to defaults
    """
    if reset:
        _handle_reset(config)
    elif unset:
        _handle_unset(config, unset)
    elif key and value:
        _handle_set(config, key, value)
    elif key:
        _handle_get(config, key)
    elif list_all:
        _handle_list_all(config)
    else:
        _handle_help()


def _handle_reset(config: Config) -> None:
    """Handle configuration reset."""
    from rich.console import Console
    from rich.prompt import Confirm
    
    console = Console()
    
    if Confirm.ask("Are you sure you want to reset configuration to defaults?"):
        config.reset()
        console.print("[green]Configuration reset to defaults[/green]")
    else:
        console.print("[yellow]Configuration reset cancelled[/yellow]")


def _handle_unset(config: Config, key: str) -> None:
    """Handle configuration unset."""
    from rich.console import Console
    
    console = Console()
    
    if config.unset(key):
        console.print(f"[green]Configuration key '{key}' removed[/green]")
        config.save()
    else:
        console.print(f"[yellow]Configuration key '{key}' not found[/yellow]")


def _handle_set(config: Config, key: str, value: str) -> None:
    """Handle configuration set."""
    from rich.console import Console
    
    console = Console()
    
    # Validate key format
    if not _is_valid_config_key(key):
        console.print(f"[red]Invalid configuration key: {key}[/red]")
        console.print("[yellow]Valid keys: cache_dir, default_author, default_license, github_token, auto_update, log_level[/yellow]")
        return
    
    # Set value
    config.set(key, value)
    config.save()
    
    console.print(f"[green]Configuration set: {key} = {value}[/green]")


def _handle_get(config: Config, key: str) -> None:
    """Handle configuration get."""
    from rich.console import Console
    
    console = Console()
    
    value = config.get(key)
    if value is not None:
        console.print(f"[cyan]{key}:[/cyan] {value}")
    else:
        console.print(f"[yellow]Configuration key '{key}' not found[/yellow]")


def _handle_list_all(config: Config) -> None:
    """Handle list all configuration."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Get all configuration
    all_config = config.get_all()
    
    # Create table
    table = Table(title="Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    
    # Add configuration entries
    for key, value in all_config.items():
        if key.startswith('_'):  # Skip metadata
            continue
        
        # Format value for display
        if isinstance(value, dict):
            value_str = _format_dict_value(value)
        elif isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
        
        # Truncate long values
        if len(value_str) > 50:
            value_str = value_str[:47] + "..."
        
        table.add_row(key, value_str)
    
    console.print(table)
    
    # Display config info
    config_info = config.get_config_info()
    console.print(f"\n[cyan]Config file:[/cyan] {config_info['config_file']}")
    console.print(f"[cyan]Cache directory:[/cyan] {config_info['cache_dir']}")


def _handle_help() -> None:
    """Display configuration help."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    help_text = """
Configuration Management Commands:

  templateheaven config --list-all              # List all configuration
  templateheaven config --key <key>             # Get configuration value
  templateheaven config --key <key> --value <value>  # Set configuration value
  templateheaven config --unset <key>           # Remove configuration key
  templateheaven config --reset                 # Reset to defaults

Available Configuration Keys:

  cache_dir              # Template cache directory
  default_author         # Default project author
  default_license        # Default project license
  github_token           # GitHub API token (optional)
  auto_update            # Enable automatic updates
  log_level              # Logging level (DEBUG, INFO, WARNING, ERROR)
  package_managers.python  # Default Python package manager
  package_managers.node    # Default Node.js package manager
  package_managers.rust    # Default Rust package manager
  package_managers.go      # Default Go package manager

Examples:

  templateheaven config --key default_author --value "John Doe"
  templateheaven config --key github_token --value "ghp_xxxxx"
  templateheaven config --unset github_token
"""
    
    console.print(Panel(Text(help_text.strip(), style="white"), title="Configuration Help"))


def _is_valid_config_key(key: str) -> bool:
    """
    Validate configuration key.
    
    Args:
        key: Configuration key to validate
        
    Returns:
        True if key is valid
    """
    valid_keys = {
        'cache_dir',
        'default_author',
        'default_license',
        'github_token',
        'auto_update',
        'log_level',
        'ui_theme',
        'package_managers.python',
        'package_managers.node',
        'package_managers.rust',
        'package_managers.go',
        'editor_preferences.default_editor',
        'editor_preferences.open_after_create',
        'template_preferences.include_git',
        'template_preferences.include_readme',
        'template_preferences.include_license',
        'template_preferences.include_contributing',
    }
    
    return key in valid_keys


def _format_dict_value(value: dict) -> str:
    """
    Format dictionary value for display.
    
    Args:
        value: Dictionary to format
        
    Returns:
        Formatted string
    """
    if not value:
        return "{}"
    
    items = []
    for k, v in value.items():
        items.append(f"{k}: {v}")
    
    return "{" + ", ".join(items) + "}"
