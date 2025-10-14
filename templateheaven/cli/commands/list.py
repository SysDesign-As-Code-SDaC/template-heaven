"""
List command implementation for Template Heaven CLI.

This module provides the command-line implementation for listing templates
in various formats.
"""

import json
import yaml
from typing import List

from ...core.models import Template
from ...utils.logger import get_logger

logger = get_logger(__name__)


def list_command(templates: List[Template], format: str = 'table') -> None:
    """
    Display templates in the specified format.
    
    Args:
        templates: List of templates to display
        format: Output format ('table', 'json', 'yaml')
    """
    if not templates:
        print("No templates found.")
        return
    
    if format == 'table':
        _display_table(templates)
    elif format == 'json':
        _display_json(templates)
    elif format == 'yaml':
        _display_yaml(templates)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _display_table(templates: List[Template]) -> None:
    """
    Display templates in a table format.
    
    Args:
        templates: List of templates to display
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    
    # Create table
    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Stack", style="green")
    table.add_column("Description", style="white")
    table.add_column("Tags", style="blue")
    table.add_column("Version", style="yellow")
    
    # Add rows
    for template in templates:
        # Truncate description if too long
        description = template.description
        if len(description) > 60:
            description = description[:57] + "..."
        
        # Format tags
        tags = ", ".join(template.tags[:3])  # Show first 3 tags
        if len(template.tags) > 3:
            tags += f" (+{len(template.tags) - 3} more)"
        
        table.add_row(
            template.name,
            template.stack.value,
            description,
            tags,
            template.version or "N/A"
        )
    
    console.print(table)
    
    # Display summary
    console.print(f"\n[green]Found {len(templates)} templates[/green]")


def _display_json(templates: List[Template]) -> None:
    """
    Display templates in JSON format.
    
    Args:
        templates: List of templates to display
    """
    data = {
        "templates": [template.to_dict() for template in templates],
        "count": len(templates)
    }
    
    print(json.dumps(data, indent=2))


def _display_yaml(templates: List[Template]) -> None:
    """
    Display templates in YAML format.
    
    Args:
        templates: List of templates to display
    """
    data = {
        "templates": [template.to_dict() for template in templates],
        "count": len(templates)
    }
    
    print(yaml.dump(data, default_flow_style=False, sort_keys=True))
