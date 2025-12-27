"""
Documentation command for Template Heaven CLI.

Provides commands to generate and manage stack documentation.
"""

import json
from pathlib import Path
from typing import Optional

import click

from ...core.stack_documentation import StackDocumentationGenerator
from ...utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
def docs():
    """Generate and manage stack documentation."""
    pass


@docs.command()
@click.option(
    "--stack",
    help="Specific stack to document (generate docs for all stacks if not specified)"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output results to JSON file"
)
@click.option(
    "--force",
    is_flag=True,
    help="Force regeneration of existing documentation"
)
def generate(stack: Optional[str], output: Optional[str], force: bool):
    """
    Generate comprehensive documentation for stack branches.

    Creates detailed README files with stack-specific information,
    technology overviews, usage guides, and template listings.
    """
    try:
        generator = StackDocumentationGenerator()

        if stack:
            click.echo(f"📝 Generating documentation for stack: {stack}")
            result = generator.generate_stack_documentation(stack)
        else:
            click.echo("📝 Generating documentation for all stacks")
            result = generator.generate_all_stack_documentation()

        # Display results
        _display_generation_results(result)

        # Save to file if requested
        if output:
            Path(output).write_text(json.dumps(result, indent=2))
            click.echo(f"\n💾 Results saved to: {output}")

    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@docs.command()
@click.argument("stack_name")
def preview(stack_name: str):
    """
    Preview documentation for a specific stack.

    Shows what the generated documentation would look like without
    actually writing files.
    """
    try:
        generator = StackDocumentationGenerator()

        click.echo(f"👀 Previewing documentation for stack: {stack_name}")

        # Get stack configuration
        stack_config = generator.stack_config.get_stack_config(stack_name)
        if not stack_config:
            click.echo(f"❌ Stack not found: {stack_name}")
            return

        # Generate content preview
        stack_info = generator._get_stack_info(stack_name)
        templates = generator._get_stack_templates(stack_name)

        click.echo("\n" + "="*50)
        click.echo(f"STACK: {stack_name.upper()}")
        click.echo("="*50)

        click.echo(f"\n📊 Stack Info:")
        click.echo(f"  Display Name: {stack_info.get('display_name')}")
        click.echo(f"  Description: {stack_config.description}")
        click.echo(f"  Technologies: {len(stack_config.technologies)}")
        click.echo(f"  Templates: {len(templates)}")

        click.echo(f"\n🛠️  Quality Standards:")
        for i, standard in enumerate(stack_config.quality_standards[:3], 1):
            click.echo(f"  {i}. {standard}")
        if len(stack_config.quality_standards) > 3:
            click.echo(f"  ... and {len(stack_config.quality_standards) - 3} more")

        click.echo(f"\n📋 Will Generate:")
        click.echo(f"  - stacks/{stack_name}/README.md")
        if templates:
            click.echo(f"  - stacks/{stack_name}/TEMPLATES.md")

    except Exception as e:
        logger.error(f"Preview failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@docs.command()
def status():
    """Show documentation status across all stacks."""
    try:
        generator = StackDocumentationGenerator()

        click.echo("📚 Documentation Status")
        click.echo("=" * 40)

        all_stacks = generator.stack_config.get_all_stacks()

        documented = 0
        total_templates = 0

        for stack_name in all_stacks:
            stack_dir = Path("stacks") / stack_name
            readme_exists = (stack_dir / "README.md").exists()

            if readme_exists:
                documented += 1
                click.echo(f"✅ {stack_name}: Documented")
            else:
                click.echo(f"❌ {stack_name}: Missing documentation")

            # Count templates
            if stack_dir.exists():
                templates = [d for d in stack_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                total_templates += len(templates)

        click.echo("-" * 40)
        click.echo(f"📝 Documented stacks: {documented}/{len(all_stacks)}")
        click.echo(f"📦 Total templates: {total_templates}")

        if documented < len(all_stacks):
            click.echo(f"\n💡 Run 'templateheaven docs generate' to create missing documentation")

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


def _display_generation_results(result: dict):
    """Display documentation generation results."""
    if "total_stacks" in result:
        # Multi-stack results
        successful = result.get("successful_generations", 0)
        total = result.get("total_stacks", 0)

        click.echo("\n📝 Documentation Generation Complete!"        click.echo(f"✅ {successful}/{total} stacks documented successfully")

        if result.get("errors"):
            click.echo(f"⚠️  {len(result['errors'])} errors encountered")

        # Show per-stack results
        click.echo("\n📊 Per-Stack Results:")
        for stack_name, stack_result in result.get("stack_results", {}).items():
            if stack_result.get("success"):
                files = len(stack_result.get("files_generated", []))
                click.echo(f"  ✅ {stack_name}: {files} files generated")
            else:
                click.echo(f"  ❌ {stack_name}: Failed")

        # Show errors if any
        if result.get("errors"):
            click.echo("\n❌ Errors:")
            for error in result["errors"][:5]:  # Show first 5 errors
                click.echo(f"  - {error}")
            if len(result["errors"]) > 5:
                click.echo(f"  - ... and {len(result['errors']) - 5} more")

    else:
        # Single stack results
        if result.get("success"):
            files = len(result.get("files_generated", []))
            click.echo("\n✅ Stack documentation generated successfully!"            click.echo(f"📄 {files} files generated")

            if result.get("files_generated"):
                click.echo("\n📋 Generated files:")
                for file_path in result["files_generated"]:
                    click.echo(f"  - {file_path}")

        else:
            click.echo("\n❌ Stack documentation generation failed!")
            if result.get("errors"):
                click.echo("Errors:")
                for error in result["errors"]:
                    click.echo(f"  - {error}")
