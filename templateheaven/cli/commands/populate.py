"""
Populate command for Template Heaven CLI.

Provides commands to populate stack branches with templates discovered from GitHub.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import click

from ...core.template_populator import TemplatePopulator
from ...utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
def populate():
    """Populate stack branches with templates from GitHub."""
    pass


@populate.command()
@click.option(
    "--stack",
    help="Specific stack to populate (populate all stacks if not specified)"
)
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Force refresh of cached GitHub data"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output results to JSON file"
)
def run(stack: Optional[str], force_refresh: bool, dry_run: bool, output: Optional[str]):
    """
    Populate stack branches with templates from GitHub.

    Discovers high-quality templates, validates them against stack requirements,
    and adds them to the appropriate stack branches.
    """
    try:
        populator = TemplatePopulator()

        if dry_run:
            click.echo("🔍 Dry run mode - no changes will be made")
            # In dry run, just show current status
            status = asyncio.run(populator.get_population_status())

            click.echo("\n📊 Current Population Status:")
            click.echo(f"Total templates across all stacks: {status['total_templates']}")

            for stack_name, stack_info in status["stacks"].items():
                click.echo(f"  {stack_name}: {stack_info['templates_count']} templates")

            return

        if stack:
            click.echo(f"🔄 Populating stack: {stack}")
            result = asyncio.run(populator.populate_stack(stack, force_refresh))
        else:
            click.echo("🔄 Populating all stacks")
            result = asyncio.run(populator.populate_all_stacks(force_refresh))

        # Display results
        _display_results(result)

        # Save to file if requested
        if output:
            Path(output).write_text(json.dumps(result, indent=2))
            click.echo(f"\n💾 Results saved to: {output}")

    except Exception as e:
        logger.error(f"Population failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@populate.command()
def status():
    """Show current template population status across all stacks."""
    try:
        populator = TemplatePopulator()
        status = asyncio.run(populator.get_population_status())

        click.echo("📊 Template Population Status")
        click.echo("=" * 40)

        total_templates = 0
        for stack_name, stack_info in status["stacks"].items():
            count = stack_info["templates_count"]
            total_templates += count

            if count == 0:
                click.echo(f"❌ {stack_name}: No templates")
            elif count == 1:
                click.echo(f"✅ {stack_name}: {count} template")
            else:
                click.echo(f"✅ {stack_name}: {count} templates")

        click.echo("-" * 40)
        click.echo(f"📈 Total: {total_templates} templates across {len(status['stacks'])} stacks")

        # Show recommendations
        empty_stacks = [name for name, info in status["stacks"].items() if info["templates_count"] == 0]
        if empty_stacks:
            click.echo(f"\n💡 Recommendation: Run 'templateheaven populate run' to populate {len(empty_stacks)} empty stacks")

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@populate.command()
@click.argument("stack_name")
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of candidates to show"
)
def discover(stack_name: str, limit: int):
    """Discover potential templates for a specific stack."""
    try:
        populator = TemplatePopulator()

        click.echo(f"🔍 Discovering templates for stack: {stack_name}")

        candidates = asyncio.run(populator.github_search.discover_templates_for_stack(stack_name, limit))

        if not candidates:
            click.echo("❌ No template candidates found")
            return

        click.echo(f"\n📋 Found {len(candidates)} potential templates:")
        click.echo("-" * 80)

        for i, candidate in enumerate(candidates, 1):
            repo = candidate["repository"]
            analysis = candidate

            valid = candidate.get("stack_validation", {}).get("valid", False)
            status_icon = "✅" if valid else "❌"

            click.echo(f"{i}. {status_icon} {repo['name']}")
            click.echo(f"   📦 {repo.get('full_name', '')}")
            click.echo(f"   ⭐ {repo.get('stargazers_count', 0)} stars")
            click.echo(f"   🍴 {repo.get('forks_count', 0)} forks")
            click.echo(f"   🎯 Potential: {analysis.get('template_potential', 0.0):.2f}")
            click.echo(f"   ✨ Quality: {analysis.get('quality_score', 0.0):.2f}")

            if valid:
                click.echo("   ✅ Valid for stack")
            else:
                issues = candidate.get("stack_validation", {}).get("issues", [])
                if issues:
                    click.echo(f"   ❌ Issues: {issues[0]}")

            click.echo()

    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


def _display_results(result: dict):
    """Display population results in a user-friendly format."""
    if result.get("total_stacks"):
        # Multi-stack results
        successful = result.get("successful_populations", 0)
        total = result.get("total_stacks", 0)
        added = result.get("templates_added", 0)

        click.echo("\n🎉 Population Complete!"        click.echo(f"✅ {successful}/{total} stacks populated successfully")
        click.echo(f"📦 {added} templates added total")

        if result.get("errors"):
            click.echo(f"⚠️  {len(result['errors'])} errors encountered")

        # Show per-stack results
        click.echo("\n📊 Per-Stack Results:")
        for stack_name, stack_result in result.get("stack_results", {}).items():
            if stack_result.get("success"):
                added_count = stack_result.get("templates_added", 0)
                click.echo(f"  ✅ {stack_name}: +{added_count} templates")
            else:
                click.echo(f"  ❌ {stack_name}: Failed")

        # Show errors if any
        if result.get("errors"):
            click.echo("\n❌ Errors:")
            for error in result["errors"][:5]:  # Show first 5 errors
                click.echo(f"  • {error}")
            if len(result["errors"]) > 5:
                click.echo(f"  • ... and {len(result['errors']) - 5} more")

    else:
        # Single stack results
        if result.get("success"):
            added = result.get("templates_added", 0)
            skipped = result.get("templates_skipped", 0)

            click.echo("\n✅ Stack populated successfully!"            click.echo(f"📦 Added: {added} templates")
            click.echo(f"⏭️  Skipped: {skipped} templates")

            if result.get("templates"):
                click.echo("\n📋 Added templates:")
                for template in result["templates"]:
                    click.echo(f"  • {template['name']} (quality: {template['quality_score']:.2f})")

        else:
            click.echo("\n❌ Stack population failed!")
            if result.get("errors"):
                click.echo("Errors:")
                for error in result["errors"]:
                    click.echo(f"  • {error}")
