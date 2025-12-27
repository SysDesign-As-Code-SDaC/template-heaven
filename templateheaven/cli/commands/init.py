"""
Init command implementation for Template Heaven CLI.

This module provides the command-line implementation for project initialization
without the interactive wizard.
"""

from pathlib import Path
from typing import Optional

from ...core.template_manager import TemplateManager
from ...core.models import ProjectConfig
from ...core.customizer import Customizer
from ...config.settings import Config
from ...utils.logger import get_logger
from ...utils.helpers import validate_project_name, sanitize_project_name

logger = get_logger(__name__)


def init_command(
    template_manager: TemplateManager,
    config: Config,
    template_name: str,
    project_name: str,
    stack: Optional[str] = None,
    output_dir: Path = Path('.'),
    author: Optional[str] = None,
    license: Optional[str] = None,
    package_manager: Optional[str] = None
) -> None:
    """
    Initialize a project from command-line arguments.
    
    Args:
        template_manager: Template manager instance
        config: Configuration instance
        template_name: Name of template to use
        project_name: Name of the project
        stack: Stack category (optional)
        output_dir: Output directory
        author: Project author
        license: Project license
        package_manager: Package manager to use
        
    Raises:
        ValueError: If template not found or invalid parameters
        OSError: If file operations fail
    """
    logger.info(f"Initializing project '{project_name}' with template '{template_name}'")
    
    # Validate project name
    try:
        validate_project_name(project_name)
    except ValueError as e:
        raise ValueError(f"Invalid project name: {e}")
    
    # Get template
    template = template_manager.get_template(template_name)
    if not template:
        available_templates = [t.name for t in template_manager.list_templates()]
        raise ValueError(
            f"Template '{template_name}' not found. "
            f"Available templates: {', '.join(available_templates[:10])}"
        )
    
    # Validate stack if provided
    if stack and template.stack.value != stack:
        logger.warning(f"Template '{template_name}' is in stack '{template.stack.value}', not '{stack}'")
    
    # Get configuration values
    author = author or config.get('default_author', 'Template Heaven User')
    license = license or config.get('default_license', 'MIT')
    
    # Determine package manager
    if not package_manager:
        if any(tag in template.tags for tag in ['python', 'fastapi', 'django', 'pytorch']):
            package_manager = config.get('package_managers.python', 'pip')
        elif any(tag in template.tags for tag in ['nodejs', 'react', 'vue', 'typescript', 'nextjs']):
            package_manager = config.get('package_managers.node', 'npm')
        elif any(tag in template.tags for tag in ['rust']):
            package_manager = config.get('package_managers.rust', 'cargo')
        elif any(tag in template.tags for tag in ['go']):
            package_manager = config.get('package_managers.go', 'go')
        else:
            package_manager = 'npm'  # Default
    
    # Create project configuration
    project_config = ProjectConfig(
        name=project_name,
        directory=str(output_dir),
        template=template,
        author=author,
        license=license,
        package_manager=package_manager
    )
    
    # Initialize customizer
    customizer = Customizer()
    
    # Create project
    try:
        success = customizer.customize(template, project_config, output_dir)
        
        if success:
            project_path = output_dir / project_name
            logger.info(f"Project created successfully: {project_path}")
            
            # Display next steps
            _display_next_steps(project_path, template, package_manager)
        else:
            raise RuntimeError("Project creation failed")
            
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise


def _display_next_steps(project_path: Path, template, package_manager: str) -> None:
    """
    Display next steps for the user.
    
    Args:
        project_path: Path to the created project
        template: Template that was used
        package_manager: Package manager used
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    # Create next steps text
    steps = [
        f"cd {project_path.name}",
    ]
    
    # Add installation step
    if package_manager in ['npm', 'yarn', 'pnpm']:
        steps.append(f"{package_manager} install")
    elif package_manager == 'pip':
        steps.append("pip install -r requirements.txt")
    elif package_manager == 'poetry':
        steps.append("poetry install")
    elif package_manager == 'cargo':
        steps.append("cargo build")
    elif package_manager == 'go':
        steps.append("go mod tidy")
    
    # Add development step
    if any(tag in template.tags for tag in ['react', 'vue', 'nextjs', 'typescript']):
        steps.append("npm run dev")
    elif any(tag in template.tags for tag in ['python', 'fastapi', 'django']):
        steps.append("python app.py  # or python main.py")
    elif any(tag in template.tags for tag in ['rust']):
        steps.append("cargo run")
    elif any(tag in template.tags for tag in ['go']):
        steps.append("go run main.go")
    
    steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(steps))
    
    # Display panel
    panel_content = Text(steps_text, style="white")
    console.print(Panel(panel_content, title="Next Steps", border_style="green"))
    
    # Additional information
    if template.upstream_url:
        console.print(f"\n[cyan]Upstream Source:[/cyan] {template.upstream_url}")
    
    if template.features:
        features_text = ", ".join(template.features)
        console.print(f"[cyan]Template Features:[/cyan] {features_text}")
