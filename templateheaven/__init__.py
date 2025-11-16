"""
Template Heaven - Interactive template management package.

This package provides a comprehensive solution for managing and initializing
projects from templates, with support for multiple technology stacks and
automated template discovery.

Key Features:
- Interactive CLI for template selection and project initialization
- Support for 24+ technology stack categories
- Local template bundling with caching
- Wizard-style project setup
- Configuration management
- Template customization with Jinja2

Example Usage:
    # CLI usage
    templateheaven init
    templateheaven list --stack frontend
    
    # Python API usage
    from templateheaven import TemplateManager, Wizard
    
    manager = TemplateManager()
    templates = manager.list_templates(stack='frontend')
    
    wizard = Wizard()
    config = wizard.run()

Author: Template Heaven Team
License: MIT
Version: 0.1.0
"""

from .__version__ import __version__

# Core imports for public API
from .core.template_manager import TemplateManager
from .core.models import Template, ProjectConfig, StackCategory
from .config.settings import Config

__all__ = [
    "__version__",
    "TemplateManager",
    "Template",
    "ProjectConfig", 
    "StackCategory",
    "Config",
]

# Package metadata
__author__ = "Template Heaven Team"
__email__ = "team@templateheaven.dev"
__license__ = "MIT"
__url__ = "https://github.com/template-heaven/templateheaven"
