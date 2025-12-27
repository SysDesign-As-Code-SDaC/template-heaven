"""
Core module for Template Heaven.

This module contains the core functionality for template management,
including data models, template manager, and customization logic.
"""

from .models import Template, ProjectConfig, StackCategory
from .template_manager import TemplateManager
from .customizer import Customizer

__all__ = [
    "Template",
    "ProjectConfig", 
    "StackCategory",
    "TemplateManager",
    "Customizer",
]
