"""
Stack Configuration Manager for Template Heaven.

This module manages stack-specific configurations, quality standards,
and validation rules for different technology stacks.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .models import StackCategory
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StackRequirements:
    """Requirements for a technology stack."""
    min_stars: int
    min_forks: int
    min_growth_rate: float
    documentation_score: float
    performance_score: float
    security_score: float


@dataclass
class ValidationRules:
    """Validation rules for a technology stack."""
    required_files: List[str]
    required_scripts: List[str]
    forbidden_patterns: List[str]


@dataclass
class StackConfiguration:
    """Complete configuration for a technology stack."""
    name: str
    description: str
    category: str
    quality_standards: List[str]
    requirements: StackRequirements
    technologies: List[str]
    validation_rules: ValidationRules


class StackConfigManager:
    """
    Manages stack-specific configurations and validation rules.

    Provides access to quality standards, requirements, and validation
    rules that are customized for each technology stack.
    """

    def __init__(self):
        """Initialize the stack configuration manager."""
        self.configurations: Dict[str, StackConfiguration] = {}
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load stack configurations from YAML file."""
        try:
            config_path = Path(__file__).parent.parent / "data" / "stack_configurations.yaml"

            if not config_path.exists():
                logger.warning(f"Stack configuration file not found: {config_path}")
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Load stack configurations
            for stack_name, stack_data in data.get('stacks', {}).items():
                try:
                    config = self._parse_stack_config(stack_name, stack_data)
                    self.configurations[stack_name] = config
                    logger.debug(f"Loaded configuration for stack: {stack_name}")
                except Exception as e:
                    logger.error(f"Failed to parse configuration for stack {stack_name}: {e}")

            logger.info(f"Loaded {len(self.configurations)} stack configurations")

        except Exception as e:
            logger.error(f"Failed to load stack configurations: {e}")

    def _parse_stack_config(self, stack_name: str, data: Dict[str, Any]) -> StackConfiguration:
        """Parse stack configuration from dictionary."""
        requirements_data = data.get('requirements', {})
        validation_data = data.get('validation_rules', {})

        requirements = StackRequirements(
            min_stars=requirements_data.get('min_stars', 100),
            min_forks=requirements_data.get('min_forks', 10),
            min_growth_rate=requirements_data.get('min_growth_rate', 0.05),
            documentation_score=requirements_data.get('documentation_score', 7.0),
            performance_score=requirements_data.get('performance_score', 7.0),
            security_score=requirements_data.get('security_score', 7.0)
        )

        validation_rules = ValidationRules(
            required_files=validation_data.get('required_files', []),
            required_scripts=validation_data.get('required_scripts', []),
            forbidden_patterns=validation_data.get('forbidden_patterns', [])
        )

        return StackConfiguration(
            name=data.get('name', stack_name),
            description=data.get('description', ''),
            category=data.get('category', 'other'),
            quality_standards=data.get('quality_standards', []),
            requirements=requirements,
            technologies=data.get('technologies', []),
            validation_rules=validation_rules
        )

    def get_stack_config(self, stack_name: str) -> Optional[StackConfiguration]:
        """
        Get configuration for a specific stack.

        Args:
            stack_name: Name of the stack

        Returns:
            Stack configuration or None if not found
        """
        return self.configurations.get(stack_name)

    def get_all_stacks(self) -> List[str]:
        """Get list of all configured stacks."""
        return list(self.configurations.keys())

    def get_stacks_by_category(self, category: str) -> List[str]:
        """Get stacks belonging to a specific category."""
        return [name for name, config in self.configurations.items()
                if config.category == category]

    def validate_template_for_stack(self, template_data: Dict[str, Any], stack_name: str) -> Dict[str, Any]:
        """
        Validate template data against stack-specific requirements.

        Args:
            template_data: Template metadata
            stack_name: Stack to validate against

        Returns:
            Validation results with issues and recommendations
        """
        config = self.get_stack_config(stack_name)
        if not config:
            return {
                'valid': True,
                'issues': [],
                'recommendations': ['Stack configuration not found - using default validation']
            }

        issues = []
        recommendations = []

        # Check star requirements
        stars = template_data.get('stars', 0)
        if stars < config.requirements.min_stars:
            issues.append(f"Template has {stars} stars, minimum required: {config.requirements.min_stars}")

        # Check fork requirements
        forks = template_data.get('forks', 0)
        if forks < config.requirements.min_forks:
            issues.append(f"Template has {forks} forks, minimum required: {config.requirements.min_forks}")

        # Check growth rate
        growth_rate = template_data.get('growth_rate', 0.0)
        if growth_rate < config.requirements.min_growth_rate:
            issues.append(f"Template growth rate {growth_rate:.1%}, minimum required: {config.requirements.min_growth_rate:.1%}")

        # Check technologies
        template_tech = set(template_data.get('technologies', []))
        stack_tech = set(config.technologies)
        if template_tech and not template_tech.intersection(stack_tech):
            recommendations.append(f"Template technologies {template_tech} don't match stack technologies {stack_tech}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'quality_standards': config.quality_standards
        }

    def get_quality_standards(self, stack_name: str) -> List[str]:
        """Get quality standards for a stack."""
        config = self.get_stack_config(stack_name)
        return config.quality_standards if config else []

    def get_validation_rules(self, stack_name: str) -> Optional[ValidationRules]:
        """Get validation rules for a stack."""
        config = self.get_stack_config(stack_name)
        return config.validation_rules if config else None

    def get_requirements(self, stack_name: str) -> Optional[StackRequirements]:
        """Get requirements for a stack."""
        config = self.get_stack_config(stack_name)
        return config.requirements if config else None

    def get_technologies(self, stack_name: str) -> List[str]:
        """Get supported technologies for a stack."""
        config = self.get_stack_config(stack_name)
        return config.technologies if config else []


# Global instance
stack_config_manager = StackConfigManager()


def get_stack_config_manager() -> StackConfigManager:
    """Get the global stack configuration manager instance."""
    return stack_config_manager
