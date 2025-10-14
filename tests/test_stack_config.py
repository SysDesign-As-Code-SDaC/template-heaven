"""
Tests for Stack Configuration Manager.

This module contains unit tests for the stack configuration system
including validation rules, quality standards, and requirements.
"""

import pytest
from unittest.mock import Mock, patch

from templateheaven.core.stack_config import (
    StackConfigManager,
    StackConfiguration,
    StackRequirements,
    ValidationRules
)
from templateheaven.core.models import Template, StackCategory


class TestStackConfigManager:
    """Test StackConfigManager functionality."""

    @pytest.fixture
    def config_manager(self):
        """Create a stack config manager instance."""
        return StackConfigManager()

    def test_initialization(self, config_manager):
        """Test configuration manager initialization."""
        assert config_manager.configurations is not None
        assert isinstance(config_manager.configurations, dict)

    def test_get_stack_config_existing(self, config_manager):
        """Test getting configuration for existing stack."""
        config = config_manager.get_stack_config('frontend')
        assert config is not None
        assert isinstance(config, StackConfiguration)
        assert config.name == "Frontend Frameworks"

    def test_get_stack_config_nonexistent(self, config_manager):
        """Test getting configuration for nonexistent stack."""
        config = config_manager.get_stack_config('nonexistent')
        assert config is None

    def test_get_all_stacks(self, config_manager):
        """Test getting all configured stacks."""
        stacks = config_manager.get_all_stacks()
        assert isinstance(stacks, list)
        assert len(stacks) > 0
        assert 'frontend' in stacks
        assert 'backend' in stacks

    def test_get_stacks_by_category(self, config_manager):
        """Test getting stacks by category."""
        core_stacks = config_manager.get_stacks_by_category('core')
        assert isinstance(core_stacks, list)
        assert 'frontend' in core_stacks
        assert 'backend' in core_stacks

    def test_get_quality_standards(self, config_manager):
        """Test getting quality standards for a stack."""
        standards = config_manager.get_quality_standards('frontend')
        assert isinstance(standards, list)
        assert len(standards) > 0
        assert "TypeScript support required" in standards

    def test_get_validation_rules(self, config_manager):
        """Test getting validation rules for a stack."""
        rules = config_manager.get_validation_rules('frontend')
        assert isinstance(rules, ValidationRules)
        assert 'package.json' in rules.required_files
        assert 'README.md' in rules.required_files

    def test_get_requirements(self, config_manager):
        """Test getting requirements for a stack."""
        reqs = config_manager.get_requirements('frontend')
        assert isinstance(reqs, StackRequirements)
        assert reqs.min_stars == 300
        assert reqs.min_forks == 30
        assert reqs.min_growth_rate == 0.08

    def test_get_technologies(self, config_manager):
        """Test getting technologies for a stack."""
        tech = config_manager.get_technologies('frontend')
        assert isinstance(tech, list)
        assert 'react' in tech
        assert 'typescript' in tech
        assert 'vite' in tech

    def test_validate_template_for_stack_valid(self, config_manager):
        """Test validating a valid template for a stack."""
        template_data = {
            'stars': 500,
            'forks': 50,
            'growth_rate': 0.10,
            'technologies': ['react', 'typescript']
        }

        result = config_manager.validate_template_for_stack(template_data, 'frontend')

        assert result['valid'] is True
        assert len(result['issues']) == 0
        assert len(result['recommendations']) >= 0

    def test_validate_template_for_stack_invalid(self, config_manager):
        """Test validating an invalid template for a stack."""
        template_data = {
            'stars': 50,  # Below minimum 300
            'forks': 5,   # Below minimum 30
            'growth_rate': 0.01,  # Below minimum 0.08
            'technologies': ['unknown-tech']
        }

        result = config_manager.validate_template_for_stack(template_data, 'frontend')

        assert result['valid'] is False
        assert len(result['issues']) > 0
        assert any('stars' in issue.lower() for issue in result['issues'])
        assert any('forks' in issue.lower() for issue in result['issues'])

    def test_validate_template_for_stack_no_config(self, config_manager):
        """Test validating template for nonexistent stack."""
        template_data = {
            'stars': 100,
            'forks': 10,
            'growth_rate': 0.05,
            'technologies': ['python']
        }

        result = config_manager.validate_template_for_stack(template_data, 'nonexistent')

        assert result['valid'] is True  # Should still be valid
        assert len(result['issues']) == 0

    def test_different_stack_requirements(self, config_manager):
        """Test that different stacks have different requirements."""
        frontend_reqs = config_manager.get_requirements('frontend')
        backend_reqs = config_manager.get_requirements('backend')
        ai_ml_reqs = config_manager.get_requirements('ai-ml')

        # Frontend should have higher star requirements than backend
        assert frontend_reqs.min_stars >= backend_reqs.min_stars

        # AI/ML should have higher growth rate requirements
        assert ai_ml_reqs.min_growth_rate > frontend_reqs.min_growth_rate

    def test_stack_categories(self, config_manager):
        """Test that stacks are properly categorized."""
        # Test core category
        core_stacks = config_manager.get_stacks_by_category('core')
        assert len(core_stacks) >= 4  # frontend, backend, fullstack, mobile
        assert 'frontend' in core_stacks
        assert 'backend' in core_stacks
        assert 'mobile' in core_stacks

        # Test ai-ml category
        ai_stacks = config_manager.get_stacks_by_category('ai-ml')
        assert len(ai_stacks) >= 1  # ai-ml
        assert 'ai-ml' in ai_stacks

        # Test infrastructure category
        infra_stacks = config_manager.get_stacks_by_category('infrastructure')
        assert len(infra_stacks) >= 1  # devops
        assert 'devops' in infra_stacks

        # Test specialized category
        specialized_stacks = config_manager.get_stacks_by_category('specialized')
        assert len(specialized_stacks) >= 1  # web3
        assert 'web3' in specialized_stacks

    def test_technology_alignment(self, config_manager):
        """Test technology alignment validation."""
        # Template with good tech alignment
        good_template = {
            'stars': 400,
            'forks': 40,
            'growth_rate': 0.09,
            'technologies': ['react', 'typescript', 'vite']
        }

        result = config_manager.validate_template_for_stack(good_template, 'frontend')
        assert result['valid'] is True

        # Template with poor tech alignment
        poor_template = {
            'stars': 400,
            'forks': 40,
            'growth_rate': 0.09,
            'technologies': ['php', 'mysql', 'apache']  # Not frontend tech
        }

        result = config_manager.validate_template_for_stack(poor_template, 'frontend')
        assert result['valid'] is True  # Still valid, but should have warnings
        assert len(result['recommendations']) > 0 or len(result['warnings']) > 0


class TestStackConfiguration:
    """Test StackConfiguration dataclass."""

    def test_stack_configuration_creation(self):
        """Test creating a stack configuration."""
        requirements = StackRequirements(
            min_stars=100,
            min_forks=10,
            min_growth_rate=0.05,
            documentation_score=7.0,
            performance_score=7.0,
            security_score=7.0
        )

        validation_rules = ValidationRules(
            required_files=['README.md', 'package.json'],
            required_scripts=['test', 'build'],
            forbidden_patterns=['TODO', 'FIXME']
        )

        config = StackConfiguration(
            name="Test Stack",
            description="Test stack description",
            category="test",
            quality_standards=["Standard 1", "Standard 2"],
            requirements=requirements,
            technologies=["tech1", "tech2"],
            validation_rules=validation_rules
        )

        assert config.name == "Test Stack"
        assert config.category == "test"
        assert len(config.quality_standards) == 2
        assert config.requirements.min_stars == 100
        assert len(config.technologies) == 2
        assert 'README.md' in config.validation_rules.required_files


class TestIntegrationWithTemplateManager:
    """Test integration between stack config and template manager."""

    def test_template_validation_with_stack_config(self):
        """Test that template validation uses stack configurations."""
        from templateheaven.core.template_manager import TemplateManager

        # Create a template that should pass frontend validation
        template = Template(
            name="test-frontend-template",
            stack=StackCategory.FRONTEND,
            description="A test frontend template",
            path="test",
            tags=["react", "typescript"],
            stars=500,
            forks=50,
            growth_rate=0.10,
            technologies=["react", "typescript", "vite"]
        )

        manager = TemplateManager()

        # Should validate successfully
        assert manager.validate_template(template) is True

        # Should pass detailed validation
        detailed_result = manager.validate_template_for_stack(template, 'frontend')
        assert detailed_result['valid'] is True
        assert detailed_result['quality_score'] > 7.0
