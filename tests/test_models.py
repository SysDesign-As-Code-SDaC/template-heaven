"""
Tests for Template Heaven data models.

This module contains unit tests for the core data models including
Template, ProjectConfig, StackCategory, and related classes.
"""

import pytest
from pathlib import Path

from templateheaven.core.models import (
    Template, ProjectConfig, StackCategory, 
    TemplateSearchResult, TemplateValidationResult
)


class TestStackCategory:
    """Test StackCategory enum."""
    
    def test_stack_category_values(self):
        """Test that stack categories have correct values."""
        assert StackCategory.FRONTEND.value == "frontend"
        assert StackCategory.BACKEND.value == "backend"
        assert StackCategory.AI_ML.value == "ai-ml"
        assert StackCategory.DEVOPS.value == "devops"
    
    def test_get_display_name(self):
        """Test getting display names for stack categories."""
        assert StackCategory.get_display_name(StackCategory.FRONTEND) == "Frontend Frameworks"
        assert StackCategory.get_display_name(StackCategory.BACKEND) == "Backend Services"
        assert StackCategory.get_display_name(StackCategory.AI_ML) == "AI/ML & Data Science"
    
    def test_get_description(self):
        """Test getting descriptions for stack categories."""
        description = StackCategory.get_description(StackCategory.FRONTEND)
        assert "Frontend frameworks" in description
        
        description = StackCategory.get_description(StackCategory.BACKEND)
        assert "Backend services" in description


class TestTemplate:
    """Test Template model."""
    
    def test_template_creation(self):
        """Test creating a template with valid data."""
        template = Template(
            name="test-template",
            stack=StackCategory.FRONTEND,
            description="A test template",
            path="bundled/frontend/test-template",
            tags=["react", "typescript"],
            dependencies={"react": "^18.0.0"}
        )
        
        assert template.name == "test-template"
        assert template.stack == StackCategory.FRONTEND
        assert template.description == "A test template"
        assert template.path == "bundled/frontend/test-template"
        assert template.tags == ["react", "typescript"]
        assert template.dependencies == {"react": "^18.0.0"}
    
    def test_template_validation(self):
        """Test template validation on creation."""
        # Test empty name
        with pytest.raises(ValueError, match="Template name cannot be empty"):
            Template(
                name="",
                stack=StackCategory.FRONTEND,
                description="Test",
                path="test"
            )
        
        # Test empty description
        with pytest.raises(ValueError, match="Template description cannot be empty"):
            Template(
                name="test",
                stack=StackCategory.FRONTEND,
                description="",
                path="test"
            )
        
        # Test empty path
        with pytest.raises(ValueError, match="Template path cannot be empty"):
            Template(
                name="test",
                stack=StackCategory.FRONTEND,
                description="Test",
                path=""
            )
    
    def test_get_display_name(self):
        """Test getting display name for template."""
        template = Template(
            name="react-vite-template",
            stack=StackCategory.FRONTEND,
            description="Test",
            path="test"
        )
        
        assert template.get_display_name() == "React Vite Template"
    
    def test_has_tag(self):
        """Test checking if template has a tag."""
        template = Template(
            name="test",
            stack=StackCategory.FRONTEND,
            description="Test",
            path="test",
            tags=["react", "typescript", "vite"]
        )
        
        assert template.has_tag("react")
        assert template.has_tag("REACT")  # Case insensitive
        assert template.has_tag("typescript")
        assert not template.has_tag("vue")
    
    def test_matches_search(self):
        """Test template search matching."""
        template = Template(
            name="react-vite-template",
            stack=StackCategory.FRONTEND,
            description="React application with Vite build tool",
            path="test",
            tags=["react", "vite", "typescript"]
        )
        
        # Test name match
        assert template.matches_search("react")
        assert template.matches_search("vite")
        
        # Test description match
        assert template.matches_search("application")
        assert template.matches_search("build tool")
        
        # Test tag match
        assert template.matches_search("typescript")
        
        # Test case insensitive
        assert template.matches_search("REACT")
        assert template.matches_search("Application")
        
        # Test no match
        assert not template.matches_search("vue")
        assert not template.matches_search("angular")
    
    def test_to_dict(self):
        """Test converting template to dictionary."""
        template = Template(
            name="test-template",
            stack=StackCategory.FRONTEND,
            description="A test template",
            path="bundled/frontend/test-template",
            tags=["react"],
            dependencies={"react": "^18.0.0"},
            upstream_url="https://github.com/test/test",
            version="1.0.0",
            author="Test Author",
            license="MIT"
        )
        
        data = template.to_dict()
        
        assert data["name"] == "test-template"
        assert data["stack"] == "frontend"
        assert data["description"] == "A test template"
        assert data["tags"] == ["react"]
        assert data["dependencies"] == {"react": "^18.0.0"}
        assert data["upstream_url"] == "https://github.com/test/test"
        assert data["version"] == "1.0.0"
        assert data["author"] == "Test Author"
        assert data["license"] == "MIT"
    
    def test_from_dict(self):
        """Test creating template from dictionary."""
        data = {
            "name": "test-template",
            "stack": "frontend",
            "description": "A test template",
            "path": "bundled/frontend/test-template",
            "tags": ["react"],
            "dependencies": {"react": "^18.0.0"},
            "upstream_url": "https://github.com/test/test",
            "version": "1.0.0",
            "author": "Test Author",
            "license": "MIT"
        }
        
        template = Template.from_dict(data)
        
        assert template.name == "test-template"
        assert template.stack == StackCategory.FRONTEND
        assert template.description == "A test template"
        assert template.tags == ["react"]
        assert template.dependencies == {"react": "^18.0.0"}
        assert template.upstream_url == "https://github.com/test/test"
        assert template.version == "1.0.0"
        assert template.author == "Test Author"
        assert template.license == "MIT"
    
    def test_from_dict_invalid_stack(self):
        """Test creating template from dictionary with invalid stack."""
        data = {
            "name": "test",
            "stack": "invalid-stack",
            "description": "Test",
            "path": "test"
        }
        
        with pytest.raises(ValueError, match="Invalid stack category"):
            Template.from_dict(data)


class TestProjectConfig:
    """Test ProjectConfig model."""
    
    def test_project_config_creation(self, sample_template):
        """Test creating a project configuration."""
        config = ProjectConfig(
            name="test-project",
            directory="/tmp",
            template=sample_template,
            author="Test User",
            license="MIT",
            package_manager="npm"
        )
        
        assert config.name == "test-project"
        assert config.directory == "/tmp"
        assert config.template == sample_template
        assert config.author == "Test User"
        assert config.license == "MIT"
        assert config.package_manager == "npm"
        assert config.version == "0.1.0"  # Default version
    
    def test_project_config_validation(self, sample_template):
        """Test project configuration validation."""
        # Test empty name
        with pytest.raises(ValueError, match="Project name cannot be empty"):
            ProjectConfig(
                name="",
                directory="/tmp",
                template=sample_template
            )
        
        # Test empty directory
        with pytest.raises(ValueError, match="Project directory cannot be empty"):
            ProjectConfig(
                name="test",
                directory="",
                template=sample_template
            )
        
        # Test invalid package manager
        with pytest.raises(ValueError, match="Invalid package manager"):
            ProjectConfig(
                name="test",
                directory="/tmp",
                template=sample_template,
                package_manager="invalid"
            )
    
    def test_get_project_path(self, sample_template):
        """Test getting project path."""
        config = ProjectConfig(
            name="test-project",
            directory="/tmp",
            template=sample_template
        )
        
        path = config.get_project_path()
        assert path == Path("/tmp/test-project")
    
    def test_get_template_variables(self, sample_template):
        """Test getting template variables."""
        config = ProjectConfig(
            name="test-project",
            directory="/tmp",
            template=sample_template,
            author="Test User",
            license="MIT",
            description="A test project"
        )
        
        variables = config.get_template_variables()
        
        assert variables["project_name"] == "test-project"
        assert variables["project_description"] == "A test project"
        assert variables["author"] == "Test User"
        assert variables["license"] == "MIT"
        assert variables["version"] == "0.1.0"
        assert variables["package_manager"] == "npm"
        assert variables["template_name"] == sample_template.name
        assert variables["template_stack"] == sample_template.stack.value
    
    def test_get_template_variables_with_custom(self, sample_template):
        """Test getting template variables with custom variables."""
        config = ProjectConfig(
            name="test-project",
            directory="/tmp",
            template=sample_template,
            custom_variables={"custom_key": "custom_value"}
        )
        
        variables = config.get_template_variables()
        
        assert variables["custom_key"] == "custom_value"
        assert variables["project_name"] == "test-project"
    
    def test_to_dict(self, sample_template):
        """Test converting project config to dictionary."""
        config = ProjectConfig(
            name="test-project",
            directory="/tmp",
            template=sample_template,
            author="Test User",
            license="MIT"
        )
        
        data = config.to_dict()
        
        assert data["name"] == "test-project"
        assert data["directory"] == "/tmp"
        assert data["author"] == "Test User"
        assert data["license"] == "MIT"
        assert data["package_manager"] == "npm"
        assert data["version"] == "0.1.0"
        assert "template" in data


class TestTemplateSearchResult:
    """Test TemplateSearchResult model."""
    
    def test_search_result_creation(self, sample_template):
        """Test creating a search result."""
        result = TemplateSearchResult(
            template=sample_template,
            score=0.8,
            match_reason="Name match"
        )
        
        assert result.template == sample_template
        assert result.score == 0.8
        assert result.match_reason == "Name match"
    
    def test_search_result_validation(self, sample_template):
        """Test search result validation."""
        # Test score too high
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            TemplateSearchResult(
                template=sample_template,
                score=1.5
            )
        
        # Test score too low
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            TemplateSearchResult(
                template=sample_template,
                score=-0.1
            )


class TestTemplateValidationResult:
    """Test TemplateValidationResult model."""
    
    def test_validation_result_creation(self):
        """Test creating a validation result."""
        result = TemplateValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.issues == []
        assert result.warnings == []
    
    def test_add_issue(self):
        """Test adding validation issues."""
        result = TemplateValidationResult(is_valid=True)
        
        result.add_issue("Test issue")
        
        assert result.is_valid is False
        assert "Test issue" in result.issues
    
    def test_add_warning(self):
        """Test adding validation warnings."""
        result = TemplateValidationResult(is_valid=True)
        
        result.add_warning("Test warning")
        
        assert result.is_valid is True  # Warnings don't make it invalid
        assert "Test warning" in result.warnings
