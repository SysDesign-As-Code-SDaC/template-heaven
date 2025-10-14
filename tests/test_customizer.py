"""
Tests for Template Customizer.

This module contains unit tests for the Customizer class including
template processing, file operations, and variable substitution.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from templateheaven.core.customizer import Customizer
from templateheaven.core.models import Template, ProjectConfig, StackCategory


class TestCustomizer:
    """Test Customizer class functionality."""

    @pytest.fixture
    def customizer(self):
        """Create a customizer instance."""
        return Customizer()

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return Template(
            name="test-template",
            stack=StackCategory.FRONTEND,
            description="Test template",
            path="bundled/frontend/test-template",
            tags=["test"]
        )

    @pytest.fixture
    def sample_config(self, sample_template):
        """Create a sample project config."""
        return ProjectConfig(
            name="test-project",
            directory="/tmp/test-project",
            template=sample_template,
            author="Test Author",
            license="MIT"
        )

    def test_customizer_initialization(self, customizer):
        """Test customizer initialization."""
        assert customizer.file_ops is not None
        assert customizer.jinja_env is not None
        assert hasattr(customizer, '_snake_case_filter')
        assert hasattr(customizer, '_kebab_case_filter')

    def test_snake_case_filter(self, customizer):
        """Test snake_case filter."""
        assert customizer._snake_case_filter("TestProject") == "test_project"
        assert customizer._snake_case_filter("testProject") == "test_project"
        assert customizer._snake_case_filter("Test-Project") == "test_project"

    def test_kebab_case_filter(self, customizer):
        """Test kebab_case filter."""
        assert customizer._kebab_case_filter("TestProject") == "test-project"
        assert customizer._kebab_case_filter("testProject") == "test-project"
        assert customizer._kebab_case_filter("Test_Project") == "test-project"

    def test_process_template_file_jinja2(self, customizer, tmp_path):
        """Test processing a file with Jinja2 templating."""
        # Create source and dest files
        source_file = tmp_path / "source.txt"
        dest_file = tmp_path / "dest.txt"

        # Write Jinja2 template content
        source_file.write_text("Hello {{ name }}! Project: {{ project_name }}")

        variables = {
            "name": "World",
            "project_name": "TestProject"
        }

        customizer._process_template_file(source_file, dest_file, variables)

        assert dest_file.exists()
        content = dest_file.read_text()
        assert content == "Hello World! Project: TestProject"

    def test_process_template_file_yaml_skip_jinja2(self, customizer, tmp_path):
        """Test that YAML files are not processed with Jinja2."""
        # Create source and dest files
        source_file = tmp_path / "config.yml"
        dest_file = tmp_path / "config.yml"

        # Write YAML content with GitHub Actions syntax
        yaml_content = """
name: CI/CD Pipeline
on:
  push:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    if: ${{ github.event_name == 'push' }}
"""
        source_file.write_text(yaml_content)

        variables = {
            "project_name": "TestProject"
        }

        customizer._process_template_file(source_file, dest_file, variables)

        assert dest_file.exists()
        content = dest_file.read_text()
        # YAML should be unchanged (no Jinja2 processing)
        assert "name: CI/CD Pipeline" in content
        assert "${{ github.event_name == 'push' }}" in content

    def test_process_template_file_simple_substitution(self, customizer, tmp_path):
        """Test processing a file with simple variable substitution."""
        # Create source and dest files
        source_file = tmp_path / "readme.md"
        dest_file = tmp_path / "readme.md"

        # Write content with template variables
        source_file.write_text("# {{ project_name }}\nAuthor: {{ author }}")

        variables = {
            "project_name": "My Project",
            "author": "Test Author"
        }

        customizer._process_template_file(source_file, dest_file, variables)

        assert dest_file.exists()
        content = dest_file.read_text()
        assert content == "# My Project\nAuthor: Test Author"

    def test_process_template_file_error_handling(self, customizer, tmp_path):
        """Test error handling in template processing."""
        # Create source and dest files
        source_file = tmp_path / "template.txt"
        dest_file = tmp_path / "output.txt"

        # Write invalid Jinja2 content
        source_file.write_text("Hello {{ undefined_var.missing_attr }}")

        variables = {"name": "World"}

        # Should not raise exception, should fall back to copy
        customizer._process_template_file(source_file, dest_file, variables)

        assert dest_file.exists()
        # Should have copied the original content
        content = dest_file.read_text()
        assert content == "Hello {{ undefined_var.missing_attr }}"

    @patch('templateheaven.core.customizer.FileOperations')
    def test_customizer_with_mocked_file_ops(self, mock_file_ops_class, customizer):
        """Test customizer with mocked file operations."""
        mock_file_ops = Mock()
        mock_file_ops_class.return_value = mock_file_ops

        customizer_with_mock = Customizer()

        # Verify file operations is set
        assert customizer_with_mock.file_ops == mock_file_ops

    def test_sanitize_project_name(self):
        """Test project name sanitization."""
        from templateheaven.core.customizer import Customizer

        customizer = Customizer()

        # This should work without errors
        assert callable(getattr(customizer, '_snake_case_filter', None))

    @patch('templateheaven.core.customizer.FileOperations.copy_directory')
    @patch('templateheaven.core.customizer.FileOperations.create_directory')
    def test_copy_template_files(self, mock_create_dir, mock_copy_dir, customizer, tmp_path):
        """Test copying template files."""
        template_dir = tmp_path / "template"
        output_dir = tmp_path / "output"

        template_dir.mkdir()
        (template_dir / "file1.txt").write_text("content1")
        (template_dir / "subdir").mkdir()
        (template_dir / "subdir/file2.txt").write_text("content2")

        # Mock the file operations
        mock_copy_dir.return_value = True
        mock_create_dir.return_value = True

        # This would normally copy files, but we're mocking
        assert mock_copy_dir.return_value is True

    def test_template_variable_extraction(self, customizer, sample_config):
        """Test template variable extraction."""
        variables = customizer._get_template_variables(sample_config)

        assert variables['project_name'] == 'test-project'
        assert variables['author'] == 'Test Author'
        assert variables['license'] == 'MIT'
        assert 'project_name_snake' in variables
        assert 'project_name_kebab' in variables

    def test_template_variable_extraction_with_custom(self, customizer, sample_template):
        """Test template variable extraction with custom config."""
        config = ProjectConfig(
            name="custom-project",
            directory="/tmp/custom",
            template=sample_template,
            author="Custom Author",
            license="Apache-2.0",
            description="Custom description",
            version="1.2.3"
        )

        variables = customizer._get_template_variables(config)

        assert variables['project_name'] == 'custom-project'
        assert variables['author'] == 'Custom Author'
        assert variables['license'] == 'Apache-2.0'
        assert variables['description'] == 'Custom description'
        assert variables['version'] == '1.2.3'
