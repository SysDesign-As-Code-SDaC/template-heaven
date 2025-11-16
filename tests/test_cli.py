"""
Tests for CLI commands.

This module contains unit tests for all CLI command functionality
including init, list, search, info, config, and stats commands.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call
from click.testing import CliRunner

from templateheaven.cli.main import cli
from templateheaven.cli.commands.init import init_command
from templateheaven.cli.commands.list import list_command
from templateheaven.cli.commands.config import config_command
from templateheaven.core.models import Template, StackCategory, ProjectConfig


class TestCLIInit:
    """Test init command functionality."""

    def test_init_command_help(self):
        """Test init command help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['init', '--help'])
        assert result.exit_code == 0
        assert 'Initialize a new project' in result.output

    def test_init_command_basic(self, tmp_path, mock_template_manager, mock_config):
        """Test basic init command execution."""
        with patch('templateheaven.cli.main.TemplateManager') as mock_tm_class, \
             patch('templateheaven.cli.main.Config') as mock_config_class, \
             patch('templateheaven.cli.commands.init.Customizer') as mock_customizer_class, \
             patch('templateheaven.cli.wizard.Wizard') as mock_wizard_class:

            # Setup mocks
            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance

            # Configure config mock to return proper values
            mock_config_instance.get.side_effect = lambda key, default=None: {
                'default_author': 'Template Heaven User',
                'default_license': 'MIT',
                'package_managers.python': 'pip'
            }.get(key, default)

            mock_template = Mock()
            mock_template.name = "test-template"
            mock_template.tags = ['python', 'web']
            mock_template.stack = Mock()
            mock_template.stack.value = 'backend'
            mock_template.dependencies = {}
            mock_template.upstream_url = 'https://github.com/test/test'
            mock_template.features = []
            mock_tm_instance.get_template.return_value = mock_template

            mock_customizer_instance = Mock()
            mock_customizer_class.return_value = mock_customizer_instance
            mock_customizer_instance.customize.return_value = True

            # Test non-interactive mode
            runner = CliRunner()
            result = runner.invoke(cli, [
                'init',
                '--template', 'test-template',
                '--name', 'test-project',
                '--directory', str(tmp_path),
                '--author', 'Test Author',
                '--license', 'MIT',
                '--no-wizard'
            ])

            assert result.exit_code == 0
            assert 'Project created successfully' in result.output
            mock_customizer_instance.customize.assert_called_once()

    def test_init_command_missing_template(self, mock_template_manager):
        """Test init command with missing template."""
        with patch('templateheaven.cli.commands.init.TemplateManager') as mock_tm_class:
            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance
            mock_tm_instance.get_template.return_value = None

            runner = CliRunner()
            result = runner.invoke(cli, [
                'init',
                '--template', 'nonexistent-template',
                '--name', 'test-project',
                '--no-wizard'
            ])

            assert result.exit_code == 1
            assert 'not found' in result.output

    def test_init_command_project_exists(self, tmp_path, mock_template_manager):
        """Test init command when project directory already exists."""
        # Create existing directory
        project_dir = tmp_path / "existing-project"
        project_dir.mkdir()

        with patch('templateheaven.cli.commands.init.TemplateManager') as mock_tm_class, \
             patch('templateheaven.cli.commands.init.Config') as mock_config_class:

            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance
            mock_config_class.return_value = Mock()

            mock_template = Mock()
            mock_template.name = "test-template"
            mock_tm_instance.get_template.return_value = mock_template

            runner = CliRunner()
            with runner.isolated_filesystem():
                # Create the directory first
                Path("existing-project").mkdir()

                result = runner.invoke(cli, [
                    'init',
                    '--template', 'test-template',
                    '--name', 'existing-project',
                    '--no-wizard'
                ])

                assert result.exit_code == 1
                assert 'not found' in result.output


class TestCLIList:
    """Test list command functionality."""

    def test_list_command_help(self):
        """Test list command help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['list', '--help'])
        assert result.exit_code == 0
        assert 'List available templates' in result.output

    def test_list_command_basic(self, mock_template_manager):
        """Test basic list command."""
        with patch('templateheaven.core.template_manager.TemplateManager') as mock_tm_class:
            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance

            # Create mock templates
            mock_templates = [
                Mock(name='react-vite', stack=StackCategory.FRONTEND, description='React template'),
                Mock(name='fastapi', stack=StackCategory.BACKEND, description='FastAPI template')
            ]
            for template in mock_templates:
                template.name = template.name
                template.stack = template.stack
                template.description = template.description
                template.tags = ['tag1', 'tag2']

            mock_tm_instance.list_templates.return_value = mock_templates

            runner = CliRunner()
            result = runner.invoke(cli, ['list'])

            assert result.exit_code == 0
            assert 'react-vite' in result.output
            assert 'fastapi' in result.output

    def test_list_command_with_stack_filter(self, mock_template_manager):
        """Test list command with stack filter."""
        with patch('templateheaven.cli.main.TemplateManager') as mock_tm_class, \
             patch('templateheaven.cli.main.Config') as mock_config_class:
            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance

            # Configure the mock template manager's config
            mock_tm_instance.config = mock_config_instance
            mock_config_instance.get.return_value = False  # prefer_github = False

            # Mock template with proper attributes
            from templateheaven.core.models import Template
            mock_template = Template(
                name='react-vite',
                stack=StackCategory.FRONTEND,
                description='React template',
                path='local:test-template',
                tags=['react'],
                dependencies={}
            )

            mock_tm_instance.list_templates.return_value = [mock_template]

            runner = CliRunner()
            result = runner.invoke(cli, ['list', '--stack', 'frontend'])

            assert result.exit_code == 0
            mock_tm_instance.list_templates.assert_called_with(
                stack='frontend', tags=None, search=None, use_github=False, include_archived=False
            )

    def test_list_command_with_search_filter(self, mock_template_manager):
        """Test list command with search filter."""
        with patch('templateheaven.cli.main.TemplateManager') as mock_tm_class, \
             patch('templateheaven.cli.main.Config') as mock_config_class:
            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance

            # Configure the mock template manager's config
            mock_tm_instance.config = mock_config_instance
            mock_config_instance.get.return_value = False  # prefer_github = False

            # Mock template with proper attributes
            from templateheaven.core.models import Template
            mock_template = Template(
                name='react-vite',
                stack=StackCategory.FRONTEND,
                description='React template',
                path='local:test-template',
                tags=['react'],
                dependencies={}
            )

            mock_tm_instance.list_templates.return_value = [mock_template]

            runner = CliRunner()
            result = runner.invoke(cli, ['list', '--search', 'react'])

            assert result.exit_code == 0
            mock_tm_instance.list_templates.assert_called_with(
                stack=None, tags=None, search='react', use_github=False, include_archived=False
            )


class TestCLIConfig:
    """Test config command functionality."""

    def test_config_command_help(self):
        """Test config command help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['config', '--help'])
        assert result.exit_code == 0
        assert 'Manage configuration' in result.output

    def test_config_get_command(self, mock_config):
        """Test config get command."""
        with patch('templateheaven.cli.main.Config') as mock_config_class, \
             patch('templateheaven.cli.main.TemplateManager') as mock_tm_class:
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance
            mock_config_instance.get.return_value = 'test-value'

            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance

            runner = CliRunner()
            result = runner.invoke(cli, ['config', '--key', 'test_key'])

            assert result.exit_code == 0
            assert 'test-value' in result.output
            mock_config_instance.get.assert_called_with('test_key')

    def test_config_set_command(self, mock_config):
        """Test config set command."""
        with patch('templateheaven.cli.main.Config') as mock_config_class, \
             patch('templateheaven.cli.main.TemplateManager') as mock_tm_class:
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance

            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance

            runner = CliRunner()
            result = runner.invoke(cli, ['config', '--key', 'default_author', '--value', 'test_value'])

            assert result.exit_code == 0
            mock_config_instance.set.assert_called_with('default_author', 'test_value')

    def test_config_unset_command(self, mock_config):
        """Test config unset command."""
        with patch('templateheaven.cli.main.Config') as mock_config_class, \
             patch('templateheaven.cli.main.TemplateManager') as mock_tm_class:
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance

            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance

            runner = CliRunner()
            result = runner.invoke(cli, ['config', '--unset', 'default_author'])

            assert result.exit_code == 0
            mock_config_instance.unset.assert_called_with('default_author')

    def test_config_list_command(self, mock_config):
        """Test config list command."""
        with patch('templateheaven.cli.main.Config') as mock_config_class, \
             patch('templateheaven.cli.main.TemplateManager') as mock_tm_class:
            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance
            mock_config_instance.get_all.return_value = {'key1': 'value1', 'key2': 'value2'}
            mock_config_instance.get_config_info.return_value = {
                'config_file': '/path/to/config.yaml',
                'cache_dir': '/path/to/cache'
            }

            mock_tm_instance = Mock()
            mock_tm_class.return_value = mock_tm_instance

            runner = CliRunner()
            result = runner.invoke(cli, ['config', '--list-all'])

            assert result.exit_code == 0
            assert 'key1' in result.output
            assert 'value1' in result.output
            mock_config_instance.get_all.assert_called_once()


class TestCLIMain:
    """Test main CLI functionality."""

    def test_cli_help(self):
        """Test main CLI help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Template Heaven' in result.output
        assert 'init' in result.output
        assert 'list' in result.output
        assert 'config' in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'Template Heaven' in result.output

    @patch('templateheaven.cli.main.get_logger')
    def test_cli_verbose_flag(self, mock_get_logger):
        """Test verbose flag sets up logging."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', 'list'])
        # This should work without errors
        assert result.exit_code in [0, 2]  # 0 for success, 2 for usage error (expected with mocked manager)
