"""
Test CLI source preference logic for `list` command.
"""

from click.testing import CliRunner
from templateheaven.cli.main import cli
from templateheaven.config.settings import Config
from templateheaven.core.template_manager import TemplateManager
from pathlib import Path
import pytest


class FakeManager(TemplateManager):
    def __init__(self, config):
        super().__init__(config)
        self.last_list_args = None

    def list_templates(self, *args, **kwargs):
        # Record kwargs and return empty list
        self.last_list_args = kwargs
        return []


def test_cli_list_auto_prefers_github(tmp_path, monkeypatch):
    # Prepare runner and mock TemplateManager to an instance preferring GitHub
    runner = CliRunner()
    cfg = Config(config_dir=tmp_path / 'config')
    created = []
    def factory(config):
        m = FakeManager(config)
        created.append(m)
        return m
    monkeypatch.setattr('templateheaven.cli.main.TemplateManager', factory)

    result = runner.invoke(cli, ['list', '--source', 'auto'])
    assert result.exit_code == 0
    # Ensure TemplateManager.list_templates was invoked and that use_github was True by default
    assert created
    assert created[0].last_list_args is not None
    assert created[0].last_list_args.get('use_github') is True


def test_cli_list_include_archived_flag(tmp_path, monkeypatch):
    runner = CliRunner()
    cfg = Config(config_dir=tmp_path / 'config')
    created = []
    def factory(config):
        m = FakeManager(config)
        created.append(m)
        return m
    monkeypatch.setattr('templateheaven.cli.main.TemplateManager', factory)

    result = runner.invoke(cli, ['list', '--source', 'local', '--include-archived'])
    assert result.exit_code == 0
    assert created
    assert created[0].last_list_args.get('include_archived') is True
