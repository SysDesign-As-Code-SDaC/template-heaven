"""
Tests for the Wizard interactive behavior regarding search source defaults.
"""

from unittest.mock import Mock
from pathlib import Path
import questionary

from templateheaven.core.template_manager import TemplateManager
from templateheaven.config.settings import Config
from templateheaven.cli.wizard import Wizard
from templateheaven.core.models import Template, StackCategory, TemplateSearchResult


def test_wizard_search_default_choice(tmp_path, monkeypatch):
    # Set up config and manager preferring GitHub
    config = Config(config_dir=tmp_path / 'config')
    manager = TemplateManager(config)
    manager.prefer_github = True
    manager.github_search.github_available = True

    # Monkeypatch questionary.text and select
    monkeypatch.setattr(questionary, 'text', lambda *args, **kwargs: Mock(ask=lambda: 'query'))

    # Capture select default passed
    def fake_select(message, choices, default=None, **kwargs):
        # Only assert the default for the search prompt
        if 'Where should we search' in message:
            assert default == 'github'
            return Mock(ask=lambda: 'local')  # return local to avoid GitHub path
        # For other selects (stack selection), default to the first choice
        return Mock(ask=lambda: choices[0])

    monkeypatch.setattr(questionary, 'select', fake_select)

    # Prepare Wizard with manager that returns a simple result
    wizard = Wizard(manager, config)

    # Patch local search to return an empty list to force fallback logic
    manager.search_templates = lambda q, limit=10: []

    # Call _search_all_templates and ensure no exceptions (we return to stack selection)
    result = wizard._search_all_templates()
    # The method may return a StackCategory or a (choice, stack) tuple depending
    # on the mocked choices; normalize and assert the stack is valid.
    if isinstance(result, tuple):
        assert isinstance(result[1], StackCategory)
    else:
        assert isinstance(result, StackCategory)
