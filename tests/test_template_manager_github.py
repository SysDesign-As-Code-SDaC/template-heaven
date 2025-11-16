"""
Tests for GitHub-first discovery behavior in TemplateManager.

These tests verify that the TemplateManager prefers GitHub when configured
and falls back to bundled templates when GitHub returns no results. They
also validate the `include_archived` behavior.
"""

from unittest.mock import Mock, patch
from pathlib import Path
import pytest

from templateheaven.core.template_manager import TemplateManager
from templateheaven.config.settings import Config
from templateheaven.core.models import Template, StackCategory


class DummyCandidate:
    def __init__(self, name, full_name, description, topics=None, archived=False):
        self.repository = {
            'name': name,
            'full_name': full_name,
            'description': description,
            'topics': topics or [],
            'archived': archived,
            'html_url': f'https://github.com/{full_name}',
            'clone_url': f'https://github.com/{full_name}.git',
        }

    def get(self, key, default=None):
        if key == 'repository':
            return self.repository
        return default


@pytest.fixture
def config(tmp_path):
    return Config(config_dir=tmp_path / 'config')


@pytest.fixture
def manager(config):
    return TemplateManager(config)


def test_github_first_discovery_returns_github_templates(manager, monkeypatch):
    # Arrange
    dummy = {'repository': {'name': 'gh-next', 'full_name': 'org/gh-next', 'description': 'A Github template', 'topics': ['fullstack'], 'archived': False}}

    # Mock github_search to return our candidate
    manager.github_search.github_available = True
    async def fake_discover(stack, limit=50):
        return [dummy]
    monkeypatch.setattr(manager.github_search, 'discover_templates_for_stack', fake_discover)

    # Act
    templates = manager.list_templates(stack='fullstack', use_github=True)

    # Assert
    assert any(t.path.startswith('github:') for t in templates)
    assert any('gh-next' in t.name or 'org/gh-next' in t.path for t in templates)
    # Ensure candidate was inferred as FULLSTACK
    assert any(t.stack == StackCategory.FULLSTACK for t in templates)


def test_github_fallback_to_bundled(manager, monkeypatch):
    # Arrange
    manager.github_search.github_available = True
    # Make discover return no candidates
    monkeypatch.setattr(manager.github_search, 'discover_templates_for_stack', lambda stack, limit=50: [])

    # Act
    templates = manager.list_templates(stack='fullstack', use_github=True)

    # Assert: Should fallback to bundled templates
    assert any(t.path.startswith('bundled/') for t in templates)


def test_search_templates_use_github_when_available(manager, monkeypatch):
    # Arrange
    manager.github_search.github_available = True

    class DummySearchResult:
        def __init__(self, template_name, score=0.9):
            self.template = Template(
                name=template_name,
                stack=StackCategory.FULLSTACK,
                description='A GitHub result',
                path=f'github:org/{template_name}',
                tags=['fullstack']
            )
            self.score = score
            self.match_reason = 'GitHub match'

    async def fake_search(query, stack=None, min_stars=50, limit=20):
        return [DummySearchResult('gh-result')]

    monkeypatch.setattr(manager.github_search, 'search_github_templates', fake_search)

    # Act
    results = manager.search_templates('gh-result', use_github=True)

    # Assert
    assert len(results) > 0
    assert any(r.template.path.startswith('github:') for r in results)


def test_github_candidate_archived_filtering(manager, monkeypatch):
    # Arrange
    manager.github_search.github_available = True

    dummy = {'repository': {'name': 'gh-old', 'full_name': 'org/gh-old', 'description': 'Archived GitHub template', 'topics': ['fullstack'], 'archived': True}}

    async def fake_discover(stack, limit=50):
        return [dummy]

    monkeypatch.setattr(manager.github_search, 'discover_templates_for_stack', fake_discover)

    # Act
    default_list = manager.list_templates(stack='fullstack', use_github=True)
    included_list = manager.list_templates(stack='fullstack', use_github=True, include_archived=True)

    # Assert
    assert all(not t.path.startswith('github:org/gh-old') for t in default_list)
    assert any(t.path.startswith('github:org/gh-old') for t in included_list)


def test_prefer_github_default(config):
    # By default, Config sets prefer_github True
    manager = TemplateManager(config)
    assert manager.prefer_github is True


def test_include_archived_flag_filters_templates(manager):
    # Insert an archived template into bundled_templates for testing
    archived_template = Template(
        name='archived-template',
        stack=StackCategory.FULLSTACK,
        description='Archived test template',
        path='bundled/fullstack/archived-template',
        tags=['archived'],
        dependencies={},
        archived=True
    )

    # Append archived template
    manager.bundled_templates.append(archived_template)

    # Default: archived should be filtered out
    default_list = manager.list_templates(stack='fullstack', use_github=False)
    assert all(t.name != 'archived-template' for t in default_list)

    # When include_archived=True, archived should appear
    included_list = manager.list_templates(stack='fullstack', include_archived=True, use_github=False)
    assert any(t.name == 'archived-template' for t in included_list)
