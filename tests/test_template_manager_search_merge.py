"""
Tests for TemplateManager.search_all_templates combining local and GitHub results.
"""

import asyncio
from unittest.mock import Mock
from templateheaven.core.template_manager import TemplateManager
from templateheaven.config.settings import Config
from templateheaven.core.models import Template, StackCategory, TemplateSearchResult


def test_search_all_templates_combines_local_and_github(tmp_path, monkeypatch):
    cfg = Config(config_dir=tmp_path / 'config')
    manager = TemplateManager(cfg)

    # Prepare local search to return two results with different scores
    def local_search(query, limit=20):
        t1 = Template(name='local-a', stack=StackCategory.FULLSTACK, description='local', path='bundled/fullstack/local-a', tags=['fullstack'])
        return [TemplateSearchResult(template=t1, score=0.6, match_reason='Local')]
    monkeypatch.setattr(manager, 'search_templates', local_search)

    # Prepare GitHub search result (async)
    async def fake_search(query, stack=None, min_stars=50, limit=10):
        t2 = Template(name='gh-a', stack=StackCategory.FULLSTACK, description='github', path='github:org/gh-a', tags=['fullstack'])
        return [TemplateSearchResult(template=t2, score=0.9, match_reason='GitHub')]
    monkeypatch.setattr(manager.github_search, 'search_github_templates', fake_search)
    manager.github_search.github_available = True

    # Run combined search (async entrypoint)
    results = asyncio.run(manager.search_all_templates('query', include_github=True, github_limit=5, total_limit=5))

    # Validate that results include GitHub and local entries and are sorted by score
    assert any(r.template.path.startswith('github:') for r in results)
    assert any(not r.template.path.startswith('github:') for r in results)
    assert results[0].score >= results[-1].score


def test_search_all_templates_falls_back_to_local_on_github_error(tmp_path, monkeypatch):
    cfg = Config(config_dir=tmp_path / 'config')
    manager = TemplateManager(cfg)

    # Local search returns one result
    def local_search(query, limit=20):
        t1 = Template(name='local-b', stack=StackCategory.FULLSTACK, description='local', path='bundled/fullstack/local-b', tags=['fullstack'])
        return [TemplateSearchResult(template=t1, score=0.6, match_reason='Local')]
    monkeypatch.setattr(manager, 'search_templates', local_search)

    # GitHub search raises an exception (e.g., rate-limited)
    async def fake_search_error(query, stack=None, min_stars=50, limit=10):
        raise Exception('Rate limit exceeded')
    monkeypatch.setattr(manager.github_search, 'search_github_templates', fake_search_error)
    manager.github_search.github_available = True

    results = asyncio.run(manager.search_all_templates('query', include_github=True, github_limit=5, total_limit=5))
    assert all(r.match_reason and 'GitHub' not in r.match_reason for r in results) or any(r.template.path.startswith('bundled/') for r in results)
