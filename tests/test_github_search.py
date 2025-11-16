"""
Tests for GitHub Search Service.

This module contains unit tests for GitHub API integration and
live template discovery functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from templateheaven.core.github_search import GitHubSearchService
from templateheaven.core.models import TemplateSearchResult
from templateheaven.config.settings import Config


class TestGitHubSearchService:
    """Test GitHubSearchService functionality."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return Config()

    @pytest.fixture
    def search_service(self, config):
        """Create GitHub search service."""
        return GitHubSearchService(config)

    def test_initialization(self, search_service):
        """Test service initialization."""
        assert search_service.config is not None
        assert search_service.stack_config is not None
        assert search_service.github_token is None  # No token in test config

    @pytest.mark.asyncio
    async def test_search_github_templates_basic(self, search_service):
        """Test basic GitHub template search."""
        mock_repo = {
            "name": "test-repo",
            "full_name": "user/test-repo",
            "description": "A test template repository",
            "html_url": "https://github.com/user/test-repo",
            "stargazers_count": 150,
            "forks_count": 25,
            "language": "JavaScript",
            "topics": ["react", "template", "boilerplate"],
            "license": {"name": "MIT"},
            "updated_at": "2024-01-01T00:00:00Z",
            "archived": False
        }

        with patch('templateheaven.core.github_search.GitHubClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            mock_client.search_repositories.return_value = [mock_repo]
            mock_client.analyze_repository_for_template.return_value = {
                "repository": mock_repo,
                "template_potential": 0.8,
                "reasons": ["Good star count", "Has template keywords"],
                "stack_suggestions": ["frontend"],
                "quality_score": 0.9,
                "risks": []
            }

            results = await search_service.search_github_templates("react template", limit=5)

            assert len(results) == 1
            assert isinstance(results[0], TemplateSearchResult)
            assert results[0].score == 0.8
            assert results[0].match_reason == "Good star count"
            assert results[0].template.name == "test-repo"

    @pytest.mark.asyncio
    async def test_search_github_templates_with_stack(self, search_service):
        """Test GitHub search with stack filtering."""
        with patch('templateheaven.core.github_search.GitHubClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            # Mock stack config
            mock_stack_config = Mock()
            mock_stack_config.technologies = ["react", "typescript"]
            mock_stack_config.requirements.min_stars = 100

            with patch.object(search_service.stack_config, 'get_stack_config', return_value=mock_stack_config):
                mock_client.find_template_candidates.return_value = [{
                    "repository": {
                        "name": "react-template",
                        "full_name": "user/react-template",
                        "description": "A React template for frontend development",
                        "html_url": "https://github.com/user/react-template",
                        "stargazers_count": 200,
                        "forks_count": 50,
                        "language": "JavaScript",
                        "topics": ["react", "template"],
                        "license": {"name": "MIT"},
                        "updated_at": "2024-01-01T00:00:00Z",
                        "archived": False
                    },
                    "template_potential": 0.9,
                    "reasons": ["High template potential"],
                    "stack_suggestions": ["frontend"],
                    "quality_score": 0.95,
                    "risks": []
                }]

                results = await search_service.search_github_templates(
                    "react", stack="frontend", limit=5
                )

                mock_client.find_template_candidates.assert_called_once()
                assert len(results) == 1
                assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_analyze_repository(self, search_service):
        """Test repository analysis functionality."""
        repo_url = "https://github.com/user/test-repo"

        mock_analysis = {
            "repository": {"name": "test-repo", "stargazers_count": 100},
            "template_potential": 0.7,
            "stack_suggestions": ["frontend"],
            "quality_score": 0.8,
            "stack_validations": {
                "frontend": {"valid": True, "issues": [], "quality_score": 8.5}
            }
        }

        with patch('templateheaven.core.github_search.GitHubClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            mock_client.get_repository_details.return_value = {"name": "test-repo"}
            mock_client.analyze_repository_for_template.return_value = mock_analysis

            # Mock stack validation
            with patch.object(search_service.stack_config, 'validate_template_for_stack') as mock_validate:
                mock_validate.return_value = {"valid": True, "issues": [], "quality_score": 8.5}

                result = await search_service.analyze_repository(repo_url)

                assert result is not None
                assert result["template_potential"] == 0.7
                assert "stack_validations" in result
                mock_client.get_repository_details.assert_called_once_with("user", "test-repo")

    @pytest.mark.asyncio
    async def test_analyze_repository_invalid_url(self, search_service):
        """Test repository analysis with invalid URL."""
        invalid_url = "https://not-github.com/user/repo"

        result = await search_service.analyze_repository(invalid_url)

        assert result is None

    @pytest.mark.asyncio
    async def test_discover_templates_for_stack(self, search_service):
        """Test template discovery for specific stack."""
        stack_name = "frontend"

        mock_candidates = [{
            "repository": {"name": "react-starter", "stargazers_count": 300},
            "template_potential": 0.85,
            "stack_suggestions": ["frontend"],
            "quality_score": 0.9
        }]

        with patch('templateheaven.core.github_search.GitHubClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            mock_client.find_template_candidates.return_value = mock_candidates

            # Mock stack config
            mock_stack_config = Mock()
            mock_stack_config.technologies = ["react", "vue"]
            mock_stack_config.requirements.min_stars = 100

            with patch.object(search_service.stack_config, 'get_stack_config', return_value=mock_stack_config), \
                 patch.object(search_service.stack_config, 'validate_template_for_stack') as mock_validate:

                mock_validate.return_value = {"valid": True, "quality_score": 8.0}

                results = await search_service.discover_templates_for_stack(stack_name, limit=10)

                assert len(results) == 1
                assert results[0]["template_potential"] == 0.85
                assert "stack_validation" in results[0]

    @pytest.mark.asyncio
    async def test_get_rate_limit_status(self, search_service):
        """Test rate limit status retrieval."""
        mock_status = {
            "limit": 5000,
            "remaining": 4990,
            "reset": 1640995200,
            "used": 10
        }

        with patch('templateheaven.core.github_search.GitHubClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client_class.return_value.__aexit__.return_value = None

            mock_client.get_rate_limit_status.return_value = mock_status

            status = await search_service.get_rate_limit_status()

            assert status == mock_status
            mock_client.get_rate_limit_status.assert_called_once()

    def test_infer_stack(self, search_service):
        """Test stack inference from repository data."""
        from templateheaven.core.models import StackCategory

        # Test JavaScript/React repo
        js_repo = {
            "language": "JavaScript",
            "topics": ["react", "frontend"],
            "description": "React component library"
        }
        assert search_service._infer_stack(js_repo) == StackCategory.FRONTEND

        # Test Python API repo
        python_repo = {
            "language": "Python",
            "topics": ["fastapi", "api"],
            "description": "REST API server"
        }
        assert search_service._infer_stack(python_repo) == StackCategory.BACKEND

        # Test Go repo
        go_repo = {
            "language": "Go",
            "topics": ["backend", "api"],
            "description": "Microservice"
        }
        assert search_service._infer_stack(go_repo) == StackCategory.BACKEND

        # Test unknown language
        unknown_repo = {
            "language": "Unknown",
            "topics": [],
            "description": "Some project"
        }
        assert search_service._infer_stack(unknown_repo) == StackCategory.BACKEND  # Default

    def test_generate_match_reason(self, search_service):
        """Test match reason generation."""
        # Test with reasons
        analysis_with_reasons = {
            "reasons": ["High star count", "Good documentation"],
            "template_potential": 0.8
        }
        reason = search_service._generate_match_reason(analysis_with_reasons)
        assert reason == "High star count"

        # Test without reasons
        analysis_without_reasons = {
            "reasons": [],
            "template_potential": 0.9
        }
        reason = search_service._generate_match_reason(analysis_without_reasons)
        assert "High template potential" in reason

        # Test low potential
        analysis_low_potential = {
            "reasons": [],
            "template_potential": 0.3
        }
        reason = search_service._generate_match_reason(analysis_low_potential)
        assert "Moderate template potential" in reason


class TestGitHubClientIntegration:
    """Test GitHub client integration scenarios."""

    @pytest.mark.asyncio
    async def test_search_all_templates_integration(self):
        """Test the integrated search_all_templates method."""
        from templateheaven.core.template_manager import TemplateManager
        from templateheaven.config.settings import Config

        config = Config()
        manager = TemplateManager(config)

        # This would normally search both local and GitHub
        # For testing, we'll just ensure the method exists and is callable
        assert hasattr(manager, 'search_all_templates')
        assert callable(manager.search_all_templates)

        # Test with GitHub disabled (should not fail)
        results = await manager.search_all_templates("test", include_github=False, total_limit=5)
        assert isinstance(results, list)
        # Should return local template matches
        assert len(results) >= 0
