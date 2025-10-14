"""
GitHub Search Service for Template Heaven.

This module provides live template search functionality using GitHub API
to discover new templates and analyze existing repositories.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .stack_config import get_stack_config_manager
from .models import Template, StackCategory, TemplateSearchResult
from ..config.settings import Config
from ..utils.logger import get_logger

try:
    from .github_client import GitHubClient
    GITHUB_AVAILABLE = True
except ImportError:
    GitHubClient = None
    GITHUB_AVAILABLE = False

logger = get_logger(__name__)


class GitHubSearchService:
    """
    Service for searching and discovering templates on GitHub.

    Integrates GitHub API with template management to provide live
    template discovery and analysis capabilities.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the GitHub search service.

        Args:
            config: Configuration instance

        Raises:
            ImportError: If GitHub integration is requested but aiohttp is not available
        """
        self.config = config or Config()
        self.stack_config = get_stack_config_manager()
        self.github_token = self.config.get('github_token')
        self.github_available = GITHUB_AVAILABLE

    async def search_github_templates(
        self,
        query: str,
        stack: Optional[str] = None,
        min_stars: int = 50,
        limit: int = 20
    ) -> List[TemplateSearchResult]:
        """
        Search for templates on GitHub.

        Args:
            query: Search query
            stack: Optional stack filter
            min_stars: Minimum star count
            limit: Maximum results

        Returns:
            List of search results with GitHub repositories

        Raises:
            ImportError: If GitHub integration is not available
        """
        if not self.github_available:
            raise ImportError("GitHub integration not available. Install aiohttp: pip install aiohttp")

        results = []

        async with GitHubClient(self.github_token) as client:
            # Search GitHub
            if stack:
                # Use stack-specific search
                stack_config = self.stack_config.get_stack_config(stack)
                if stack_config:
                    technologies = stack_config.technologies
                    candidates = await client.find_template_candidates(
                        technologies, min_stars, limit
                    )
                else:
                    # Fallback to general search
                    repos = await client.search_repositories(
                        query, min_stars=min_stars, per_page=limit
                    )
                    candidates = []
                    for repo in repos:
                        analysis = await client.analyze_repository_for_template(repo)
                        candidates.append(analysis)
            else:
                # General search
                repos = await client.search_repositories(
                    query, min_stars=min_stars, per_page=limit
                )
                candidates = []
                for repo in repos:
                    analysis = await client.analyze_repository_for_template(repo)
                    candidates.append(analysis)

            # Convert to TemplateSearchResult objects
            for candidate in candidates:
                repo_data = candidate["repository"]

                # Create a template-like object for search results
                template = Template(
                    name=repo_data.get("name", ""),
                    stack=self._infer_stack(repo_data),
                    description=repo_data.get("description", ""),
                    path=f"github:{repo_data.get('full_name', '')}",
                    tags=repo_data.get("topics", [])[:10],  # Limit tags
                    upstream_url=repo_data.get("html_url", ""),
                    stars=repo_data.get("stargazers_count", 0),
                    forks=repo_data.get("forks_count", 0),
                    technologies=candidate.get("stack_suggestions", [])
                )

                # Create search result
                result = TemplateSearchResult(
                    template=template,
                    score=candidate["template_potential"],
                    match_reason=self._generate_match_reason(candidate),
                    metadata={
                        "github_data": repo_data,
                        "analysis": candidate,
                        "quality_score": candidate["quality_score"],
                        "risks": candidate["risks"]
                    }
                )

                results.append(result)

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(results)} GitHub template candidates for query: {query}")
        return results

    async def analyze_repository(
        self,
        repo_url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a specific GitHub repository for template potential.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Analysis results or None if analysis failed
        """
        # Extract owner/repo from URL
        if "github.com/" not in repo_url:
            logger.error(f"Invalid GitHub URL: {repo_url}")
            return None

        try:
            path_parts = repo_url.split("github.com/")[1].split("/")
            owner, repo = path_parts[0], path_parts[1]

            async with GitHubClient(self.github_token) as client:
                # Get repository details
                repo_data = await client.get_repository_details(owner, repo)
                if not repo_data:
                    return None

                # Analyze for template potential
                analysis = await client.analyze_repository_for_template(repo_data)

                # Add stack-specific validation
                stack_suggestions = analysis.get("stack_suggestions", [])
                stack_validations = {}

                for stack_name in stack_suggestions:
                    stack_config = self.stack_config.get_stack_config(stack_name)
                    if stack_config:
                        validation = self.stack_config.validate_template_for_stack(
                            {
                                'stars': repo_data.get('stargazers_count', 0),
                                'forks': repo_data.get('forks_count', 0),
                                'growth_rate': 0.0,  # Would need historical data
                                'technologies': analysis.get('stack_suggestions', [])
                            },
                            stack_name
                        )
                        stack_validations[stack_name] = validation

                analysis["stack_validations"] = stack_validations

                # Store repository metadata in cache for future use
                try:
                    repo_metadata = repo_data.copy()
                    repo_metadata.update({
                        'template_potential': analysis.get('template_potential', 0.0),
                        'quality_score': analysis.get('quality_score', 0.0),
                        'stack_suggestions': stack_suggestions,
                        'analysis_data': analysis
                    })

                    # Access cache through template manager (assuming it's available)
                    # This is a bit of a circular dependency issue, but we'll handle it gracefully
                    if hasattr(self, '_cache') and self._cache:
                        self._cache.store_repository_metadata(repo_metadata)
                except Exception as e:
                    logger.debug(f"Failed to cache repository metadata: {e}")

                logger.info(f"Analyzed repository {owner}/{repo}: potential={analysis['template_potential']:.2f}")
                return analysis

        except Exception as e:
            logger.error(f"Failed to analyze repository {repo_url}: {e}")
            return None

    async def discover_templates_for_stack(
        self,
        stack_name: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Discover potential templates for a specific stack.

        Args:
            stack_name: Stack to find templates for
            limit: Maximum number of candidates to return

        Returns:
            List of template candidates with analysis
        """
        stack_config = self.stack_config.get_stack_config(stack_name)
        if not stack_config:
            logger.error(f"Unknown stack: {stack_name}")
            return []

        async with GitHubClient(self.github_token) as client:
            candidates = await client.find_template_candidates(
                stack_config.technologies,
                min_stars=stack_config.requirements.min_stars,
                limit=limit
            )

            # Add stack-specific validation
            for candidate in candidates:
                validation = self.stack_config.validate_template_for_stack(
                    {
                        'stars': candidate['repository'].get('stargazers_count', 0),
                        'forks': candidate['repository'].get('forks_count', 0),
                        'growth_rate': 0.0,
                        'technologies': candidate.get('stack_suggestions', [])
                    },
                    stack_name
                )
                candidate["stack_validation"] = validation

            # Sort by stack validation score
            candidates.sort(key=lambda x: x.get("stack_validation", {}).get("valid", False), reverse=True)

            logger.info(f"Discovered {len(candidates)} template candidates for stack {stack_name}")
            return candidates

    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get GitHub API rate limit status.

        Returns:
            Rate limit information
        """
        async with GitHubClient(self.github_token) as client:
            return await client.get_rate_limit_status()

    def _infer_stack(self, repo_data: Dict) -> StackCategory:
        """
        Infer the most likely stack for a repository.

        Args:
            repo_data: GitHub repository data

        Returns:
            Inferred stack category
        """
        language = repo_data.get("language", "").lower()
        topics = repo_data.get("topics", [])
        description = repo_data.get("description", "").lower()

        # Language-based inference
        if language in ["javascript", "typescript"]:
            if any(topic in ["react", "vue", "angular", "svelte", "frontend"] for topic in topics):
                return StackCategory.FRONTEND
            return StackCategory.FULLSTACK

        elif language == "python":
            if any(topic in ["fastapi", "django", "flask", "api", "backend"] for topic in topics):
                return StackCategory.BACKEND
            elif any(topic in ["ml", "machine-learning", "tensorflow", "pytorch", "ai"] for topic in topics):
                return StackCategory.AI_ML
            return StackCategory.BACKEND

        elif language in ["go", "rust", "java", "csharp"]:
            return StackCategory.BACKEND

        elif language in ["docker", "shell", "yaml"]:
            return StackCategory.DEVOPS

        elif language in ["html", "css"]:
            return StackCategory.FRONTEND

        # Topic-based inference
        if any(topic in ["mobile", "react-native", "flutter", "ios", "android"] for topic in topics):
            return StackCategory.MOBILE

        if any(topic in ["web3", "blockchain", "ethereum", "solidity"] for topic in topics):
            return StackCategory.WEB3

        # Default to backend for unknown languages
        return StackCategory.BACKEND

    def _generate_match_reason(self, analysis: Dict) -> str:
        """
        Generate a human-readable match reason from analysis.

        Args:
            analysis: Repository analysis results

        Returns:
            Match reason string
        """
        reasons = analysis.get("reasons", [])
        if reasons:
            return reasons[0]  # Return the primary reason

        potential = analysis.get("template_potential", 0.0)
        if potential > 0.7:
            return "High template potential based on metrics"
        elif potential > 0.5:
            return "Good template potential"
        else:
            return "Moderate template potential"
