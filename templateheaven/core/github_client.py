"""
GitHub API Client for Template Heaven.

This module provides GitHub API integration for live template discovery
and repository analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

from .models import Template, StackCategory
from ..utils.logger import get_logger

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

logger = get_logger(__name__)


class GitHubClient:
    """
    GitHub API client for template discovery and repository analysis.

    Provides methods to search for repositories, analyze them for template
    potential, and extract metadata for template creation.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.

        Args:
            token: GitHub API token (optional, increases rate limits)

        Raises:
            ImportError: If aiohttp is not available
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for GitHub API integration. Install with: pip install aiohttp")
        self.token = token
        self.base_url = "https://api.github.com"
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self.requests_made = 0
        self.rate_limit_remaining = 5000  # Default for authenticated users
        self.rate_limit_reset = datetime.now() + timedelta(hours=1)

        # Headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "TemplateHeaven/1.0"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if we can make requests, False if rate limited
        """
        if self.requests_made >= self.rate_limit_remaining:
            if datetime.now() < self.rate_limit_reset:
                logger.warning("Rate limit exceeded, waiting for reset")
                return False
            else:
                # Reset counters
                self.requests_made = 0
                self.rate_limit_remaining = 5000

        return True

    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a request to the GitHub API with rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response or None if failed
        """
        if not await self._check_rate_limit():
            return None

        if not self.session:
            logger.error("No active session")
            return None

        try:
            async with self.session.get(url, params=params) as response:
                self.requests_made += 1

                # Update rate limit info from headers
                if 'X-RateLimit-Remaining' in response.headers:
                    self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                if 'X-RateLimit-Reset' in response.headers:
                    reset_timestamp = int(response.headers['X-RateLimit-Reset'])
                    self.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)

                if response.status == 200:
                    return await response.json()
                elif response.status == 403:
                    logger.warning("Rate limit exceeded")
                    return None
                elif response.status == 404:
                    logger.debug(f"Resource not found: {url}")
                    return None
                else:
                    logger.error(f"GitHub API error {response.status}: {url}")
                    return None

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    async def search_repositories(
        self,
        query: str,
        language: Optional[str] = None,
        min_stars: int = 10,
        max_stars: Optional[int] = None,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 30
    ) -> List[Dict]:
        """
        Search for repositories on GitHub.

        Args:
            query: Search query
            language: Programming language filter
            min_stars: Minimum star count
            max_stars: Maximum star count
            sort: Sort field (stars, forks, updated)
            order: Sort order (asc, desc)
            per_page: Results per page (max 100)

        Returns:
            List of repository dictionaries
        """
        # Build search query
        search_terms = [query]
        search_terms.append(f"stars:>={min_stars}")
        if max_stars:
            search_terms.append(f"stars:<={max_stars}")
        if language:
            search_terms.append(f"language:{language}")

        search_query = " ".join(search_terms)

        params = {
            "q": search_query,
            "sort": sort,
            "order": order,
            "per_page": min(per_page, 100)
        }

        url = f"{self.base_url}/search/repositories"
        response = await self._make_request(url, params)

        if response and "items" in response:
            return response["items"]
        return []

    async def get_repository_details(self, owner: str, repo: str) -> Optional[Dict]:
        """
        Get detailed information about a repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Repository details or None if not found
        """
        url = f"{self.base_url}/repos/{owner}/{repo}"
        return await self._make_request(url)

    async def analyze_repository_for_template(self, repo_data: Dict) -> Dict[str, Any]:
        """
        Analyze a repository to determine its template potential.

        Args:
            repo_data: Repository data from GitHub API

        Returns:
            Analysis results with template suitability score
        """
        analysis = {
            "repository": repo_data,
            "template_potential": 0.0,
            "reasons": [],
            "stack_suggestions": [],
            "quality_score": 0.0,
            "risks": []
        }

        # Basic metrics
        stars = repo_data.get("stargazers_count", 0)
        forks = repo_data.get("forks_count", 0)
        language = repo_data.get("language", "").lower()
        description = repo_data.get("description", "").lower()
        topics = repo_data.get("topics", [])

        # Template potential scoring
        score = 0.0

        # Stars and forks indicate popularity/usefulness
        if stars >= 1000:
            score += 0.3
            analysis["reasons"].append("High star count indicates popularity")
        elif stars >= 100:
            score += 0.2
            analysis["reasons"].append("Good star count")
        elif stars >= 10:
            score += 0.1

        if forks >= 100:
            score += 0.2
            analysis["reasons"].append("High fork count indicates reusability")

        # Check for template indicators in description
        template_keywords = ["template", "starter", "boilerplate", "scaffold", "example"]
        if any(keyword in description for keyword in template_keywords):
            score += 0.2
            analysis["reasons"].append("Description indicates template/boilerplate")

        # Check for documentation
        if repo_data.get("has_wiki") or repo_data.get("has_pages"):
            score += 0.1
            analysis["reasons"].append("Has documentation")

        # Language-based stack suggestions
        if language == "javascript" or language == "typescript":
            if any(topic in ["react", "vue", "angular", "svelte"] for topic in topics):
                analysis["stack_suggestions"].append("frontend")
            else:
                analysis["stack_suggestions"].append("fullstack")
        elif language == "python":
            if any(topic in ["fastapi", "django", "flask", "api"] for topic in topics):
                analysis["stack_suggestions"].append("backend")
            elif any(topic in ["ml", "machine-learning", "tensorflow", "pytorch"] for topic in topics):
                analysis["stack_suggestions"].append("ai-ml")
        elif language in ["go", "rust", "java"]:
            analysis["stack_suggestions"].append("backend")
        elif language in ["docker", "kubernetes", "terraform", "ansible"]:
            analysis["stack_suggestions"].append("devops")

        # Quality assessment
        quality_score = 0.0
        if repo_data.get("license"):
            quality_score += 0.2
        if len(topics) >= 3:
            quality_score += 0.2
        if repo_data.get("has_issues"):
            quality_score += 0.2
        if not repo_data.get("archived", False):
            quality_score += 0.2
        if repo_data.get("updated_at"):
            # Check if updated in last 6 months
            updated = datetime.fromisoformat(repo_data["updated_at"].replace('Z', '+00:00'))
            if (datetime.now(updated.tzinfo) - updated).days < 180:
                quality_score += 0.2

        analysis["quality_score"] = quality_score
        analysis["template_potential"] = min(score, 1.0)

        # Risk assessment
        if stars < 10:
            analysis["risks"].append("Very low star count")
        if not repo_data.get("license"):
            analysis["risks"].append("No license specified")
        if repo_data.get("archived", False):
            analysis["risks"].append("Repository is archived")

        return analysis

    async def find_template_candidates(
        self,
        technologies: List[str],
        min_stars: int = 50,
        limit: int = 50
    ) -> List[Dict]:
        """
        Find potential template repositories for given technologies.

        Args:
            technologies: List of technologies to search for
            min_stars: Minimum star count
            limit: Maximum results to return

        Returns:
            List of analyzed repository candidates
        """
        candidates = []

        for tech in technologies:
            # Search for repositories with this technology
            repos = await self.search_repositories(
                query=tech,
                min_stars=min_stars,
                per_page=min(30, limit)
            )

            for repo in repos:
                # Analyze each repository
                analysis = await self.analyze_repository_for_template(repo)

                if analysis["template_potential"] > 0.3:  # Only include decent candidates
                    candidates.append(analysis)

                if len(candidates) >= limit:
                    break

            if len(candidates) >= limit:
                break

        # Sort by template potential
        candidates.sort(key=lambda x: x["template_potential"], reverse=True)

        return candidates[:limit]

    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status.

        Returns:
            Rate limit information
        """
        url = f"{self.base_url}/rate_limit"
        response = await self._make_request(url)

        if response:
            return {
                "limit": response.get("rate", {}).get("limit", 0),
                "remaining": response.get("rate", {}).get("remaining", 0),
                "reset": response.get("rate", {}).get("reset", 0),
                "used": self.requests_made
            }

        return {"error": "Could not fetch rate limit status"}
