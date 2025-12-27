import logging
from typing import Dict, List
import aiohttp
from datetime import datetime, timedelta
import re

class GitHubIntegration:
    """GitHub API integration."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_token = config['github']['api_token']
        self.base_url = 'https://api.github.com'
        self.headers = {
            'Authorization': f'token {self.api_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def _parse_link_header(self, link_header: str) -> Dict[str, str]:
        """Parse the Link header from a GitHub API response."""
        links = {}
        if link_header:
            parts = link_header.split(',')
            for part in parts:
                match = re.match(r'<(.*)>; rel="(.*)"', part.strip())
                if match:
                    links[match.group(2)] = match.group(1)
        return links

    async def get_trending_repositories(self) -> List[Dict]:
        """Get trending repositories from GitHub."""
        since = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = f"{self.base_url}/search/repositories?q=stars:>100+created:>{since}&sort=stars&order=desc"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get('items', [])
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching trending repositories: {e}")
            return []

    async def get_high_fork_repositories(self) -> List[Dict]:
        """Get repositories with high fork counts."""
        since = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = f"{self.base_url}/search/repositories?q=forks:>100+created:>{since}&sort=forks&order=desc"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get('items', [])
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching high fork repositories: {e}")
            return []

    async def get_recent_repositories(self) -> List[Dict]:
        """Get recently created repositories."""
        since = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        url = f"{self.base_url}/search/repositories?q=created:>{since}&sort=stars&order=desc"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get('items', [])
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching recent repositories: {e}")
            return []

    async def get_repository_details(self, repo_name: str) -> Dict:
        """Get detailed repository information."""
        url = f"{self.base_url}/repos/{repo_name}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching repository details for {repo_name}: {e}")
            return {}

    async def get_repository_stats(self, repo_name: str) -> Dict:
        """Get repository stats for commits and pull requests."""
        pulls_url = f"{self.base_url}/repos/{repo_name}/pulls?state=all&per_page=1"
        commits_url = f"{self.base_url}/repos/{repo_name}/commits?per_page=1"
        contributors_url = f"{self.base_url}/repos/{repo_name}/contributors?per_page=1"

        try:
            async with aiohttp.ClientSession() as session:
                # Get pull requests
                async with session.get(pulls_url, headers=self.headers) as response:
                    response.raise_for_status()
                    links = self._parse_link_header(response.headers.get('Link'))
                    if 'last' in links:
                        num_pulls = int(re.search(r'page=(\d+)', links['last']).group(1))
                    else:
                        num_pulls = len(await response.json())

                # Get commits
                async with session.get(commits_url, headers=self.headers) as response:
                    response.raise_for_status()
                    links = self._parse_link_header(response.headers.get('Link'))
                    if 'last' in links:
                        num_commits = int(re.search(r'page=(\d+)', links['last']).group(1))
                    else:
                        num_commits = len(await response.json())

                # Get contributors
                async with session.get(contributors_url, headers=self.headers) as response:
                    response.raise_for_status()
                    links = self._parse_link_header(response.headers.get('Link'))
                    if 'last' in links:
                        num_contributors = int(re.search(r'page=(\d+)', links['last']).group(1))
                    else:
                        num_contributors = len(await response.json())

            return {
                'pull_requests': num_pulls,
                'commits': num_commits,
                'contributors': num_contributors
            }
        except aiohttp.ClientError as e:
            self.logger.error(f"Error fetching repository stats for {repo_name}: {e}")
            return {}