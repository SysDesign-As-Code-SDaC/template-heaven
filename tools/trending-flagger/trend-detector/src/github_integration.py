import logging
from typing import Dict, List
import aiohttp
from datetime import datetime, timedelta

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

    async def get_trending_repositories(self) -> List[Dict]:
        """Get trending repositories from GitHub."""
        since = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = f"{self.base_url}/search/repositories?q=stars:>100+created:>{since}&sort=stars&order=desc"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('items', [])

    async def get_high_fork_repositories(self) -> List[Dict]:
        """Get repositories with high fork counts."""
        since = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        url = f"{self.base_url}/search/repositories?q=forks:>100+created:>{since}&sort=forks&order=desc"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('items', [])

    async def get_recent_repositories(self) -> List[Dict]:
        """Get recently created repositories."""
        since = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        url = f"{self.base_url}/search/repositories?q=created:>{since}&sort=stars&order=desc"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('items', [])

    async def get_repository_details(self, repo_name: str) -> Dict:
        """Get detailed repository information."""
        url = f"{self.base_url}/repos/{repo_name}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()

    async def get_repository_stats(self, repo_name: str) -> Dict:
        """Get repository stats for commits and pull requests."""
        pulls_url = f"{self.base_url}/repos/{repo_name}/pulls?state=all"
        commits_url = f"{self.base_url}/repos/{repo_name}/commits?per_page=1"
        contributors_url = f"{self.base_url}/repos/{repo_name}/contributors?per_page=1"

        async with aiohttp.ClientSession() as session:
            # Get pull requests
            async with session.get(pulls_url, headers=self.headers) as response:
                response.raise_for_status()
                pulls = await response.json()
                num_pulls = len(pulls)

            # Get commits
            async with session.get(commits_url, headers=self.headers) as response:
                response.raise_for_status()
                # The number of commits is in the Link header
                link_header = response.headers.get('Link', '')
                if 'rel="last"' in link_header:
                    num_commits = int(link_header.split('page=')[-1].split('>')[0])
                else:
                    num_commits = len(await response.json())

            # Get contributors
            async with session.get(contributors_url, headers=self.headers) as response:
                response.raise_for_status()
                # The number of contributors is in the Link header
                link_header = response.headers.get('Link', '')
                if 'rel="last"' in link_header:
                    num_contributors = int(link_header.split('page=')[-1].split('>')[0])
                else:
                    num_contributors = len(await response.json())

        return {
            'pull_requests': num_pulls,
            'commits': num_commits,
            'contributors': num_contributors
        }