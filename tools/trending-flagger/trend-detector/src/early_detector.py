import logging
from typing import Dict, List
from .star_monitor import StarMonitor
from .fork_monitor import ForkMonitor

class EarlyDetector:
    """Detect early trending repositories."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.star_monitor = StarMonitor(config)
        self.fork_monitor = ForkMonitor(config)

    async def detect_early_trends(self, repositories: List[Dict]) -> List[Dict]:
        """Detect early trending repositories."""
        early_trending_repos = []

        for repo in repositories:
            star_growth = await self.star_monitor.monitor_star_growth(repo['full_name'])
            fork_growth = await self.fork_monitor.monitor_fork_activity(repo['full_name'])

            if star_growth > self.config.get('early_star_growth_threshold', 0.5) or \
               fork_growth > self.config.get('early_fork_growth_threshold', 0.5):
                early_trending_repos.append(repo)

        return early_trending_repos