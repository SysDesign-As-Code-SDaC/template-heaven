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
        """Detect early trending repositories based on star and fork trends."""
        early_trending_repos = []
        star_slope_threshold = self.config.get('early_star_slope_threshold', 1.0)
        fork_slope_threshold = self.config.get('early_fork_slope_threshold', 0.5)

        self.logger.info(f"Using early trend detection thresholds: star_slope={star_slope_threshold}, fork_slope={fork_slope_threshold}")

        for repo in repositories:
            repo_full_name = repo.get('full_name')
            if not repo_full_name:
                self.logger.warning(f"Skipping repository with missing 'full_name': {repo}")
                continue

            star_trend = await self.star_monitor.monitor_star_growth(repo_full_name)
            fork_trend = await self.fork_monitor.monitor_fork_activity(repo_full_name)

            if star_trend > star_slope_threshold or fork_trend > fork_slope_threshold:
                self.logger.info(f"Detected early trend for {repo_full_name}: star_trend={star_trend}, fork_trend={fork_trend}")
                early_trending_repos.append(repo)

        return early_trending_repos