import logging
from typing import Dict
from .models import RepositoryMetrics
from .historical_tracker import HistoricalTracker

class TrendAnalyzer:
    """AI-powered trend analysis."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.historical_tracker = HistoricalTracker(config)

    async def calculate_trend_score(self, metrics: RepositoryMetrics, repo_url: str) -> float:
        """Calculate trend score using a more sophisticated model."""
        historical_data = await self.historical_tracker.get_historical_data(repo_url)

        if len(historical_data) < 2:
            return 0.0

        # Calculate growth rates
        star_growth = self._calculate_growth_rate([d['stars'] for d in historical_data])
        fork_growth = self._calculate_growth_rate([d['forks'] for d in historical_data])
        watcher_growth = self._calculate_growth_rate([d['watchers'] for d in historical_data])
        issue_growth = self._calculate_growth_rate([d['issues'] for d in historical_data])
        commit_growth = self._calculate_growth_rate([d['commits'] for d in historical_data])

        # Normalize metrics
        normalized_stars = min(metrics.stars / 10000, 1.0)
        normalized_forks = min(metrics.forks / 1000, 1.0)
        normalized_watchers = min(metrics.watchers / 1000, 1.0)

        # Calculate trend score
        trend_score = (
            (star_growth * 0.3) +
            (fork_growth * 0.2) +
            (watcher_growth * 0.1) +
            (issue_growth * 0.1) +
            (commit_growth * 0.1) +
            (normalized_stars * 0.1) +
            (normalized_forks * 0.05) +
            (normalized_watchers * 0.05)
        )

        return min(trend_score, 1.0)

    def _calculate_growth_rate(self, data: list) -> float:
        if len(data) < 2:
            return 0.0

        recent_value = data[-1]
        previous_value = data[-2]

        if previous_value == 0:
            return 1.0 if recent_value > 0 else 0.0

        growth_rate = (recent_value - previous_value) / previous_value
        return max(0.0, growth_rate)