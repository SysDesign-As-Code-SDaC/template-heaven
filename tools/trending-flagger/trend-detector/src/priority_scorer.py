import logging
from typing import Dict
from .trend_monitor import RepositoryMetrics
from datetime import datetime, timedelta

class PriorityScorer:
    """Calculate priority scores for repositories."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def calculate_priority_score(self, metrics: RepositoryMetrics, trend_type: str) -> float:
        """Calculate priority score for a repository."""
        # Weighted scoring based on multiple factors
        star_score = min(metrics.stars / 10000, 1.0) * 0.3
        fork_score = min(metrics.forks / 1000, 1.0) * 0.2
        growth_score = await self._calculate_growth_score(metrics) * 0.2
        activity_score = await self._calculate_activity_score(metrics) * 0.2
        quality_score = await self._calculate_quality_score(metrics) * 0.1

        total_score = star_score + fork_score + growth_score + activity_score + quality_score
        return min(total_score, 1.0)

    async def _calculate_growth_score(self, metrics: RepositoryMetrics) -> float:
        """Calculate growth score based on age and stars."""
        days_since_creation = (datetime.now() - metrics.created_at).days
        if days_since_creation > 0:
            return (metrics.stars / days_since_creation) / 10
        return 0.0

    async def _calculate_activity_score(self, metrics: RepositoryMetrics) -> float:
        """Calculate activity score based on recent updates and issues."""
        days_since_update = (datetime.now() - metrics.updated_at).days
        update_score = max(0, 1 - (days_since_update / 365))
        issue_score = min(metrics.issues / 100, 1.0)
        return (update_score * 0.7) + (issue_score * 0.3)

    async def _calculate_quality_score(self, metrics: RepositoryMetrics) -> float:
        """Calculate quality score based on license and topics."""
        license_score = 1.0 if metrics.license != 'Unknown' else 0.0
        topics_score = min(len(metrics.topics) / 5, 1.0)
        return (license_score * 0.5) + (topics_score * 0.5)