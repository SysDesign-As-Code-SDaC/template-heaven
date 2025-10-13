import logging
from typing import Dict, List
from .historical_tracker import HistoricalTracker

class ForkMonitor:
    """Monitor repository fork activity."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.historical_tracker = HistoricalTracker(config)

    async def monitor_fork_activity(self, repository: str) -> float:
        """Monitor fork activity for a repository."""
        historical_data = await self.historical_tracker.get_historical_data(repository)

        if len(historical_data) < 2:
            return 0.0

        # Calculate growth rate
        recent_forks = historical_data[-1]['forks']
        previous_forks = historical_data[-2]['forks']

        if previous_forks == 0:
            return 1.0 if recent_forks > 0 else 0.0

        growth_rate = (recent_forks - previous_forks) / previous_forks
        return max(0.0, growth_rate)