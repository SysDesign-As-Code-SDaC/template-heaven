import logging
from typing import Dict, List
from .historical_tracker import HistoricalTracker

class StarMonitor:
    """Monitor repository star counts."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.historical_tracker = HistoricalTracker(config)

    async def monitor_star_growth(self, repository: str) -> float:
        """Monitor star growth for a repository."""
        historical_data = await self.historical_tracker.get_historical_data(repository)

        if len(historical_data) < 2:
            return 0.0

        # Calculate growth rate
        recent_stars = historical_data[-1]['stars']
        previous_stars = historical_data[-2]['stars']

        if previous_stars == 0:
            return 1.0 if recent_stars > 0 else 0.0

        growth_rate = (recent_stars - previous_stars) / previous_stars
        return max(0.0, growth_rate)