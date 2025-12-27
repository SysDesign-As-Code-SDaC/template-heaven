import logging
from typing import Dict, List
from .utils.database import Database

class HistoricalTracker:
    """Track historical repository data."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db = Database(config)

    async def store_historical_data(self, repository_url: str, metrics):
        """Store historical data for a repository."""
        self.db.store_historical_data(repository_url, metrics)

    async def get_historical_data(self, repository_url: str) -> List[Dict]:
        """Get historical data for a repository."""
        historical_data = []
        records = self.db.get_historical_data(repository_url)

        for record in records:
            historical_data.append({
                'timestamp': record[2],
                'stars': record[3],
                'forks': record[4],
                'watchers': record[5],
                'issues': record[6],
                'pull_requests': record[7],
                'commits': record[8],
                'contributors': record[9]
            })

        return historical_data