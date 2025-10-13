import logging
import numpy as np
from typing import Dict, List
from .historical_tracker import HistoricalTracker

class ForkMonitor:
    """Monitor repository fork activity."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.historical_tracker = HistoricalTracker(config)

    async def monitor_fork_activity(self, repository: str) -> float:
        """
        Monitor fork activity for a repository using linear regression.
        Returns the slope of the linear regression line, which represents the trend.
        """
        historical_data = await self.historical_tracker.get_historical_data(repository)

        if len(historical_data) < 2:
            self.logger.info(f"Not enough historical data for {repository} to calculate fork growth.")
            return 0.0

        timestamps = np.array([d['timestamp'].timestamp() for d in historical_data])
        forks = np.array([d['forks'] for d in historical_data])

        # Normalize timestamps to days from the first data point
        timestamps = (timestamps - timestamps[0]) / (24 * 3600)

        try:
            # Perform linear regression
            slope, _ = np.polyfit(timestamps, forks, 1)
            self.logger.info(f"Calculated fork growth trend for {repository}: {slope}")
            return slope
        except np.linalg.LinAlgError:
            self.logger.error(f"Failed to calculate fork growth for {repository} due to a singular matrix.")
            return 0.0