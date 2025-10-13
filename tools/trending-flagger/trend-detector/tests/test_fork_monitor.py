import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.fork_monitor import ForkMonitor

class TestForkMonitor(unittest.IsolatedAsyncioTestCase):
    @patch('psycopg2.connect')
    def setUp(self, mock_connect):
        self.config = {'postgresql': {}}
        self.fork_monitor = ForkMonitor(self.config)

    @patch('src.historical_tracker.HistoricalTracker.get_historical_data', new_callable=AsyncMock)
    async def test_monitor_fork_activity(self, mock_get_historical_data):
        mock_get_historical_data.return_value = [
            {'forks': 50},
            {'forks': 60}
        ]
        growth_rate = await self.fork_monitor.monitor_fork_activity('test_repo')
        self.assertAlmostEqual(growth_rate, 0.2)

    @patch('src.historical_tracker.HistoricalTracker.get_historical_data', new_callable=AsyncMock)
    async def test_monitor_fork_activity_no_data(self, mock_get_historical_data):
        mock_get_historical_data.return_value = []
        growth_rate = await self.fork_monitor.monitor_fork_activity('test_repo')
        self.assertEqual(growth_rate, 0.0)

    @patch('src.historical_tracker.HistoricalTracker.get_historical_data', new_callable=AsyncMock)
    async def test_monitor_fork_activity_single_data_point(self, mock_get_historical_data):
        mock_get_historical_data.return_value = [{'forks': 50}]
        growth_rate = await self.fork_monitor.monitor_fork_activity('test_repo')
        self.assertEqual(growth_rate, 0.0)

if __name__ == '__main__':
    unittest.main()