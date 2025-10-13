import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.star_monitor import StarMonitor

class TestStarMonitor(unittest.IsolatedAsyncioTestCase):
    @patch('psycopg2.connect')
    def setUp(self, mock_connect):
        self.config = {'postgresql': {}}
        self.star_monitor = StarMonitor(self.config)

    @patch('src.historical_tracker.HistoricalTracker.get_historical_data', new_callable=AsyncMock)
    async def test_monitor_star_growth(self, mock_get_historical_data):
        mock_get_historical_data.return_value = [
            {'stars': 100},
            {'stars': 110}
        ]
        growth_rate = await self.star_monitor.monitor_star_growth('test_repo')
        self.assertAlmostEqual(growth_rate, 0.1)

    @patch('src.historical_tracker.HistoricalTracker.get_historical_data', new_callable=AsyncMock)
    async def test_monitor_star_growth_no_data(self, mock_get_historical_data):
        mock_get_historical_data.return_value = []
        growth_rate = await self.star_monitor.monitor_star_growth('test_repo')
        self.assertEqual(growth_rate, 0.0)

    @patch('src.historical_tracker.HistoricalTracker.get_historical_data', new_callable=AsyncMock)
    async def test_monitor_star_growth_single_data_point(self, mock_get_historical_data):
        mock_get_historical_data.return_value = [{'stars': 100}]
        growth_rate = await self.star_monitor.monitor_star_growth('test_repo')
        self.assertEqual(growth_rate, 0.0)

if __name__ == '__main__':
    unittest.main()