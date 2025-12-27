import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.early_detector import EarlyDetector

class TestEarlyDetector(unittest.IsolatedAsyncioTestCase):
    @patch('psycopg2.connect')
    def setUp(self, mock_connect):
        self.config = {
            'postgresql': {},
            'early_star_slope_threshold': 1.0,
            'early_fork_slope_threshold': 0.5
        }
        self.early_detector = EarlyDetector(self.config)

    @patch('src.star_monitor.StarMonitor.monitor_star_growth', new_callable=AsyncMock)
    @patch('src.fork_monitor.ForkMonitor.monitor_fork_activity', new_callable=AsyncMock)
    async def test_detect_early_trends(self, mock_fork_monitor, mock_star_monitor):
        mock_star_monitor.return_value = 1.1
        mock_fork_monitor.return_value = 0.4
        repositories = [{'full_name': 'test_repo'}]
        result = await self.early_detector.detect_early_trends(repositories)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['full_name'], 'test_repo')

    @patch('src.star_monitor.StarMonitor.monitor_star_growth', new_callable=AsyncMock)
    @patch('src.fork_monitor.ForkMonitor.monitor_fork_activity', new_callable=AsyncMock)
    async def test_detect_early_trends_no_growth(self, mock_fork_monitor, mock_star_monitor):
        mock_star_monitor.return_value = 0.1
        mock_fork_monitor.return_value = 0.1
        repositories = [{'full_name': 'test_repo'}]
        result = await self.early_detector.detect_early_trends(repositories)
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()