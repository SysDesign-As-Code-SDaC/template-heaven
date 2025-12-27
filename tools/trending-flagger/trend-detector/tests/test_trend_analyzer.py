import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.trend_analyzer import TrendAnalyzer
from src.models import RepositoryMetrics
from datetime import datetime

class TestTrendAnalyzer(unittest.IsolatedAsyncioTestCase):
    @patch('psycopg2.connect')
    def setUp(self, mock_connect):
        self.config = {'postgresql': {}}
        self.trend_analyzer = TrendAnalyzer(self.config)

    @patch('src.historical_tracker.HistoricalTracker.get_historical_data', new_callable=AsyncMock)
    async def test_calculate_trend_score(self, mock_get_historical_data):
        mock_get_historical_data.return_value = [
            {'stars': 100, 'forks': 10, 'watchers': 5, 'issues': 1, 'commits': 10},
            {'stars': 110, 'forks': 12, 'watchers': 6, 'issues': 2, 'commits': 15}
        ]
        metrics = RepositoryMetrics(
            stars=110, forks=12, watchers=6, issues=2, pull_requests=1, commits=15, contributors=2,
            created_at=datetime.now(), updated_at=datetime.now(), language='Python', topics=[], size=100, license='MIT'
        )
        score = await self.trend_analyzer.calculate_trend_score(metrics, 'test_repo')
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()