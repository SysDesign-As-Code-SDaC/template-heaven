import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.priority_scorer import PriorityScorer
from src.models import RepositoryMetrics
from datetime import datetime

class TestPriorityScorer(unittest.IsolatedAsyncioTestCase):
    @patch('psycopg2.connect')
    def setUp(self, mock_connect):
        self.config = {'postgresql': {}}
        self.priority_scorer = PriorityScorer(self.config)

    async def test_calculate_priority_score(self):
        metrics = RepositoryMetrics(
            stars=5000, forks=500, watchers=100, issues=10, pull_requests=5, commits=100, contributors=10,
            created_at=datetime(2023, 1, 1), updated_at=datetime(2023, 10, 1), language='Python', topics=['python', 'test'], size=100, license='MIT'
        )
        trend_score = 0.8
        score = await self.priority_scorer.calculate_priority_score(metrics, trend_score, 'high_stars')
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()