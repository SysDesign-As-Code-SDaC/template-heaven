import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.historical_tracker import HistoricalTracker

class TestHistoricalTracker(unittest.IsolatedAsyncioTestCase):
    @patch('psycopg2.connect')
    def setUp(self, mock_connect):
        self.config = {'postgresql': {}}
        self.historical_tracker = HistoricalTracker(self.config)

    @patch('src.utils.database.Database')
    async def test_store_historical_data(self, mock_db):
        mock_db_instance = mock_db.return_value
        metrics = MagicMock()
        self.historical_tracker.db = mock_db_instance
        await self.historical_tracker.store_historical_data('test_repo', metrics)
        mock_db_instance.store_historical_data.assert_called_once_with('test_repo', metrics)

    @patch('src.utils.database.Database')
    async def test_get_historical_data(self, mock_db):
        mock_db_instance = mock_db.return_value
        mock_db_instance.get_historical_data.return_value = [
            (1, 'test_repo', '2023-01-01', 100, 10, 5, 1, 1, 1, 1),
            (2, 'test_repo', '2023-01-02', 110, 12, 6, 2, 2, 2, 2)
        ]
        self.historical_tracker.db = mock_db_instance
        result = await self.historical_tracker.get_historical_data('test_repo')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['stars'], 100)
        self.assertEqual(result[1]['stars'], 110)

if __name__ == '__main__':
    unittest.main()