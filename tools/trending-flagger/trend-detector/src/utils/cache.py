import redis
from typing import Dict
import logging
import json

class Cache:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        try:
            self.client = redis.Redis(host=config['redis']['host'], port=config['redis']['port'], db=0)
            self.client.ping()
        except redis.exceptions.ConnectionError as e:
            self.logger.error(f"Could not connect to Redis: {e}")
            raise

    def add_to_review_queue(self, alert):
        try:
            queue_key = f"review_queue:{alert.trend_level.value}"
            alert_data = {
                'repository_url': alert.repository_url,
                'repository_name': alert.repository_name,
                'trend_score': alert.trend_score,
                'priority_score': alert.priority_score,
                'created_at': alert.created_at.isoformat()
            }

            self.client.lpush(queue_key, json.dumps(alert_data))
        except redis.exceptions.RedisError as e:
            self.logger.error(f"Error adding alert to review queue: {e}")

    def get_review_queue(self, trend_level):
        try:
            queue_key = f"review_queue:{trend_level.value}"
            return self.client.lrange(queue_key, 0, -1)
        except redis.exceptions.RedisError as e:
            self.logger.error(f"Error getting review queue from Redis: {e}")
            return []