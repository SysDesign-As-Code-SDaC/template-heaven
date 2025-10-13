import redis
from typing import Dict

class Cache:
    def __init__(self, config: Dict):
        self.client = redis.Redis(host=config['redis']['host'], port=config['redis']['port'])

    def add_to_review_queue(self, alert):
        queue_key = f"review_queue:{alert.trend_level.value}"
        alert_data = {
            'repository_url': alert.repository_url,
            'repository_name': alert.repository_name,
            'trend_score': alert.trend_score,
            'priority_score': alert.priority_score,
            'created_at': alert.created_at.isoformat()
        }

        self.client.lpush(queue_key, str(alert_data))

    def get_review_queue(self, trend_level):
        queue_key = f"review_queue:{trend_level.value}"
        return self.client.lrange(queue_key, 0, -1)