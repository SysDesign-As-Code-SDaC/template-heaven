import asyncio
import logging
from typing import Dict, List
import redis
import psycopg2
import json
from .trend_monitor import TrendAlert, TrendLevel

class ReviewQueue:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = redis.Redis(host=config['redis']['host'], port=config['redis']['port'])
        self.db_connection = psycopg2.connect(**config['postgresql'])

    async def get_review_queue_alerts(self) -> List[TrendAlert]:
        """Get alerts from review queue."""
        alerts = []

        # Get alerts from Redis queue
        for trend_level in TrendLevel:
            queue_key = f"review_queue:{trend_level.value}"
            queue_data = await self.redis_client.lrange(queue_key, 0, -1)

            for data in queue_data:
                alert_data = json.loads(data)
                alert = await self._reconstruct_alert(alert_data)
                alerts.append(alert)

        # Sort by priority score
        alerts.sort(key=lambda x: x.priority_score, reverse=True)

        return alerts

    async def _reconstruct_alert(self, alert_data: Dict) -> TrendAlert:
        """Reconstruct alert from stored data."""
        # This is a simplified reconstruction. A real implementation would
        # need to query the database to get the full alert details.
        return TrendAlert(
            repository_url=alert_data['repository_url'],
            repository_name=alert_data['repository_name'],
            trend_level=None,
            trend_score=alert_data['trend_score'],
            metrics=None,
            trend_reasons=[],
            priority_score=alert_data['priority_score'],
            created_at=alert_data['created_at'],
            template_type=None,
            human_review_required=True
        )

    async def process_human_review(self, alert: TrendAlert):
        """Process human review for alert."""
        # This would handle human review workflow
        # Could include creating tickets, sending emails, etc.
        self.logger.info(f"Processing human review for {alert.repository_name}")