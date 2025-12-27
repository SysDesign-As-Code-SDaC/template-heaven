import asyncio
import logging
from typing import Dict, List
import psycopg2
from .trend_monitor import TrendAlert, TrendLevel

class AlertSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_connection = psycopg2.connect(**config['postgresql'])

    async def send_high_priority_notification(self, alert: TrendAlert):
        """Send high-priority notification."""
        message = f"🚨 HIGH PRIORITY TREND ALERT 🚨\n\n"
        message += f"Repository: {alert.repository_name}\n"
        message += f"Trend Score: {alert.trend_score:.2f}\n"
        message += f"Priority Score: {alert.priority_score:.2f}\n"
        message += f"Stars: {alert.metrics.stars}\n"
        message += f"Forks: {alert.metrics.forks}\n"
        message += f"URL: {alert.repository_url}\n"
        message += f"Template Type: {alert.template_type.value}\n"

        # Send to notification system
        await self._send_notification_message(message, priority='high')

    async def send_medium_priority_notification(self, alert: TrendAlert):
        """Send medium-priority notification."""
        message = f"📈 TREND ALERT 📈\n\n"
        message += f"Repository: {alert.repository_name}\n"
        message += f"Trend Score: {alert.trend_score:.2f}\n"
        message += f"Priority Score: {alert.priority_score:.2f}\n"
        message += f"URL: {alert.repository_url}\n"

        # Send to notification system
        await self._send_notification_message(message, priority='medium')

    async def _send_notification_message(self, message: str, priority: str):
        """Send notification message."""
        # Implementation would depend on notification system (Slack, email, etc.)
        self.logger.info(f"Notification ({priority}): {message}")

    async def get_high_priority_alerts(self) -> List[TrendAlert]:
        """Get high-priority alerts."""
        cursor = self.db_connection.cursor()

        query = """
        SELECT * FROM trend_alerts
        WHERE priority_score >= 0.8 OR trend_level = 'critical'
        ORDER BY priority_score DESC, created_at DESC
        LIMIT 10
        """

        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()

        # Convert results to TrendAlert objects
        alerts = []
        for result in results:
            alert = await self._convert_result_to_alert(result)
            alerts.append(alert)

        return alerts

    async def _convert_result_to_alert(self, result: tuple) -> TrendAlert:
        """Convert database result to TrendAlert object."""
        return TrendAlert(
            repository_url=result[1],
            repository_name=result[2],
            trend_level=TrendLevel(result[3]),
            trend_score=result[4],
            priority_score=result[5],
            metrics=None, # In a real implementation, you'd reconstruct this
            trend_reasons=result[19],
            created_at=result[13],
            template_type=TemplateType(result[20]),
            human_review_required=result[21]
        )