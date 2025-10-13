import psycopg2
from typing import Dict, List, Tuple
import logging

class Database:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        try:
            self.connection = psycopg2.connect(**config['postgresql'])
        except psycopg2.OperationalError as e:
            self.logger.error(f"Could not connect to PostgreSQL database: {e}")
            raise

    def store_alert(self, alert):
        try:
            with self.connection.cursor() as cursor:
                query = """
                INSERT INTO trend_alerts
                (repository_url, repository_name, trend_level, trend_score, priority_score,
                 stars, forks, watchers, issues, pull_requests, commits, contributors,
                 created_at, updated_at, language, topics, size, license,
                 trend_reasons, template_type, human_review_required)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                values = (
                    alert.repository_url,
                    alert.repository_name,
                    alert.trend_level.value,
                    alert.trend_score,
                    alert.priority_score,
                    alert.metrics.stars,
                    alert.metrics.forks,
                    alert.metrics.watchers,
                    alert.metrics.issues,
                    alert.metrics.pull_requests,
                    alert.metrics.commits,
                    alert.metrics.contributors,
                    alert.metrics.created_at,
                    alert.metrics.updated_at,
                    alert.metrics.language,
                    alert.metrics.topics,
                    alert.metrics.size,
                    alert.metrics.license,
                    alert.trend_reasons,
                    alert.template_type.value,
                    alert.human_review_required
                )

                cursor.execute(query, values)
                self.connection.commit()
        except psycopg2.Error as e:
            self.logger.error(f"Error storing alert in database: {e}")

    def get_high_priority_alerts(self) -> List[Tuple]:
        try:
            with self.connection.cursor() as cursor:
                query = """
                SELECT * FROM trend_alerts
                WHERE priority_score >= 0.8 OR trend_level = 'critical'
                ORDER BY priority_score DESC, created_at DESC
                LIMIT 10
                """

                cursor.execute(query)
                return cursor.fetchall()
        except psycopg2.Error as e:
            self.logger.error(f"Error fetching high priority alerts from database: {e}")
            return []

    def get_alert_by_url(self, url: str) -> Tuple:
        try:
            with self.connection.cursor() as cursor:
                query = "SELECT * FROM trend_alerts WHERE repository_url = %s"
                cursor.execute(query, (url,))
                return cursor.fetchone()
        except psycopg2.Error as e:
            self.logger.error(f"Error fetching alert by URL from database: {e}")
            return None