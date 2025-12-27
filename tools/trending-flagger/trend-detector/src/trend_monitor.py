# src/trend_monitor.py
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

from .star_monitor import StarMonitor
from .fork_monitor import ForkMonitor
from .early_detector import EarlyDetector
from .priority_scorer import PriorityScorer
from .github_integration import GitHubIntegration
from .trend_analyzer import TrendAnalyzer
from .historical_tracker import HistoricalTracker
from .utils.database import Database
from .utils.cache import Cache
from .utils.notifications import Notifications
from .models import TrendLevel, TemplateType, RepositoryMetrics, TrendAlert

class TrendingFlagger:
    """Main trending template flagger system."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.star_monitor = StarMonitor(config)
        self.fork_monitor = ForkMonitor(config)
        self.early_detector = EarlyDetector(config)
        self.priority_scorer = PriorityScorer(config)
        self.github_integration = GitHubIntegration(config)
        self.trend_analyzer = TrendAnalyzer(config)
        self.historical_tracker = HistoricalTracker(config)

        # Initialize utilities
        self.db = Database(config)
        self.cache = Cache(config)
        self.notifications = Notifications(config)

        # Configuration
        self.star_threshold = config.get('star_threshold', 1000)
        self.fork_threshold = config.get('fork_threshold', 100)
        self.growth_rate_threshold = config.get('growth_rate_threshold', 0.1)
        self.trend_score_threshold = config.get('trend_score_threshold', 0.7)

    async def start_monitoring(self):
        """Start the trending monitoring system."""
        self.logger.info("Starting trending template flagger system")

        while True:
            try:
                # Start monitoring tasks
                tasks = [
                    self._monitor_stars(),
                    self._monitor_forks(),
                    self._detect_early_trends(),
                    self._process_review_queue(),
                    self._send_alerts()
                ]

                await asyncio.gather(*tasks)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in the main monitoring loop: {e}")

            self.logger.info("Monitoring cycle complete. Waiting for next cycle.")
            await asyncio.sleep(60)

    async def _monitor_stars(self):
        """Monitor repository star counts."""
        self.logger.info("Starting star monitoring task.")
        while True:
            try:
                # Get trending repositories
                trending_repos = await self.github_integration.get_trending_repositories()
                self.logger.info(f"Found {len(trending_repos)} trending repositories.")

                for repo in trending_repos:
                    # Check if repository meets star threshold
                    if repo['stargazers_count'] >= self.star_threshold:
                        self.logger.info(f"Repository {repo['full_name']} has {repo['stargazers_count']} stars, which is above the threshold of {self.star_threshold}.")
                        # Create trend alert
                        alert = await self._create_trend_alert(repo, 'high_stars')
                        await self._queue_for_review(alert)

                # Wait before next check
                await asyncio.sleep(self.config.get('monitor_interval', 300))  # 5 minutes

            except Exception as e:
                self.logger.error(f"Error monitoring stars: {e}")
                await asyncio.sleep(60)

    async def _monitor_forks(self):
        """Monitor repository fork activity."""
        self.logger.info("Starting fork monitoring task.")
        while True:
            try:
                # Get repositories with high fork activity
                high_fork_repos = await self.github_integration.get_high_fork_repositories()
                self.logger.info(f"Found {len(high_fork_repos)} repositories with high fork activity.")

                for repo in high_fork_repos:
                    # Check if repository meets fork threshold
                    if repo['forks_count'] >= self.fork_threshold:
                        self.logger.info(f"Repository {repo['full_name']} has {repo['forks_count']} forks, which is above the threshold of {self.fork_threshold}.")
                        # Create trend alert
                        alert = await self._create_trend_alert(repo, 'high_forks')
                        await self._queue_for_review(alert)

                # Wait before next check
                await asyncio.sleep(self.config.get('monitor_interval', 300))

            except Exception as e:
                self.logger.error(f"Error monitoring forks: {e}")
                await asyncio.sleep(60)

    async def _detect_early_trends(self):
        """Detect early trending repositories."""
        self.logger.info("Starting early trend detection task.")
        while True:
            try:
                # Get recently created repositories
                recent_repos = await self.github_integration.get_recent_repositories()
                self.logger.info(f"Found {len(recent_repos)} recently created repositories.")

                early_trending_repos = await self.early_detector.detect_early_trends(recent_repos)
                self.logger.info(f"Found {len(early_trending_repos)} early trending repositories.")

                for repo in early_trending_repos:
                    # Create trend alert
                    alert = await self._create_trend_alert(repo, 'early_trend')
                    await self._queue_for_review(alert)

                # Wait before next check
                await asyncio.sleep(self.config.get('early_detection_interval', 600))  # 10 minutes

            except Exception as e:
                self.logger.error(f"Error detecting early trends: {e}")
                await asyncio.sleep(60)

    async def _create_trend_alert(self, repo: Dict, trend_type: str) -> TrendAlert:
        """Create a trend alert for a repository."""
        self.logger.info(f"Creating trend alert for {repo['full_name']} due to {trend_type}.")
        # Get repository metrics
        metrics = await self._get_repository_metrics(repo)
        await self.historical_tracker.store_historical_data(repo['html_url'], metrics)

        # Calculate trend score
        trend_score = await self.trend_analyzer.calculate_trend_score(metrics, repo['html_url'])

        # Determine trend level
        trend_level = self._determine_trend_level(trend_score, metrics)

        # Calculate priority score
        priority_score = await self.priority_scorer.calculate_priority_score(metrics, trend_score, trend_type)

        # Determine template type
        template_type = self._determine_template_type(repo)

        # Create trend alert
        alert = TrendAlert(
            repository_url=repo['html_url'],
            repository_name=repo['full_name'],
            trend_level=trend_level,
            trend_score=trend_score,
            metrics=metrics,
            trend_reasons=[trend_type],
            priority_score=priority_score,
            created_at=datetime.now(),
            template_type=template_type,
            human_review_required=True
        )

        self.logger.info(f"Created trend alert for {repo['full_name']} with trend_score={trend_score:.2f} and priority_score={priority_score:.2f}")

        return alert

    async def _get_repository_metrics(self, repo: Dict) -> RepositoryMetrics:
        """Get comprehensive repository metrics."""
        # Get additional repository data
        repo_details = await self.github_integration.get_repository_details(repo['full_name'])
        repo_stats = await self.github_integration.get_repository_stats(repo['full_name'])

        return RepositoryMetrics(
            stars=repo['stargazers_count'],
            forks=repo['forks_count'],
            watchers=repo['watchers_count'],
            issues=repo_details.get('open_issues_count', 0),
            pull_requests=repo_stats.get('pull_requests', 0),
            commits=repo_stats.get('commits', 0),
            contributors=repo_stats.get('contributors', 0),
            created_at=datetime.fromisoformat(repo['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00')),
            language=repo.get('language', 'Unknown'),
            topics=repo.get('topics', []),
            size=repo['size'],
            license=repo.get('license', {}).get('name', 'Unknown')
        )

    def _determine_trend_level(self, trend_score: float, metrics: RepositoryMetrics) -> TrendLevel:
        """Determine trend level based on score and metrics."""
        if trend_score >= 0.9 or metrics.stars >= 10000:
            return TrendLevel.CRITICAL
        elif trend_score >= 0.7 or metrics.stars >= 5000:
            return TrendLevel.HIGH
        elif trend_score >= 0.5 or metrics.stars >= 1000:
            return TrendLevel.MEDIUM
        else:
            return TrendLevel.LOW

    def _determine_template_type(self, repo: Dict) -> TemplateType:
        """Determine template type based on repository characteristics."""
        name = repo['name'].lower()
        description = repo.get('description', '').lower()

        if any(keyword in name or keyword in description for keyword in ['template', 'boilerplate', 'starter']):
            return TemplateType.TEMPLATE
        elif any(keyword in name or keyword in description for keyword in ['framework', 'library']):
            return TemplateType.FRAMEWORK
        elif any(keyword in name or keyword in description for keyword in ['tool', 'cli', 'utility']):
            return TemplateType.TOOL
        elif any(keyword in name or keyword in description for keyword in ['tutorial', 'guide', 'example']):
            return TemplateType.TUTORIAL
        elif any(keyword in name or keyword in description for keyword in ['doc', 'documentation', 'wiki']):
            return TemplateType.DOCUMENTATION
        else:
            return TemplateType.LIBRARY

    async def _queue_for_review(self, alert: TrendAlert):
        """Queue alert for human review."""
        # Store in database
        self.db.store_alert(alert)

        # Add to Redis queue
        self.cache.add_to_review_queue(alert)

        # Send notification
        await self._send_notification(alert)

        self.logger.info(f"Queued {alert.repository_name} for human review (Priority: {alert.priority_score:.2f})")

    async def _send_notification(self, alert: TrendAlert):
        """Send notification for high-priority alerts."""
        if alert.priority_score >= 0.8 or alert.trend_level == TrendLevel.CRITICAL:
            # Send high-priority notification
            await self._send_high_priority_notification(alert)
        elif alert.priority_score >= 0.6:
            # Send medium-priority notification
            await self._send_medium_priority_notification(alert)

    async def _send_high_priority_notification(self, alert: TrendAlert):
        """Send high-priority notification."""
        message = f"🚨 HIGH PRIORITY TREND ALERT 🚨\n\n"
        message += f"Repository: {alert.repository_name}\n"
        message += f"Trend Score: {alert.trend_score:.2f}\n"
        message += f"Priority Score: {alert.priority_score:.2f}\n"
        message += f"Stars: {alert.metrics.stars}\n"
        message += f"Forks: {alert.metrics.forks}\n"
        message += f"URL: {alert.repository_url}\n"
        message += f"Template Type: {alert.template_type.value}\n"

        self.notifications.send_notification_message(message, priority='high')

    async def _send_medium_priority_notification(self, alert: TrendAlert):
        """Send medium-priority notification."""
        message = f"📈 TREND ALERT 📈\n\n"
        message += f"Repository: {alert.repository_name}\n"
        message += f"Trend Score: {alert.trend_score:.2f}\n"
        message += f"Priority Score: {alert.priority_score:.2f}\n"
        message += f"URL: {alert.repository_url}\n"

        self.notifications.send_notification_message(message, priority='medium')

    async def _process_review_queue(self):
        """Process human review queue."""
        self.logger.info("Starting review queue processing task.")
        while True:
            try:
                # Get alerts from review queue
                alerts = await self._get_review_queue_alerts()
                self.logger.info(f"Found {len(alerts)} alerts in the review queue.")

                for alert in alerts:
                    # Check if alert needs human review
                    if alert.human_review_required:
                        await self._process_human_review(alert)

                # Wait before next check
                await asyncio.sleep(self.config.get('review_interval', 1800))  # 30 minutes

            except Exception as e:
                self.logger.error(f"Error processing review queue: {e}")
                await asyncio.sleep(60)

    async def _get_review_queue_alerts(self) -> List[TrendAlert]:
        """Get alerts from review queue."""
        alerts = []

        # Get alerts from Redis queue
        for trend_level in TrendLevel:
            queue_data = self.cache.get_review_queue(trend_level)

            for data in queue_data:
                alert_data = json.loads(data.decode('utf-8'))
                alert = await self._reconstruct_alert(alert_data)
                if alert:
                    alerts.append(alert)

        # Sort by priority score
        alerts.sort(key=lambda x: x.priority_score, reverse=True)

        return alerts

    async def _reconstruct_alert(self, alert_data: Dict) -> Optional[TrendAlert]:
        """Reconstruct alert from stored data."""
        result = self.db.get_alert_by_url(alert_data['repository_url'])

        if result:
            return await self._convert_result_to_alert(result)
        return None

    async def _process_human_review(self, alert: TrendAlert):
        """Process human review for alert."""
        # This would handle human review workflow
        # Could include creating tickets, sending emails, etc.
        self.logger.info(f"Processing human review for {alert.repository_name}")

    async def _send_alerts(self):
        """Send alerts for trending repositories."""
        self.logger.info("Starting alert sending task.")
        while True:
            try:
                # Get high-priority alerts
                high_priority_alerts = await self._get_high_priority_alerts()
                self.logger.info(f"Found {len(high_priority_alerts)} high-priority alerts to send.")

                for alert in high_priority_alerts:
                    await self._send_notification(alert)

                # Wait before next check
                await asyncio.sleep(self.config.get('alert_interval', 3600))  # 1 hour

            except Exception as e:
                self.logger.error(f"Error sending alerts: {e}")
                await asyncio.sleep(60)

    async def _get_high_priority_alerts(self) -> List[TrendAlert]:
        """Get high-priority alerts."""
        results = self.db.get_high_priority_alerts()

        # Convert results to TrendAlert objects
        alerts = []
        for result in results:
            alert = await self._convert_result_to_alert(result)
            alerts.append(alert)

        return alerts

    async def _convert_result_to_alert(self, result: Tuple) -> TrendAlert:
        """Convert database result to TrendAlert object."""
        return TrendAlert(
            repository_url=result[1],
            repository_name=result[2],
            trend_level=TrendLevel(result[3]),
            trend_score=result[4],
            priority_score=result[5],
            metrics=RepositoryMetrics(
                stars=result[6],
                forks=result[7],
                watchers=result[8],
                issues=result[9],
                pull_requests=result[10],
                commits=result[11],
                contributors=result[12],
                created_at=result[13],
                updated_at=result[14],
                language=result[15],
                topics=result[16],
                size=result[17],
                license=result[18]
            ),
            trend_reasons=result[19],
            template_type=TemplateType(result[20]),
            human_review_required=result[21]
        )

if __name__ == "__main__":
    # Load configuration
    import yaml
    with open('config/github_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Start trending flagger
    flagger = TrendingFlagger(config)
    asyncio.run(flagger.start_monitoring())