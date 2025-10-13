# 🚩 Trending Template Flagger System

A comprehensive system for monitoring and flagging templates with high stars, fork numbers, or early trend indicators for human review and prioritization.

## 🚀 Features

- **High Stars Monitor** - Tracks repositories with significant star counts
- **Fork Activity Monitor** - Monitors repositories with high fork activity
- **Early Trend Detection** - Identifies emerging technologies and rapid growth
- **Human Review Queue** - Flags templates for human review and prioritization
- **Trend Analysis** - AI-powered trend prediction and analysis
- **Alert System** - Real-time notifications for trending templates
- **Priority Scoring** - Intelligent scoring system for template prioritization
- **GitHub Integration** - Direct integration with GitHub API
- **Historical Tracking** - Tracks trends over time
- **Custom Thresholds** - Configurable thresholds for different metrics

## 📋 Prerequisites

- Python 3.9+
- GitHub API Token
- Redis (for caching)
- PostgreSQL (for data storage)
- 8GB+ RAM (for trend analysis)

## 🛠️ Quick Start

### 1. Setup Trending Flagger

```bash
cd tools/trending-flagger/trend-detector
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure GitHub API

```bash
cp config/github_config.yaml.example config/github_config.yaml
# Add your GitHub API token and configure settings
```

### 3. Run Trending Detection

```bash
# Start trend monitoring
python src/trend_monitor.py --config config/github_config.yaml

# Start human review queue
python src/review_queue.py --config config/review_config.yaml

# Start alert system
python src/alert_system.py --config config/alert_config.yaml
```

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── trend_monitor.py       # Main trend monitoring system
│   ├── star_monitor.py        # Star count monitoring
│   ├── fork_monitor.py        # Fork activity monitoring
│   ├── early_detector.py      # Early trend detection
│   ├── review_queue.py        # Human review queue management
│   ├── alert_system.py        # Alert and notification system
│   ├── priority_scorer.py     # Priority scoring algorithm
│   ├── github_integration.py  # GitHub API integration
│   ├── trend_analyzer.py      # AI-powered trend analysis
│   ├── historical_tracker.py  # Historical trend tracking
│   └── utils/                 # Utility functions
│       ├── database.py        # Database operations
│       ├── cache.py           # Redis caching
│       └── notifications.py   # Notification utilities
├── config/                    # Configuration files
│   ├── github_config.yaml
│   ├── review_config.yaml
│   └── alert_config.yaml
├── data/                      # Data storage
├── tests/                     # Test files
├── docs/                      # Documentation
└── examples/                  # Example implementations
```

## 🔧 Available Scripts

```bash
# Trend Monitoring
python src/trend_monitor.py          # Start trend monitoring
python src/star_monitor.py           # Monitor star counts
python src/fork_monitor.py           # Monitor fork activity
python src/early_detector.py         # Early trend detection

# Review and Alerts
python src/review_queue.py           # Human review queue
python src/alert_system.py           # Alert system
python src/priority_scorer.py        # Priority scoring

# Analysis and Integration
python src/trend_analyzer.py         # Trend analysis
python src/github_integration.py     # GitHub API integration
python src/historical_tracker.py     # Historical tracking
```

## 🚩 Trending Flagger Implementation

### Main Trend Monitor

```python
# src/trend_monitor.py
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
import redis
import psycopg2
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class TrendLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TemplateType(Enum):
    FRAMEWORK = "framework"
    LIBRARY = "library"
    TOOL = "tool"
    TEMPLATE = "template"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"

@dataclass
class RepositoryMetrics:
    """Repository metrics for trend analysis."""
    stars: int
    forks: int
    watchers: int
    issues: int
    pull_requests: int
    commits: int
    contributors: int
    created_at: datetime
    updated_at: datetime
    language: str
    topics: List[str]
    size: int
    license: str

@dataclass
class TrendAlert:
    """Trend alert for human review."""
    repository_url: str
    repository_name: str
    trend_level: TrendLevel
    trend_score: float
    metrics: RepositoryMetrics
    trend_reasons: List[str]
    priority_score: float
    created_at: datetime
    template_type: TemplateType
    human_review_required: bool

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
        
        # Initialize databases
        self.redis_client = redis.Redis(host=config['redis']['host'], port=config['redis']['port'])
        self.db_connection = psycopg2.connect(**config['postgresql'])
        
        # Configuration
        self.star_threshold = config.get('star_threshold', 1000)
        self.fork_threshold = config.get('fork_threshold', 100)
        self.growth_rate_threshold = config.get('growth_rate_threshold', 0.1)
        self.trend_score_threshold = config.get('trend_score_threshold', 0.7)
    
    async def start_monitoring(self):
        """Start the trending monitoring system."""
        self.logger.info("Starting trending template flagger system")
        
        # Start monitoring tasks
        tasks = [
            self._monitor_stars(),
            self._monitor_forks(),
            self._detect_early_trends(),
            self._process_review_queue(),
            self._send_alerts()
        ]
        
        await asyncio.gather(*tasks)
    
    async def _monitor_stars(self):
        """Monitor repository star counts."""
        while True:
            try:
                # Get trending repositories
                trending_repos = await self.github_integration.get_trending_repositories()
                
                for repo in trending_repos:
                    # Check if repository meets star threshold
                    if repo['stargazers_count'] >= self.star_threshold:
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
        while True:
            try:
                # Get repositories with high fork activity
                high_fork_repos = await self.github_integration.get_high_fork_repositories()
                
                for repo in high_fork_repos:
                    # Check if repository meets fork threshold
                    if repo['forks_count'] >= self.fork_threshold:
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
        while True:
            try:
                # Get recently created repositories
                recent_repos = await self.github_integration.get_recent_repositories()
                
                for repo in recent_repos:
                    # Calculate growth rate
                    growth_rate = await self._calculate_growth_rate(repo)
                    
                    if growth_rate >= self.growth_rate_threshold:
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
        # Get repository metrics
        metrics = await self._get_repository_metrics(repo)
        
        # Calculate trend score
        trend_score = await self.trend_analyzer.calculate_trend_score(metrics)
        
        # Determine trend level
        trend_level = self._determine_trend_level(trend_score, metrics)
        
        # Calculate priority score
        priority_score = await self.priority_scorer.calculate_priority_score(metrics, trend_type)
        
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
        
        return alert
    
    async def _get_repository_metrics(self, repo: Dict) -> RepositoryMetrics:
        """Get comprehensive repository metrics."""
        # Get additional repository data
        repo_details = await self.github_integration.get_repository_details(repo['full_name'])
        
        return RepositoryMetrics(
            stars=repo['stargazers_count'],
            forks=repo['forks_count'],
            watchers=repo['watchers_count'],
            issues=repo_details.get('open_issues_count', 0),
            pull_requests=repo_details.get('open_pull_requests_count', 0),
            commits=repo_details.get('commits_count', 0),
            contributors=repo_details.get('contributors_count', 0),
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
    
    async def _calculate_growth_rate(self, repo: Dict) -> float:
        """Calculate growth rate for a repository."""
        # Get historical data
        historical_data = await self.historical_tracker.get_historical_data(repo['full_name'])
        
        if len(historical_data) < 2:
            return 0.0
        
        # Calculate growth rate
        recent_stars = historical_data[-1]['stars']
        previous_stars = historical_data[-2]['stars']
        
        if previous_stars == 0:
            return 1.0 if recent_stars > 0 else 0.0
        
        growth_rate = (recent_stars - previous_stars) / previous_stars
        return max(0.0, growth_rate)
    
    async def _queue_for_review(self, alert: TrendAlert):
        """Queue alert for human review."""
        # Store in database
        await self._store_alert(alert)
        
        # Add to Redis queue
        await self._add_to_review_queue(alert)
        
        # Send notification
        await self._send_notification(alert)
        
        self.logger.info(f"Queued {alert.repository_name} for human review (Priority: {alert.priority_score:.2f})")
    
    async def _store_alert(self, alert: TrendAlert):
        """Store alert in database."""
        cursor = self.db_connection.cursor()
        
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
        self.db_connection.commit()
        cursor.close()
    
    async def _add_to_review_queue(self, alert: TrendAlert):
        """Add alert to Redis review queue."""
        queue_key = f"review_queue:{alert.trend_level.value}"
        alert_data = {
            'repository_url': alert.repository_url,
            'repository_name': alert.repository_name,
            'trend_score': alert.trend_score,
            'priority_score': alert.priority_score,
            'created_at': alert.created_at.isoformat()
        }
        
        await self.redis_client.lpush(queue_key, str(alert_data))
    
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
        
        # Send to notification system
        await self._send_notification_message(message, priority='high')
    
    async def _send_medium_priority_notification(self, alert: TrendAlert):
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
    
    async def _process_review_queue(self):
        """Process human review queue."""
        while True:
            try:
                # Get alerts from review queue
                alerts = await self._get_review_queue_alerts()
                
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
            queue_key = f"review_queue:{trend_level.value}"
            queue_data = await self.redis_client.lrange(queue_key, 0, -1)
            
            for data in queue_data:
                alert_data = eval(data)  # In production, use proper JSON parsing
                alert = await self._reconstruct_alert(alert_data)
                alerts.append(alert)
        
        # Sort by priority score
        alerts.sort(key=lambda x: x.priority_score, reverse=True)
        
        return alerts
    
    async def _reconstruct_alert(self, alert_data: Dict) -> TrendAlert:
        """Reconstruct alert from stored data."""
        # This would reconstruct the alert from database/Redis data
        # Implementation depends on data storage format
        pass
    
    async def _process_human_review(self, alert: TrendAlert):
        """Process human review for alert."""
        # This would handle human review workflow
        # Could include creating tickets, sending emails, etc.
        self.logger.info(f"Processing human review for {alert.repository_name}")
    
    async def _send_alerts(self):
        """Send alerts for trending repositories."""
        while True:
            try:
                # Get high-priority alerts
                high_priority_alerts = await self._get_high_priority_alerts()
                
                for alert in high_priority_alerts:
                    await self._send_alert(alert)
                
                # Wait before next check
                await asyncio.sleep(self.config.get('alert_interval', 3600))  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Error sending alerts: {e}")
                await asyncio.sleep(60)
    
    async def _get_high_priority_alerts(self) -> List[TrendAlert]:
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
    
    async def _convert_result_to_alert(self, result: Tuple) -> TrendAlert:
        """Convert database result to TrendAlert object."""
        # Implementation would convert database row to TrendAlert
        pass
    
    async def _send_alert(self, alert: TrendAlert):
        """Send alert for trending repository."""
        # Implementation would send alert via configured channels
        self.logger.info(f"Sending alert for {alert.repository_name}")

class StarMonitor:
    """Monitor repository star counts."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def monitor_star_growth(self, repository: str) -> float:
        """Monitor star growth for a repository."""
        # Implementation would track star growth over time
        pass

class ForkMonitor:
    """Monitor repository fork activity."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def monitor_fork_activity(self, repository: str) -> float:
        """Monitor fork activity for a repository."""
        # Implementation would track fork activity over time
        pass

class EarlyDetector:
    """Detect early trending repositories."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def detect_early_trends(self, repositories: List[Dict]) -> List[Dict]:
        """Detect early trending repositories."""
        # Implementation would use ML models to detect early trends
        pass

class PriorityScorer:
    """Calculate priority scores for repositories."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def calculate_priority_score(self, metrics: RepositoryMetrics, trend_type: str) -> float:
        """Calculate priority score for a repository."""
        # Weighted scoring based on multiple factors
        star_score = min(metrics.stars / 10000, 1.0) * 0.3
        fork_score = min(metrics.forks / 1000, 1.0) * 0.2
        growth_score = await self._calculate_growth_score(metrics) * 0.2
        activity_score = await self._calculate_activity_score(metrics) * 0.2
        quality_score = await self._calculate_quality_score(metrics) * 0.1
        
        total_score = star_score + fork_score + growth_score + activity_score + quality_score
        return min(total_score, 1.0)
    
    async def _calculate_growth_score(self, metrics: RepositoryMetrics) -> float:
        """Calculate growth score."""
        # Implementation would calculate growth rate
        return 0.5
    
    async def _calculate_activity_score(self, metrics: RepositoryMetrics) -> float:
        """Calculate activity score."""
        # Implementation would calculate activity level
        return 0.5
    
    async def _calculate_quality_score(self, metrics: RepositoryMetrics) -> float:
        """Calculate quality score."""
        # Implementation would calculate quality metrics
        return 0.5

class GitHubIntegration:
    """GitHub API integration."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_token = config['github']['api_token']
        self.base_url = 'https://api.github.com'
    
    async def get_trending_repositories(self) -> List[Dict]:
        """Get trending repositories from GitHub."""
        # Implementation would use GitHub API to get trending repos
        pass
    
    async def get_high_fork_repositories(self) -> List[Dict]:
        """Get repositories with high fork counts."""
        # Implementation would use GitHub API
        pass
    
    async def get_recent_repositories(self) -> List[Dict]:
        """Get recently created repositories."""
        # Implementation would use GitHub API
        pass
    
    async def get_repository_details(self, repo_name: str) -> Dict:
        """Get detailed repository information."""
        # Implementation would use GitHub API
        pass

class TrendAnalyzer:
    """AI-powered trend analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def calculate_trend_score(self, metrics: RepositoryMetrics) -> float:
        """Calculate trend score using AI."""
        # Implementation would use ML models to calculate trend score
        return 0.5

class HistoricalTracker:
    """Track historical repository data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def get_historical_data(self, repository: str) -> List[Dict]:
        """Get historical data for a repository."""
        # Implementation would retrieve historical data
        pass

if __name__ == "__main__":
    # Load configuration
    import yaml
    with open('config/github_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Start trending flagger
    flagger = TrendingFlagger(config)
    asyncio.run(flagger.start_monitoring())
```

## 📚 Learning Resources

- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Trend Analysis](https://trend-analysis.com/)
- [Repository Monitoring](https://repo-monitoring.com/)

## 🔗 Upstream Source

- **Repository**: [Trending Flagger](https://github.com/trending-flagger)
- **GitHub Monitor**: [GitHub Monitor](https://github.com/github-monitor)
- **Trend Analysis**: [Trend Analysis](https://github.com/trend-analysis)
- **License**: MIT
