# 📈 Trend Detection Integration

This document explains how the trend detection system integrates with the multi-branch stack architecture to automatically discover and propose new templates.

## 🎯 Overview

The trend detection system monitors GitHub for trending repositories that could be valuable templates for our organization. It automatically analyzes repositories, scores them based on various metrics, and creates pull requests to add them to appropriate stack branches.

## 🏗️ Architecture

### Core Components

1. **Trend Monitor** (`src/trend_monitor.py`)
   - Main orchestrator for trend detection
   - Coordinates all monitoring components
   - Manages the detection workflow

2. **GitHub Integration** (`src/github_integration.py`)
   - Interfaces with GitHub API
   - Fetches repository data and metrics
   - Handles rate limiting and authentication

3. **Star Monitor** (`src/star_monitor.py`)
   - Tracks repository star counts
   - Uses linear regression for trend analysis
   - Identifies rapidly growing repositories

4. **Fork Monitor** (`src/fork_monitor.py`)
   - Monitors fork activity
   - Analyzes community engagement
   - Detects high-activity repositories

5. **Early Detector** (`src/early_detector.py`)
   - Identifies emerging trends
   - Analyzes growth patterns
   - Flags potential early-stage templates

6. **Priority Scorer** (`src/priority_scorer.py`)
   - Calculates priority scores
   - Combines multiple metrics
   - Ranks repositories by importance

7. **Trend Analyzer** (`src/trend_analyzer.py`)
   - AI-powered trend analysis
   - Predicts future growth
   - Identifies technology trends

8. **Historical Tracker** (`src/historical_tracker.py`)
   - Stores historical data
   - Tracks trends over time
   - Provides trend context

## 🔧 Configuration

### Global Configuration

The main configuration is in `config/github_config.yaml`:

```yaml
github:
  token: ${GITHUB_TOKEN}
  base_url: https://api.github.com
  timeout: 30
  rate_limit: 5000

monitoring:
  interval: 300  # 5 minutes
  early_detection_interval: 600  # 10 minutes
  review_interval: 1800  # 30 minutes
  alert_interval: 3600  # 1 hour

thresholds:
  star_threshold: 1000
  fork_threshold: 100
  growth_rate_threshold: 0.1
  trend_score_threshold: 0.7

database:
  host: localhost
  port: 5432
  name: templateheaven
  user: templateuser
  password: templatepass

redis:
  host: localhost
  port: 6379
  db: 0

notifications:
  enabled: true
  slack_webhook: ${SLACK_WEBHOOK}
  email_smtp: ${EMAIL_SMTP}
  email_from: ${EMAIL_FROM}
  email_to: ${EMAIL_TO}
```

### Stack-Specific Configuration

Each stack has its own `.trend-detection-config.yml`:

```yaml
stack_name: "frontend"
enabled: true

# Keywords to search for in repository names and descriptions
keywords:
  - "frontend"
  - "react"
  - "vue"
  - "svelte"
  - "template"
  - "boilerplate"
  - "starter"

# Repository characteristics to look for
characteristics:
  - has_readme: true
  - has_license: true
  - has_ci: true
  - language_specific: true

# Thresholds for trend detection
thresholds:
  stars:
    minimum: 100
    trending: 1000
    critical: 5000
  forks:
    minimum: 10
    trending: 100
    critical: 500
  growth_rate:
    minimum: 0.1
    trending: 0.5
    critical: 1.0

# Auto-sync settings
auto_sync:
  enabled: true
  high_priority_only: true
  require_approval: true

# Notification settings
notifications:
  enabled: true
  channels:
    - "slack"
    - "email"
  priority_threshold: 0.7
```

## 🔄 Workflow

### Daily Trend Detection

1. **Repository Discovery**
   - Search GitHub for trending repositories
   - Filter by stack-specific keywords
   - Apply characteristic filters

2. **Metrics Collection**
   - Fetch star counts, forks, issues, PRs
   - Calculate growth rates and trends
   - Analyze repository activity

3. **Trend Analysis**
   - Apply AI-powered trend analysis
   - Calculate trend scores
   - Predict future growth

4. **Priority Scoring**
   - Combine multiple metrics
   - Calculate priority scores
   - Rank repositories by importance

5. **Alert Generation**
   - Create trend alerts for high-priority repositories
   - Queue for human review
   - Send notifications

### Automated Template Addition

1. **High-Priority Detection**
   - Identify repositories with high priority scores
   - Verify they meet stack criteria
   - Check for existing templates

2. **Template Analysis**
   - Analyze repository structure
   - Verify it's a template/boilerplate
   - Check documentation quality

3. **Automated Sync**
   - Create PR to appropriate stack branch
   - Add template with proper attribution
   - Update stack documentation

4. **Review Process**
   - Create GitHub issue for review
   - Notify stack maintainers
   - Track approval status

## 📊 Metrics and Scoring

### Star Metrics
- **Current Stars**: Total star count
- **Star Growth Rate**: Rate of star accumulation
- **Star Velocity**: Acceleration of star growth
- **Star Trend**: Linear regression trend

### Fork Metrics
- **Current Forks**: Total fork count
- **Fork Growth Rate**: Rate of fork accumulation
- **Fork Velocity**: Acceleration of fork growth
- **Fork Trend**: Linear regression trend

### Activity Metrics
- **Recent Commits**: Commits in last 30 days
- **Issue Activity**: Open/closed issues ratio
- **PR Activity**: Open/closed PRs ratio
- **Contributor Count**: Number of contributors

### Quality Metrics
- **Documentation**: README quality score
- **License**: License presence and type
- **CI/CD**: Automated testing presence
- **Code Quality**: Linting and formatting

### Trend Score Calculation

```python
def calculate_trend_score(repo_data):
    # Star trend (40% weight)
    star_score = calculate_star_trend(repo_data['stars'])
    
    # Fork trend (20% weight)
    fork_score = calculate_fork_trend(repo_data['forks'])
    
    # Activity trend (20% weight)
    activity_score = calculate_activity_trend(repo_data['activity'])
    
    # Quality score (20% weight)
    quality_score = calculate_quality_score(repo_data['quality'])
    
    # Weighted combination
    trend_score = (
        star_score * 0.4 +
        fork_score * 0.2 +
        activity_score * 0.2 +
        quality_score * 0.2
    )
    
    return min(trend_score, 1.0)  # Cap at 1.0
```

### Priority Score Calculation

```python
def calculate_priority_score(trend_score, metrics, stack_relevance):
    # Base trend score (50% weight)
    base_score = trend_score * 0.5
    
    # Stack relevance (30% weight)
    relevance_score = stack_relevance * 0.3
    
    # Template quality (20% weight)
    quality_score = assess_template_quality(metrics) * 0.2
    
    priority_score = base_score + relevance_score + quality_score
    return min(priority_score, 1.0)  # Cap at 1.0
```

## 🚀 Usage

### Running Trend Detection

#### Manual Execution
```bash
# Run trend detection for all stacks
cd tools/trending-flagger/trend-detector
python src/trend_monitor.py

# Run for specific stack
python src/trend_monitor.py --stack frontend

# Run with custom configuration
python src/trend_monitor.py --config custom_config.yaml
```

#### Automated Execution
The system runs automatically via GitHub Actions:

```yaml
# .github/workflows/trend-detection.yml
name: Trend Detection
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:
```

### Monitoring Results

#### GitHub Issues
High-priority trends create GitHub issues:

```markdown
## 🚀 Trending Template: react-vite (frontend stack)

**Repository**: [vitejs/vite](https://github.com/vitejs/vite)
**Stack**: frontend
**Stars**: 50,000+
**Forks**: 4,000+
**Trend Score**: 0.95
**Priority Score**: 0.92

### Action Required
Please review this trending template and consider adding it to the frontend stack.

### Review Checklist
- [ ] Template follows organization standards
- [ ] Documentation is comprehensive
- [ ] Dependencies are up-to-date
- [ ] Security best practices are followed
- [ ] License is compatible
```

#### Pull Requests
Automated PRs are created for high-priority templates:

```markdown
## Add react-vite template to frontend stack

This PR adds the react-vite template to the frontend stack.

### Changes
- Added react-vite template from upstream
- Updated stack documentation
- Updated template index

### Template Details
- **Name**: react-vite
- **Stack**: frontend
- **Upstream**: https://github.com/vitejs/vite
- **Trend Score**: 0.95
- **Priority Score**: 0.92

### Testing
- [ ] Template can be successfully scaffolded
- [ ] All dependencies install correctly
- [ ] Development server starts without errors
- [ ] Tests pass (if included)
- [ ] Build process completes successfully
```

### Notifications

#### Slack Notifications
```json
{
  "text": "🚨 HIGH PRIORITY TREND ALERT 🚨",
  "attachments": [
    {
      "color": "danger",
      "fields": [
        {
          "title": "Repository",
          "value": "vitejs/vite",
          "short": true
        },
        {
          "title": "Stack",
          "value": "frontend",
          "short": true
        },
        {
          "title": "Trend Score",
          "value": "0.95",
          "short": true
        },
        {
          "title": "Priority Score",
          "value": "0.92",
          "short": true
        }
      ]
    }
  ]
}
```

#### Email Notifications
```html
<h2>🚨 High Priority Trend Alert</h2>
<p><strong>Repository:</strong> vitejs/vite</p>
<p><strong>Stack:</strong> frontend</p>
<p><strong>Trend Score:</strong> 0.95</p>
<p><strong>Priority Score:</strong> 0.92</p>
<p><strong>URL:</strong> <a href="https://github.com/vitejs/vite">https://github.com/vitejs/vite</a></p>
```

## 🛠️ Customization

### Adding New Metrics

1. **Extend Metrics Model**
   ```python
   # src/models.py
   class RepositoryMetrics:
       # Existing metrics...
       new_metric: float
   ```

2. **Update Collection Logic**
   ```python
   # src/github_integration.py
   async def get_repository_metrics(self, repo_name):
       # Existing collection...
       new_metric = await self.collect_new_metric(repo_name)
       return RepositoryMetrics(..., new_metric=new_metric)
   ```

3. **Update Scoring Algorithm**
   ```python
   # src/trend_analyzer.py
   def calculate_trend_score(self, metrics):
       # Existing calculation...
       new_score = self.calculate_new_metric_score(metrics.new_metric)
       return base_score + new_score * weight
   ```

### Custom Stack Configuration

1. **Create Stack Config**
   ```yaml
   # stacks/new-stack/.trend-detection-config.yml
   stack_name: "new-stack"
   enabled: true
   keywords:
     - "new-stack"
     - "custom-keyword"
   thresholds:
     stars:
       trending: 500  # Lower threshold for new stack
   ```

2. **Update Global Config**
   ```yaml
   # config/github_config.yaml
   stacks:
     new-stack:
       enabled: true
       custom_settings:
         special_handling: true
   ```

### Custom Notifications

1. **Add Notification Channel**
   ```python
   # src/utils/notifications.py
   class Notifications:
       def send_discord_notification(self, message, priority='medium'):
           # Discord webhook implementation
           pass
   ```

2. **Update Configuration**
   ```yaml
   # config/github_config.yaml
   notifications:
     channels:
       - "slack"
       - "email"
       - "discord"  # New channel
     discord_webhook: ${DISCORD_WEBHOOK}
   ```

## 📈 Performance Optimization

### Database Optimization
- **Indexing**: Create indexes on frequently queried fields
- **Partitioning**: Partition large tables by date
- **Caching**: Use Redis for frequently accessed data

### API Optimization
- **Rate Limiting**: Respect GitHub API rate limits
- **Caching**: Cache API responses to reduce calls
- **Batching**: Batch API requests when possible

### Processing Optimization
- **Parallel Processing**: Process multiple repositories concurrently
- **Incremental Updates**: Only process changed data
- **Background Processing**: Use queues for heavy operations

## 🚨 Troubleshooting

### Common Issues

#### API Rate Limiting
```bash
# Check rate limit status
curl -H "Authorization: token $GITHUB_TOKEN" \
     https://api.github.com/rate_limit

# Solution: Implement exponential backoff
```

#### Database Connection Issues
```bash
# Check database connectivity
psql -h localhost -U templateuser -d templateheaven -c "SELECT 1;"

# Solution: Check connection string and credentials
```

#### Redis Connection Issues
```bash
# Check Redis connectivity
redis-cli ping

# Solution: Verify Redis server is running
```

### Debugging

#### Enable Debug Logging
```python
# config/github_config.yaml
logging:
  level: DEBUG
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

#### Monitor Performance
```python
# Add performance monitoring
import time
import logging

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## 📚 Related Documentation

- [Branch Strategy](./BRANCH_STRATEGY.md) - Overall architecture
- [Stack Branch Guide](./STACK_BRANCH_GUIDE.md) - Working with stacks
- [Contributing to Stacks](./CONTRIBUTING_TO_STACKS.md) - Contribution guidelines
- [Trend Detector README](../tools/trending-flagger/trend-detector/README.md) - Tool documentation

---

**Last Updated**: 2024-01-15  
**Version**: 1.0  
**Maintainer**: Template Team
