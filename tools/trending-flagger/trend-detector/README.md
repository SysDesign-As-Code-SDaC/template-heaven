# 🚩 Trending Template Flagger System

A comprehensive system for monitoring and flagging templates with high stars, fork numbers, or early trend indicators for human review and prioritization.

## 🚀 Current Status

This project is currently in the initial development phase. The core architecture is in place, but some features are not yet fully implemented. The following is a summary of the current status:

- **Implemented:**
    - Core monitoring logic for stars and forks.
    - GitHub API integration for fetching repository data.
    - Priority scoring with basic heuristics.
    - Trend analysis with a simple scoring model.
    - Review queue system using Redis.
    - PostgreSQL database integration for storing alerts.
- **Partially Implemented:**
    - Historical tracking.
- **To Be Implemented:**
    - Comprehensive test suite.
    - User interface for interacting with the system.

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

## 🛠️ Quick Start

### 1. Setup Database

Before running the application, you need to set up the PostgreSQL database. The database schema is defined in `data/schema.sql`. You can create the table using the following command:

```bash
psql -U your_user -d templateheaven -f data/schema.sql
```

### 2. Setup Trending Flagger

```bash
cd tools/trending-flagger/trend-detector
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure GitHub API

```bash
cp config/github_config.yaml.example config/github_config.yaml
# Add your GitHub API token and configure settings in config/github_config.yaml
```

### 4. Run Trending Detection

```bash
# Start trend monitoring
python src/trend_monitor.py
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
│   └── schema.sql             # Database schema
├── tests/                     # Test files
├── docs/                      # Documentation
└── examples/                  # Example implementations
```

## 🔧 Available Scripts

```bash
# Trend Monitoring
python src/trend_monitor.py          # Start trend monitoring
```

## 🚩 Trending Flagger Implementation

The main entry point for the application is `src/trend_monitor.py`. This script starts the monitoring system, which consists of several components:

- **`TrendingFlagger`**: The main class that orchestrates the monitoring process.
- **`GitHubIntegration`**: Handles all communication with the GitHub API.
- **`PriorityScorer`**: Calculates a priority score for each repository.
- **`TrendAnalyzer`**: Calculates a trend score for each repository.
- **`HistoricalTracker`**: Tracks historical data for repositories.
- **`StarMonitor`**, **`ForkMonitor`**: These modules now use a linear regression model to analyze historical data and determine the trend of star and fork counts over time. This provides a more accurate and stable measure of a repository's growth.
- **`EarlyDetector`**: This module uses the trend data from the `StarMonitor` and `ForkMonitor` to identify repositories that are showing early signs of trending.

The system uses Redis for queuing review tasks and PostgreSQL for storing trend alerts.

## 📚 Learning Resources

- [GitHub API Documentation](https://docs.github.com/en/rest)
- [aiohttp Documentation](https://docs.aiohttp.org/en/stable/)
- [psycopg2 Documentation](https://www.psycopg.org/docs/)
- [Redis-py Documentation](https://redis.io/docs/clients/python/)

## 🔗 Upstream Source

- **Repository**: [Trending Flagger](https://github.com/trending-flagger)
- **GitHub Monitor**: [GitHub Monitor](https://github.com/github-monitor)
- **Trend Analysis**: [Trend Analysis](https://github.com/trend-analysis)
- **License**: MIT