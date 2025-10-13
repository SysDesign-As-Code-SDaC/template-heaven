# src/branch_aware_monitor.py
"""
Branch-Aware Trend Monitor for Multi-Branch Stack Architecture

This module extends the original trend monitor to support the multi-branch
stack architecture, automatically detecting which stack branch a trending
template should be added to and creating appropriate pull requests.
"""

import asyncio
import logging
import yaml
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from .trend_monitor import TrendingFlagger
from .github_integration import GitHubIntegration
from .models import TrendLevel, TemplateType, RepositoryMetrics, TrendAlert
from .utils.database import Database
from .utils.cache import Cache
from .utils.notifications import Notifications


class BranchAwareTrendMonitor(TrendingFlagger):
    """
    Branch-aware trending template monitor that supports the multi-branch
    stack architecture.
    """

    def __init__(self, config_path: str):
        """
        Initialize the branch-aware trend monitor.
        
        Args:
            config_path: Path to the branch-aware configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize base class with global config
        super().__init__(self.config['global'])
        
        # Branch-specific configurations
        self.stack_configs = self.config.get('stacks', {})
        self.branch_management = self.config.get('branch_management', {})
        
        # Initialize branch-aware components
        self.branch_github = GitHubIntegration(self.config['global'])
        self.branch_db = Database(self.config['global'])
        self.branch_cache = Cache(self.config['global'])
        self.branch_notifications = Notifications(self.config['global'])
        
        # Track processed repositories to avoid duplicates
        self.processed_repos: Set[str] = set()
        
        self.logger.info(f"Initialized branch-aware trend monitor with {len(self.stack_configs)} stack configurations")

    def _load_config(self) -> Dict:
        """Load the branch-aware configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_path}: {e}")

    def _detect_stack_category(self, repo_metrics: RepositoryMetrics) -> Optional[str]:
        """
        Detect which stack category a repository belongs to based on its metadata.
        
        Args:
            repo_metrics: Repository metrics and metadata
            
        Returns:
            Stack category name or None if no match found
        """
        repo_name = repo_metrics.name.lower()
        repo_description = (repo_metrics.description or "").lower()
        repo_topics = [topic.lower() for topic in (repo_metrics.topics or [])]
        
        # Combine all text for matching
        all_text = f"{repo_name} {repo_description} {' '.join(repo_topics)}"
        
        best_match = None
        best_score = 0
        
        for stack_name, stack_config in self.stack_configs.items():
            keywords = stack_config.get('keywords', [])
            exclude_keywords = stack_config.get('exclude_keywords', [])
            
            # Check for exclude keywords first
            if any(exclude in all_text for exclude in exclude_keywords):
                continue
            
            # Calculate match score based on keywords
            score = 0
            for keyword in keywords:
                if keyword.lower() in all_text:
                    score += 1
            
            # Normalize score by number of keywords
            if keywords:
                score = score / len(keywords)
            
            if score > best_score and score > 0.1:  # Minimum threshold
                best_score = score
                best_match = stack_name
        
        return best_match

    def _get_stack_config(self, stack_name: str) -> Dict:
        """Get configuration for a specific stack."""
        return self.stack_configs.get(stack_name, {})

    def _meets_stack_criteria(self, repo_metrics: RepositoryMetrics, stack_name: str) -> bool:
        """
        Check if a repository meets the criteria for a specific stack.
        
        Args:
            repo_metrics: Repository metrics
            stack_name: Name of the stack to check against
            
        Returns:
            True if repository meets stack criteria
        """
        stack_config = self._get_stack_config(stack_name)
        
        # Check minimum stars
        min_stars = stack_config.get('min_stars', self.config['global'].get('star_threshold', 1000))
        if repo_metrics.stars < min_stars:
            return False
        
        # Check minimum forks
        min_forks = stack_config.get('min_forks', self.config['global'].get('fork_threshold', 100))
        if repo_metrics.forks < min_forks:
            return False
        
        # Check growth rate threshold
        growth_threshold = stack_config.get('growth_rate_threshold', 
                                          self.config['global'].get('growth_rate_threshold', 0.1))
        if repo_metrics.growth_rate < growth_threshold:
            return False
        
        return True

    async def _create_stack_branch_pr(self, repo_metrics: RepositoryMetrics, stack_name: str) -> bool:
        """
        Create a pull request to add a template to a specific stack branch.
        
        Args:
            repo_metrics: Repository metrics for the trending template
            stack_name: Target stack name
            
        Returns:
            True if PR was created successfully
        """
        try:
            stack_config = self._get_stack_config(stack_name)
            branch_name = stack_config.get('branch', f"stack/{stack_name}")
            
            # Create a new branch for the template addition
            template_branch = f"add-{repo_metrics.name.lower().replace('_', '-')}"
            
            # Create the PR
            pr_title = f"Add {repo_metrics.name} template to {stack_name} stack"
            pr_body = self._generate_pr_body(repo_metrics, stack_name)
            
            # Get maintainers for the stack
            maintainers = stack_config.get('maintainers', [])
            
            # Create PR using GitHub integration
            pr_url = await self.branch_github.create_pull_request(
                base_branch=branch_name,
                head_branch=template_branch,
                title=pr_title,
                body=pr_body,
                reviewers=maintainers
            )
            
            if pr_url:
                self.logger.info(f"Created PR for {repo_metrics.name} to {stack_name} stack: {pr_url}")
                
                # Send notifications
                await self._send_stack_notification(repo_metrics, stack_name, pr_url)
                return True
            else:
                self.logger.error(f"Failed to create PR for {repo_metrics.name} to {stack_name} stack")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating stack branch PR for {repo_metrics.name}: {e}")
            return False

    def _generate_pr_body(self, repo_metrics: RepositoryMetrics, stack_name: str) -> str:
        """Generate PR body for template addition."""
        return f"""
## Template Addition: {repo_metrics.name}

**Repository**: [{repo_metrics.name}]({repo_metrics.url})
**Stars**: {repo_metrics.stars:,}
**Forks**: {repo_metrics.forks:,}
**Growth Rate**: {repo_metrics.growth_rate:.2%}
**Trend Score**: {repo_metrics.trend_score:.2f}

### Description
{repo_metrics.description or 'No description available'}

### Topics
{', '.join(repo_metrics.topics or [])}

### Why This Template?
This template has been automatically detected as trending and meets the criteria for the {stack_name} stack:

- ✅ Meets minimum star threshold
- ✅ Meets minimum fork threshold  
- ✅ Exceeds growth rate threshold
- ✅ Matches stack keywords

### Next Steps
1. Review the template for quality and relevance
2. Test the template in a new project
3. Update documentation if needed
4. Merge if approved

### Automated Detection
This PR was created automatically by the branch-aware trend detection system.
"""

    async def _send_stack_notification(self, repo_metrics: RepositoryMetrics, stack_name: str, pr_url: str):
        """Send notification about new template addition to stack."""
        try:
            stack_config = self._get_stack_config(stack_name)
            maintainers = stack_config.get('maintainers', [])
            
            message = f"""
🚀 **New Trending Template Detected**

**Template**: {repo_metrics.name}
**Stack**: {stack_name}
**Stars**: {repo_metrics.stars:,}
**Growth Rate**: {repo_metrics.growth_rate:.2%}

**PR**: {pr_url}

Please review and approve if suitable for the {stack_name} stack.
"""
            
            # Send to maintainers
            for maintainer in maintainers:
                await self.branch_notifications.send_notification(
                    recipient=maintainer,
                    subject=f"New Template for {stack_name} Stack",
                    message=message
                )
                
        except Exception as e:
            self.logger.error(f"Error sending stack notification: {e}")

    async def process_trending_repository(self, repo_metrics: RepositoryMetrics) -> bool:
        """
        Process a trending repository and determine which stack it belongs to.
        
        Args:
            repo_metrics: Repository metrics
            
        Returns:
            True if repository was processed successfully
        """
        try:
            # Skip if already processed
            if repo_metrics.url in self.processed_repos:
                return True
            
            # Detect stack category
            stack_name = self._detect_stack_category(repo_metrics)
            if not stack_name:
                self.logger.info(f"No stack category detected for {repo_metrics.name}")
                return False
            
            # Check if repository meets stack criteria
            if not self._meets_stack_criteria(repo_metrics, stack_name):
                self.logger.info(f"{repo_metrics.name} does not meet criteria for {stack_name} stack")
                return False
            
            # Create PR to stack branch
            success = await self._create_stack_branch_pr(repo_metrics, stack_name)
            
            if success:
                self.processed_repos.add(repo_metrics.url)
                self.logger.info(f"Successfully processed {repo_metrics.name} for {stack_name} stack")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing trending repository {repo_metrics.name}: {e}")
            return False

    async def run_branch_aware_monitoring(self):
        """
        Run the branch-aware monitoring system.
        """
        self.logger.info("Starting branch-aware trend monitoring")
        
        try:
            # Get trending repositories
            trending_repos = await self.get_trending_repositories()
            
            self.logger.info(f"Found {len(trending_repos)} trending repositories")
            
            # Process each repository
            processed_count = 0
            for repo_metrics in trending_repos:
                if await self.process_trending_repository(repo_metrics):
                    processed_count += 1
            
            self.logger.info(f"Processed {processed_count} repositories successfully")
            
        except Exception as e:
            self.logger.error(f"Error in branch-aware monitoring: {e}")

    async def get_trending_repositories(self) -> List[RepositoryMetrics]:
        """
        Get list of trending repositories from the base monitoring system.
        
        Returns:
            List of trending repository metrics
        """
        # Use the base class method to get trending repositories
        return await self.detect_trending_templates()

    def get_stack_statistics(self) -> Dict:
        """
        Get statistics about stack configurations and processed repositories.
        
        Returns:
            Dictionary with stack statistics
        """
        return {
            'total_stacks': len(self.stack_configs),
            'processed_repositories': len(self.processed_repos),
            'stack_configs': {
                name: {
                    'branch': config.get('branch'),
                    'keywords_count': len(config.get('keywords', [])),
                    'min_stars': config.get('min_stars'),
                    'min_forks': config.get('min_forks'),
                    'auto_create_pr': config.get('auto_create_pr', False)
                }
                for name, config in self.stack_configs.items()
            }
        }


# CLI interface for branch-aware monitoring
async def main():
    """Main entry point for branch-aware trend monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Branch-Aware Trend Monitor')
    parser.add_argument('--config', required=True, help='Path to branch-aware configuration file')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuously')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run monitor
    monitor = BranchAwareTrendMonitor(args.config)
    
    if args.once:
        await monitor.run_branch_aware_monitoring()
    else:
        # Run continuously
        while True:
            try:
                await monitor.run_branch_aware_monitoring()
                await asyncio.sleep(monitor.config['global'].get('monitor_interval', 300))
            except KeyboardInterrupt:
                break
            except Exception as e:
                monitor.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying


if __name__ == "__main__":
    asyncio.run(main())
