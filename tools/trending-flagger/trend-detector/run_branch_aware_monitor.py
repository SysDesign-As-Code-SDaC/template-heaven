#!/usr/bin/env python3
"""
Branch-Aware Trend Monitor Runner

This script runs the branch-aware trend detection system for the multi-branch
stack architecture. It automatically detects trending templates and creates
pull requests to the appropriate stack branches.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import argparse
import yaml

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from branch_aware_monitor import BranchAwareTrendMonitor


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trend_monitor.log')
        ]
    )


def validate_config(config_path: str) -> bool:
    """Validate the configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['global', 'stacks']
        for section in required_sections:
            if section not in config:
                print(f"Error: Missing required section '{section}' in config file")
                return False
        
        # Check global configuration
        global_config = config['global']
        required_global = ['github', 'star_threshold', 'fork_threshold']
        for key in required_global:
            if key not in global_config:
                print(f"Error: Missing required global config '{key}'")
                return False
        
        # Check GitHub configuration
        github_config = global_config['github']
        required_github = ['api_token', 'base_repo']
        for key in required_github:
            if key not in github_config:
                print(f"Error: Missing required GitHub config '{key}'")
                return False
        
        # Check stack configurations
        stacks = config['stacks']
        if not stacks:
            print("Error: No stack configurations found")
            return False
        
        for stack_name, stack_config in stacks.items():
            required_stack = ['branch', 'keywords']
            for key in required_stack:
                if key not in stack_config:
                    print(f"Error: Missing required stack config '{key}' for stack '{stack_name}'")
                    return False
        
        print(f"Configuration validation passed. Found {len(stacks)} stack configurations.")
        return True
        
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return False


async def run_monitor(config_path: str, once: bool = False, verbose: bool = False):
    """Run the branch-aware trend monitor."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        if not validate_config(config_path):
            sys.exit(1)
        
        # Initialize monitor
        logger.info(f"Initializing branch-aware trend monitor with config: {config_path}")
        monitor = BranchAwareTrendMonitor(config_path)
        
        # Print statistics
        stats = monitor.get_stack_statistics()
        logger.info(f"Stack statistics: {stats}")
        
        if once:
            logger.info("Running trend monitoring once...")
            await monitor.run_branch_aware_monitoring()
        else:
            logger.info("Starting continuous trend monitoring...")
            monitor_interval = monitor.config['global'].get('monitor_interval', 300)
            
            while True:
                try:
                    await monitor.run_branch_aware_monitoring()
                    logger.info(f"Monitoring cycle completed. Sleeping for {monitor_interval} seconds...")
                    await asyncio.sleep(monitor_interval)
                except KeyboardInterrupt:
                    logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
        
        logger.info("Branch-aware trend monitoring completed")
        
    except Exception as e:
        logger.error(f"Fatal error in trend monitoring: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Branch-Aware Trend Monitor for Multi-Branch Stack Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once with default config
  python run_branch_aware_monitor.py --config config/branch_aware_config.yaml --once
  
  # Run continuously with verbose logging
  python run_branch_aware_monitor.py --config config/branch_aware_config.yaml --verbose
  
  # Validate configuration only
  python run_branch_aware_monitor.py --config config/branch_aware_config.yaml --validate-only
        """
    )
    
    parser.add_argument(
        '--config',
        required=True,
        help='Path to branch-aware configuration file'
    )
    
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once instead of continuously'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    
    # Validate only mode
    if args.validate_only:
        if validate_config(args.config):
            print("Configuration is valid!")
            sys.exit(0)
        else:
            print("Configuration validation failed!")
            sys.exit(1)
    
    # Run the monitor
    try:
        asyncio.run(run_monitor(args.config, args.once, args.verbose))
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
