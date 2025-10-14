#!/usr/bin/env python3
"""
Database Setup Script for Template Heaven Trend Detection System

This script initializes the PostgreSQL database with the required schema,
creates initial data, and sets up the necessary configurations.

Usage:
    python setup_database.py [--config CONFIG_FILE] [--reset] [--seed]

Options:
    --config CONFIG_FILE    Path to configuration file (default: config/database_config.yaml)
    --reset                 Drop and recreate all tables (WARNING: This will delete all data!)
    --seed                  Populate database with initial seed data
    --help                  Show this help message

Examples:
    python setup_database.py
    python setup_database.py --reset --seed
    python setup_database.py --config custom_config.yaml
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import psycopg2.extras

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.database import DatabaseManager
from utils.notifications import NotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseSetup:
    """Handles database setup and initialization."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database setup.
        
        Args:
            config: Database configuration dictionary
        """
        self.config = config
        self.db_manager = None
        
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.db_manager = DatabaseManager(self.config)
            self.db_manager.connect()
            logger.info("Database connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_database(self) -> bool:
        """
        Create the database if it doesn't exist.
        
        Returns:
            True if database created successfully, False otherwise
        """
        try:
            # Connect to postgres database to create our database
            postgres_config = self.config.copy()
            postgres_config['database'] = 'postgres'
            
            conn = psycopg2.connect(**postgres_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config['database'],)
            )
            
            if not cursor.fetchone():
                # Create database
                cursor.execute(f"CREATE DATABASE {self.config['database']}")
                logger.info(f"Database '{self.config['database']}' created successfully")
            else:
                logger.info(f"Database '{self.config['database']}' already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False
    
    def drop_tables(self) -> bool:
        """
        Drop all tables (WARNING: This will delete all data!).
        
        Returns:
            True if tables dropped successfully, False otherwise
        """
        try:
            if not self.db_manager:
                logger.error("Database connection not established")
                return False
            
            # Get list of all tables
            cursor = self.db_manager.connection.cursor()
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            if tables:
                # Drop all tables
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    logger.info(f"Dropped table: {table}")
                
                # Drop all functions
                cursor.execute("""
                    SELECT proname FROM pg_proc 
                    WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
                """)
                functions = [row[0] for row in cursor.fetchall()]
                
                for func in functions:
                    cursor.execute(f"DROP FUNCTION IF EXISTS {func} CASCADE")
                    logger.info(f"Dropped function: {func}")
                
                # Drop all views
                cursor.execute("""
                    SELECT viewname FROM pg_views 
                    WHERE schemaname = 'public'
                """)
                views = [row[0] for row in cursor.fetchall()]
                
                for view in views:
                    cursor.execute(f"DROP VIEW IF EXISTS {view} CASCADE")
                    logger.info(f"Dropped view: {view}")
                
                self.db_manager.connection.commit()
                logger.info("All tables, functions, and views dropped successfully")
            else:
                logger.info("No tables to drop")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            if self.db_manager and self.db_manager.connection:
                self.db_manager.connection.rollback()
            return False
    
    def create_schema(self) -> bool:
        """
        Create the database schema from schema.sql.
        
        Returns:
            True if schema created successfully, False otherwise
        """
        try:
            if not self.db_manager:
                logger.error("Database connection not established")
                return False
            
            # Read schema file
            schema_file = Path(__file__).parent / "data" / "schema.sql"
            if not schema_file.exists():
                logger.error(f"Schema file not found: {schema_file}")
                return False
            
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Execute schema
            cursor = self.db_manager.connection.cursor()
            cursor.execute(schema_sql)
            self.db_manager.connection.commit()
            cursor.close()
            
            logger.info("Database schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            if self.db_manager and self.db_manager.connection:
                self.db_manager.connection.rollback()
            return False
    
    def seed_initial_data(self) -> bool:
        """
        Populate database with initial seed data.
        
        Returns:
            True if seed data created successfully, False otherwise
        """
        try:
            if not self.db_manager:
                logger.error("Database connection not established")
                return False
            
            cursor = self.db_manager.connection.cursor()
            
            # Insert initial stack configurations
            stack_configs = [
                {
                    'stack_name': 'frontend',
                    'enabled': True,
                    'keywords': ['react', 'vue', 'angular', 'svelte', 'frontend', 'ui', 'component'],
                    'characteristics': {
                        'package_managers': ['npm', 'yarn', 'pnpm'],
                        'build_tools': ['vite', 'webpack', 'rollup', 'parcel'],
                        'frameworks': ['react', 'vue', 'angular', 'svelte']
                    },
                    'thresholds': {
                        'min_stars': 100,
                        'min_forks': 10,
                        'min_contributors': 3,
                        'trend_threshold': 0.6
                    },
                    'auto_sync_enabled': False,
                    'require_approval': True,
                    'notification_channels': ['slack', 'email'],
                    'priority_threshold': 0.7
                },
                {
                    'stack_name': 'backend',
                    'enabled': True,
                    'keywords': ['api', 'server', 'backend', 'express', 'fastapi', 'django', 'rails'],
                    'characteristics': {
                        'languages': ['javascript', 'python', 'java', 'go', 'rust', 'csharp'],
                        'frameworks': ['express', 'fastapi', 'django', 'rails', 'spring', 'gin'],
                        'databases': ['postgresql', 'mysql', 'mongodb', 'redis']
                    },
                    'thresholds': {
                        'min_stars': 200,
                        'min_forks': 20,
                        'min_contributors': 5,
                        'trend_threshold': 0.7
                    },
                    'auto_sync_enabled': False,
                    'require_approval': True,
                    'notification_channels': ['slack', 'email'],
                    'priority_threshold': 0.8
                },
                {
                    'stack_name': 'ai-ml',
                    'enabled': True,
                    'keywords': ['ai', 'ml', 'machine-learning', 'deep-learning', 'pytorch', 'tensorflow'],
                    'characteristics': {
                        'languages': ['python', 'r', 'julia'],
                        'frameworks': ['pytorch', 'tensorflow', 'scikit-learn', 'keras'],
                        'domains': ['nlp', 'computer-vision', 'reinforcement-learning']
                    },
                    'thresholds': {
                        'min_stars': 500,
                        'min_forks': 50,
                        'min_contributors': 10,
                        'trend_threshold': 0.8
                    },
                    'auto_sync_enabled': False,
                    'require_approval': True,
                    'notification_channels': ['slack', 'email'],
                    'priority_threshold': 0.9
                },
                {
                    'stack_name': 'devops',
                    'enabled': True,
                    'keywords': ['devops', 'ci', 'cd', 'docker', 'kubernetes', 'terraform', 'ansible'],
                    'characteristics': {
                        'tools': ['docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'github-actions'],
                        'cloud_providers': ['aws', 'azure', 'gcp'],
                        'monitoring': ['prometheus', 'grafana', 'datadog']
                    },
                    'thresholds': {
                        'min_stars': 300,
                        'min_forks': 30,
                        'min_contributors': 8,
                        'trend_threshold': 0.7
                    },
                    'auto_sync_enabled': False,
                    'require_approval': True,
                    'notification_channels': ['slack', 'email'],
                    'priority_threshold': 0.8
                }
            ]
            
            for config in stack_configs:
                cursor.execute("""
                    INSERT INTO stack_configurations (
                        stack_name, enabled, keywords, characteristics, thresholds,
                        auto_sync_enabled, require_approval, notification_channels, priority_threshold
                    ) VALUES (
                        %(stack_name)s, %(enabled)s, %(keywords)s, %(characteristics)s, %(thresholds)s,
                        %(auto_sync_enabled)s, %(require_approval)s, %(notification_channels)s, %(priority_threshold)s
                    ) ON CONFLICT (stack_name) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        keywords = EXCLUDED.keywords,
                        characteristics = EXCLUDED.characteristics,
                        thresholds = EXCLUDED.thresholds,
                        auto_sync_enabled = EXCLUDED.auto_sync_enabled,
                        require_approval = EXCLUDED.require_approval,
                        notification_channels = EXCLUDED.notification_channels,
                        priority_threshold = EXCLUDED.priority_threshold,
                        updated_at = CURRENT_TIMESTAMP
                """, config)
            
            self.db_manager.connection.commit()
            cursor.close()
            
            logger.info("Initial seed data created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create seed data: {e}")
            if self.db_manager and self.db_manager.connection:
                self.db_manager.connection.rollback()
            return False
    
    def verify_setup(self) -> bool:
        """
        Verify that the database setup is correct.
        
        Returns:
            True if setup is correct, False otherwise
        """
        try:
            if not self.db_manager:
                logger.error("Database connection not established")
                return False
            
            cursor = self.db_manager.connection.cursor()
            
            # Check if all required tables exist
            required_tables = [
                'repositories', 'repository_metrics', 'trend_alerts',
                'stack_configurations', 'template_candidates', 'sync_history',
                'notifications'
            ]
            
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = set(required_tables) - set(existing_tables)
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                return False
            
            # Check if views exist
            required_views = ['trending_repositories', 'high_priority_alerts', 'approved_templates']
            cursor.execute("""
                SELECT viewname FROM pg_views 
                WHERE schemaname = 'public'
            """)
            existing_views = [row[0] for row in cursor.fetchall()]
            
            missing_views = set(required_views) - set(existing_views)
            if missing_views:
                logger.error(f"Missing views: {missing_views}")
                return False
            
            # Check if functions exist
            cursor.execute("""
                SELECT proname FROM pg_proc 
                WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """)
            existing_functions = [row[0] for row in cursor.fetchall()]
            
            if 'update_repository_metrics' not in existing_functions:
                logger.error("Missing function: update_repository_metrics")
                return False
            
            # Check if stack configurations exist
            cursor.execute("SELECT COUNT(*) FROM stack_configurations")
            config_count = cursor.fetchone()[0]
            
            if config_count == 0:
                logger.warning("No stack configurations found")
            else:
                logger.info(f"Found {config_count} stack configurations")
            
            cursor.close()
            logger.info("Database setup verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify setup: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.db_manager:
            self.db_manager.close()


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables if they exist
        if 'database' in config:
            db_config = config['database']
            db_config['host'] = os.getenv('DB_HOST', db_config.get('host', 'localhost'))
            db_config['port'] = int(os.getenv('DB_PORT', db_config.get('port', 5432)))
            db_config['database'] = os.getenv('DB_NAME', db_config.get('database', 'template_heaven'))
            db_config['user'] = os.getenv('DB_USER', db_config.get('user', 'postgres'))
            db_config['password'] = os.getenv('DB_PASSWORD', db_config.get('password', ''))
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup database for Template Heaven Trend Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        default='config/database_config.yaml',
        help='Path to configuration file (default: config/database_config.yaml)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Drop and recreate all tables (WARNING: This will delete all data!)'
    )
    
    parser.add_argument(
        '--seed',
        action='store_true',
        help='Populate database with initial seed data'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify the current database setup'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize database setup
    db_setup = DatabaseSetup(config['database'])
    
    try:
        # Create database if it doesn't exist
        if not db_setup.create_database():
            logger.error("Failed to create database")
            sys.exit(1)
        
        # Connect to database
        if not db_setup.connect():
            logger.error("Failed to connect to database")
            sys.exit(1)
        
        # Verify only mode
        if args.verify_only:
            if db_setup.verify_setup():
                logger.info("Database setup verification passed")
                sys.exit(0)
            else:
                logger.error("Database setup verification failed")
                sys.exit(1)
        
        # Reset database if requested
        if args.reset:
            logger.warning("Resetting database - all data will be lost!")
            if not db_setup.drop_tables():
                logger.error("Failed to drop tables")
                sys.exit(1)
        
        # Create schema
        if not db_setup.create_schema():
            logger.error("Failed to create schema")
            sys.exit(1)
        
        # Seed initial data if requested
        if args.seed:
            if not db_setup.seed_initial_data():
                logger.error("Failed to create seed data")
                sys.exit(1)
        
        # Verify setup
        if not db_setup.verify_setup():
            logger.error("Database setup verification failed")
            sys.exit(1)
        
        logger.info("Database setup completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        sys.exit(1)
    finally:
        db_setup.close()


if __name__ == '__main__':
    main()
