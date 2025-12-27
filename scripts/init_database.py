#!/usr/bin/env python3
"""
Database initialization script for Template Heaven.

This script initializes the database schema and creates all necessary tables.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from templateheaven.database.connection import init_database, db_manager
from templateheaven.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Initialize the database."""
    try:
        logger.info("🚀 Starting database initialization...")
        
        # Initialize database and create tables
        await init_database()
        
        logger.info("✅ Database initialized successfully!")
        logger.info(f"📊 Database URL: {db_manager.engine.url}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

