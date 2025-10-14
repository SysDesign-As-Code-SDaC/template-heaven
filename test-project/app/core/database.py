"""
Database configuration and session management.

This module provides database connection, session management, and
initialization for the application.
"""

from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import logging

from app.core.config import get_settings
from app.core.models import Base

logger = logging.getLogger(__name__)

# Global variables for database engine and session
_engine: Optional[create_async_engine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.DEBUG,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_recycle=settings.DATABASE_POOL_RECYCLE,
            pool_pre_ping=True,
            future=True
        )
        logger.info("Database engine created")
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        logger.info("Session factory created")
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
            # Run any initialization queries
            await _run_init_queries(conn)
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def _run_init_queries(conn) -> None:
    """Run initialization queries."""
    try:
        # Create indexes that might not be created by SQLAlchemy
        init_queries = [
            # Add any custom indexes or constraints here
            "CREATE INDEX IF NOT EXISTS ix_users_email_lower ON users (LOWER(email));",
            "CREATE INDEX IF NOT EXISTS ix_posts_title_search ON posts USING gin(to_tsvector('english', title));",
            "CREATE INDEX IF NOT EXISTS ix_posts_content_search ON posts USING gin(to_tsvector('english', content));",
        ]
        
        for query in init_queries:
            try:
                await conn.execute(text(query))
            except Exception as e:
                logger.warning(f"Failed to execute init query '{query}': {e}")
        
        logger.info("Database initialization queries completed")
        
    except Exception as e:
        logger.error(f"Failed to run initialization queries: {e}")
        raise


async def close_db() -> None:
    """Close database connections."""
    global _engine, _session_factory
    
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine disposed")
    
    _session_factory = None


async def check_db_connection() -> bool:
    """Check if database connection is working."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def get_db_stats() -> dict:
    """Get database statistics."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Get table counts
            stats = {}
            
            # Count users
            result = await conn.execute(text("SELECT COUNT(*) FROM users"))
            stats['users'] = result.scalar()
            
            # Count posts
            result = await conn.execute(text("SELECT COUNT(*) FROM posts"))
            stats['posts'] = result.scalar()
            
            # Count active sessions
            result = await conn.execute(text("SELECT COUNT(*) FROM sessions WHERE is_active = true"))
            stats['active_sessions'] = result.scalar()
            
            # Count API keys
            result = await conn.execute(text("SELECT COUNT(*) FROM api_keys WHERE is_active = true"))
            stats['active_api_keys'] = result.scalar()
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}


class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self):
        self.engine = get_engine()
        self.session_factory = get_session_factory()
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def reset_database(self) -> None:
        """Reset database by dropping and recreating all tables."""
        await self.drop_tables()
        await self.create_tables()
    
    async def backup_database(self, backup_path: str) -> None:
        """Create a database backup."""
        # This would need to be implemented based on the database type
        # For PostgreSQL, you might use pg_dump
        # For SQLite, you might copy the file
        raise NotImplementedError("Database backup not implemented")
    
    async def restore_database(self, backup_path: str) -> None:
        """Restore database from backup."""
        # This would need to be implemented based on the database type
        raise NotImplementedError("Database restore not implemented")
