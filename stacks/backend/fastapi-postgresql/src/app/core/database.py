"""
Database Connection and Session Management

Provides async SQLAlchemy database connectivity with connection pooling,
migrations support, and health monitoring.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import QueuePool

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global database engine
_engine = None
_async_session_maker = None


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.SQLALCHEMY_DATABASE_URI,
            echo=settings.DEBUG,
            future=True,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300,  # Recycle connections every 5 minutes
        )
        logger.info("Database engine created")
    return _engine


def get_async_session_maker():
    """Get async session maker."""
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False
        )
    return _async_session_maker


async def create_db_and_tables():
    """Create database and tables if they don't exist."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def drop_db_and_tables():
    """Drop all tables (use with caution - for testing only)."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Database session dependency.

    Yields:
        AsyncSession: Database session for the request

    Automatically handles commit/rollback and cleanup.
    """
    session_maker = get_async_session_maker()
    session = session_maker()

    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def get_db_session() -> AsyncSession:
    """Get database session (for use in services/utilities)."""
    session_maker = get_async_session_maker()
    return session_maker()


async def check_database_health() -> dict:
    """
    Check database connectivity and health.

    Returns:
        dict: Health status information
    """
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Simple query to test connectivity
            result = await conn.execute("SELECT 1 as test")
            row = result.fetchone()

            return {
                "status": "healthy",
                "connection": True,
                "test_query": row[0] if row else None,
                "database_url": settings.SQLALCHEMY_DATABASE_URI.replace(
                    settings.POSTGRES_PASSWORD, "***" if settings.POSTGRES_PASSWORD else ""
                )
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connection": False,
            "error": str(e)
        }


async def init_database():
    """Initialize database connection and verify connectivity."""
    try:
        # Create engine and test connection
        engine = get_engine()

        # Test connection
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")

        logger.info("Database connection established successfully")

        # Create tables
        await create_db_and_tables()

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_database():
    """Close database connections gracefully."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database connections closed")


# Database migration utilities (for Alembic integration)
def get_database_url() -> str:
    """Get database URL for Alembic migrations."""
    return settings.SQLALCHEMY_DATABASE_URI


def get_database_url_sync() -> str:
    """Get synchronous database URL for tools that need it."""
    return settings.SQLALCHEMY_DATABASE_URI.replace("postgresql+asyncpg://", "postgresql://")
