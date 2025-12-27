"""
Database connection management for Template Heaven.

This module provides database connection setup, session management,
and connection utilities.
"""

import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData

from ..api.dependencies import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Database metadata
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)


class Base(DeclarativeBase):
    """Base class for all database models."""
    metadata = metadata


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection and session factory."""
        if self._initialized:
            return
        
        settings = get_settings()
        
        # Create async engine
        self._engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            future=True,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        self._initialized = True
        logger.info("Database manager initialized successfully")
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self._initialized:
            await self.initialize()
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        if not self._initialized:
            await self.initialize()
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("Database tables dropped successfully")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        if not self._initialized:
            await self.initialize()
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        return self._engine


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    if not db_manager._initialized:
        await db_manager.initialize()
    
    async with db_manager.get_session() as session:
        yield session


async def init_database() -> None:
    """Initialize database connection and create tables."""
    await db_manager.initialize()
    await db_manager.create_tables()


async def close_database() -> None:
    """Close database connections."""
    await db_manager.close()
