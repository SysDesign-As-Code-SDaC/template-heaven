"""
Database management for MCP SDK Template.

Provides database connection management, migrations, and basic ORM functionality
for the MCP server and client applications.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import asyncpg

from .config import DatabaseSettings
from .exceptions import MCPSDKError, MCPConnectionError


logger = logging.getLogger(__name__)

# SQLAlchemy base class
Base = declarative_base()


class DatabaseManager:
    """
    Database connection and session management.
    
    Provides async database operations with connection pooling,
    health checks, and transaction management.
    """
    
    def __init__(self, settings: DatabaseSettings):
        """
        Initialize database manager.
        
        Args:
            settings: Database configuration settings
        """
        self.settings = settings
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize database connections and session factories.
        
        Raises:
            MCPConnectionError: If database connection fails
        """
        try:
            logger.info("Initializing database connections...")
            
            # Create sync engine for migrations and admin operations
            self.engine = create_engine(
                self.settings.url,
                poolclass=QueuePool,
                pool_size=self.settings.pool_size,
                max_overflow=self.settings.max_overflow,
                pool_timeout=self.settings.pool_timeout,
                pool_recycle=self.settings.pool_recycle,
                echo=False,
                future=True
            )
            
            # Create async engine for application operations
            async_url = self._convert_to_async_url(self.settings.url)
            self.async_engine = create_async_engine(
                async_url,
                pool_size=self.settings.pool_size,
                max_overflow=self.settings.max_overflow,
                pool_timeout=self.settings.pool_timeout,
                pool_recycle=self.settings.pool_recycle,
                echo=False,
                future=True
            )
            
            # Create session factories
            self.session_factory = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connections
            await self._test_connections()
            
            # Create tables
            await self._create_tables()
            
            self._initialized = True
            logger.info("Database initialization complete")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise MCPConnectionError(f"Database connection failed: {str(e)}")
    
    def _convert_to_async_url(self, url: str) -> str:
        """
        Convert database URL to async version.
        
        Args:
            url: Original database URL
            
        Returns:
            Async database URL
        """
        if url.startswith('postgresql://'):
            return url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        elif url.startswith('postgresql+psycopg2://'):
            return url.replace('postgresql+psycopg2://', 'postgresql+asyncpg://', 1)
        elif url.startswith('sqlite://'):
            return url.replace('sqlite://', 'sqlite+aiosqlite://', 1)
        else:
            return url
    
    async def _test_connections(self) -> None:
        """Test database connections."""
        try:
            # Test sync connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Test async connection
            async with self.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("Database connections tested successfully")
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise MCPConnectionError(f"Database connection test failed: {str(e)}")
    
    async def _create_tables(self) -> None:
        """Create database tables."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise MCPSDKError(f"Table creation failed: {str(e)}")
    
    @asynccontextmanager
    async def get_session(self):
        """
        Get database session context manager.
        
        Usage:
            async with db_manager.get_session() as session:
                # Use session for database operations
                pass
        """
        if not self._initialized:
            raise MCPSDKError("Database manager not initialized")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            parameters: Optional query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), parameters or {})
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise MCPSDKError(f"Query execution failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Health check results
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test basic connectivity
            result = await self.execute_query("SELECT 1 as health_check")
            
            end_time = asyncio.get_event_loop().time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Get connection pool status
            pool_status = {
                "size": self.async_engine.pool.size(),
                "checked_in": self.async_engine.pool.checkedin(),
                "checked_out": self.async_engine.pool.checkedout(),
                "overflow": self.async_engine.pool.overflow(),
                "invalid": self.async_engine.pool.invalid()
            }
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "pool_status": pool_status,
                "test_result": result[0] if result else None
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": None,
                "pool_status": None,
                "test_result": None
            }
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """
        Get database connection information.
        
        Returns:
            Connection information
        """
        try:
            # Get database version
            version_result = await self.execute_query("SELECT version() as version")
            version = version_result[0]["version"] if version_result else "Unknown"
            
            # Get database size
            size_result = await self.execute_query(
                "SELECT pg_database_size(current_database()) as size_bytes"
            )
            size_bytes = size_result[0]["size_bytes"] if size_result else 0
            
            return {
                "url": self.settings.url.split("@")[-1] if "@" in self.settings.url else "hidden",
                "version": version,
                "size_bytes": size_bytes,
                "pool_size": self.settings.pool_size,
                "max_overflow": self.settings.max_overflow,
                "pool_timeout": self.settings.pool_timeout,
                "pool_recycle": self.settings.pool_recycle
            }
            
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {
                "url": "unknown",
                "version": "unknown",
                "size_bytes": 0,
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup database connections."""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.engine:
                self.engine.dispose()
            
            logger.info("Database connections cleaned up")
            
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized."""
        return self._initialized
