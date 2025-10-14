"""
Advanced SQLite-based caching system for Template Heaven.

This module provides a high-performance caching layer using SQLite
for storing metadata, search results, and template analysis data.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from contextlib import contextmanager

from .logger import get_logger

logger = get_logger(__name__)


class SQLiteCache:
    """
    Advanced SQLite-based caching system with metadata storage.

    Features:
    - Thread-safe operations
    - TTL (time-to-live) support
    - Metadata storage for templates and repositories
    - Search result caching
    - Template validation status tracking
    - Connection pooling and automatic cleanup
    """

    def __init__(self, db_path: Path, max_age_days: int = 30):
        """
        Initialize SQLite cache.

        Args:
            db_path: Path to SQLite database file
            max_age_days: Default maximum age for cached data in days
        """
        self.db_path = db_path
        self.max_age_days = max_age_days
        self._lock = threading.RLock()
        self._local = threading.local()

        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"SQLite cache initialized at {db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            self._local.connection.execute("PRAGMA synchronous = NORMAL")
            self._local.connection.execute("PRAGMA cache_size = 1000000")  # 1GB cache
            self._local.connection.execute("PRAGMA temp_store = MEMORY")

        try:
            yield self._local.connection
        except Exception as e:
            logger.error(f"Database error: {e}")
            # Reset connection on error
            if hasattr(self._local, 'connection'):
                try:
                    self._local.connection.close()
                except:
                    pass
                delattr(self._local, 'connection')
            raise

    def _init_database(self):
        """Initialize database schema."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Cache entries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TIMESTAMP,
                        ttl_seconds INTEGER
                    )
                """)

                # Template metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS template_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        stack TEXT NOT NULL,
                        source_url TEXT,
                        local_path TEXT,
                        github_id INTEGER,
                        stars INTEGER DEFAULT 0,
                        forks INTEGER DEFAULT 0,
                        growth_rate REAL DEFAULT 0.0,
                        quality_score REAL DEFAULT 0.0,
                        validation_status TEXT DEFAULT 'pending',
                        last_validated TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, stack)
                    )
                """)

                # Repository metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS repository_metadata (
                        github_id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        full_name TEXT NOT NULL,
                        description TEXT,
                        url TEXT,
                        clone_url TEXT,
                        language TEXT,
                        topics TEXT[],  -- JSON array
                        license TEXT,
                        size INTEGER,
                        default_branch TEXT DEFAULT 'main',
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_template BOOLEAN DEFAULT FALSE,
                        template_potential REAL DEFAULT 0.0,
                        quality_score REAL DEFAULT 0.0,
                        stack_suggestions TEXT[],  -- JSON array
                        analysis_data TEXT,  -- JSON blob
                        UNIQUE(full_name)
                    )
                """)

                # Search results cache
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS search_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        stack_filter TEXT,
                        results TEXT NOT NULL,  -- JSON blob
                        result_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)

                # Template validation results
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS validation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        template_name TEXT NOT NULL,
                        stack TEXT NOT NULL,
                        validation_result TEXT NOT NULL,  -- JSON blob
                        is_valid BOOLEAN DEFAULT FALSE,
                        issues TEXT[],  -- JSON array
                        warnings TEXT[],  -- JSON array
                        quality_score REAL DEFAULT 0.0,
                        validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)

                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_entries_expires
                    ON cache_entries(expires_at) WHERE expires_at IS NOT NULL
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_template_metadata_stack
                    ON template_metadata(stack)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_repository_metadata_template_potential
                    ON repository_metadata(template_potential DESC)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_search_results_query_stack
                    ON search_results(query, stack_filter)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_search_results_created_at
                    ON search_results(created_at)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_validation_results_template_stack
                    ON validation_results(template_name, stack)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_validation_results_validated_at
                    ON validation_results(validated_at)
                """)

                conn.commit()
                logger.debug("Database schema initialized")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT value, metadata FROM cache_entries
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """, (key,))

                row = cursor.fetchone()
                if row:
                    # Update access statistics
                    cursor.execute("""
                        UPDATE cache_entries
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE key = ?
                    """, (key,))
                    conn.commit()

                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError:
                        return row[0]

        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, metadata: Optional[Dict] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time-to-live in seconds
            metadata: Additional metadata

        Returns:
            True if successful
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                expires_at = None
                if ttl_seconds:
                    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

                try:
                    json_value = json.dumps(value)
                    json_metadata = json.dumps(metadata) if metadata else None

                    cursor.execute("""
                        INSERT OR REPLACE INTO cache_entries
                        (key, value, metadata, expires_at, ttl_seconds, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (key, json_value, json_metadata, expires_at, ttl_seconds))

                    conn.commit()
                    return True

                except Exception as e:
                    logger.error(f"Failed to cache value for key {key}: {e}")
                    return False

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted

    def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if successful
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries")
                conn.commit()
                return True

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_entries WHERE expires_at <= CURRENT_TIMESTAMP")
                removed = cursor.rowcount
                conn.commit()

                if removed > 0:
                    logger.info(f"Cleaned up {removed} expired cache entries")

                return removed

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # General stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_entries,
                        COUNT(CASE WHEN expires_at IS NOT NULL THEN 1 END) as expiring_entries,
                        COUNT(CASE WHEN expires_at <= CURRENT_TIMESTAMP THEN 1 END) as expired_entries,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses
                    FROM cache_entries
                """)

                row = cursor.fetchone()
                stats = dict(row) if row else {}

                # Template metadata stats
                cursor.execute("SELECT COUNT(*) as template_count FROM template_metadata")
                stats.update(dict(cursor.fetchone()))

                # Repository metadata stats
                cursor.execute("SELECT COUNT(*) as repo_count FROM repository_metadata")
                stats.update(dict(cursor.fetchone()))

                # Search results stats
                cursor.execute("SELECT COUNT(*) as search_count FROM search_results")
                stats.update(dict(cursor.fetchone()))

                return stats

    # Template metadata methods
    def store_template_metadata(self, template_data: Dict[str, Any]) -> bool:
        """
        Store template metadata.

        Args:
            template_data: Template metadata dictionary

        Returns:
            True if successful
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO template_metadata
                        (name, stack, source_url, local_path, github_id, stars, forks,
                         growth_rate, quality_score, validation_status, last_validated, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        template_data.get('name'),
                        template_data.get('stack'),
                        template_data.get('source_url'),
                        template_data.get('local_path'),
                        template_data.get('github_id'),
                        template_data.get('stars', 0),
                        template_data.get('forks', 0),
                        template_data.get('growth_rate', 0.0),
                        template_data.get('quality_score', 0.0),
                        template_data.get('validation_status', 'pending'),
                        template_data.get('last_validated')
                    ))

                    conn.commit()
                    return True

                except Exception as e:
                    logger.error(f"Failed to store template metadata: {e}")
                    return False

    def get_template_metadata(self, name: str, stack: str) -> Optional[Dict[str, Any]]:
        """
        Get template metadata.

        Args:
            name: Template name
            stack: Template stack

        Returns:
            Template metadata or None
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM template_metadata
                    WHERE name = ? AND stack = ?
                """, (name, stack))

                row = cursor.fetchone()
                return dict(row) if row else None

    # Repository metadata methods
    def store_repository_metadata(self, repo_data: Dict[str, Any]) -> bool:
        """
        Store repository metadata.

        Args:
            repo_data: Repository metadata from GitHub API

        Returns:
            True if successful
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO repository_metadata
                        (github_id, name, full_name, description, url, clone_url, language,
                         topics, license, size, default_branch, created_at, updated_at,
                         is_template, template_potential, quality_score, stack_suggestions, analysis_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        repo_data.get('id'),
                        repo_data.get('name'),
                        repo_data.get('full_name'),
                        repo_data.get('description'),
                        repo_data.get('html_url'),
                        repo_data.get('clone_url'),
                        repo_data.get('language'),
                        json.dumps(repo_data.get('topics', [])),
                        repo_data.get('license', {}).get('name') if repo_data.get('license') else None,
                        repo_data.get('size'),
                        repo_data.get('default_branch'),
                        repo_data.get('created_at'),
                        repo_data.get('updated_at'),
                        repo_data.get('is_template', False),
                        repo_data.get('template_potential', 0.0),
                        repo_data.get('quality_score', 0.0),
                        json.dumps(repo_data.get('stack_suggestions', [])),
                        json.dumps(repo_data.get('analysis_data', {}))
                    ))

                    conn.commit()
                    return True

                except Exception as e:
                    logger.error(f"Failed to store repository metadata: {e}")
                    return False

    def get_repository_metadata(self, github_id: int) -> Optional[Dict[str, Any]]:
        """
        Get repository metadata by GitHub ID.

        Args:
            github_id: GitHub repository ID

        Returns:
            Repository metadata or None
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM repository_metadata WHERE github_id = ?", (github_id,))
                row = cursor.fetchone()
                return dict(row) if row else None

    def find_template_candidates(self, stack: Optional[str] = None, min_potential: float = 0.5,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find template candidate repositories.

        Args:
            stack: Optional stack filter
            min_potential: Minimum template potential score
            limit: Maximum results

        Returns:
            List of repository candidates
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT * FROM repository_metadata
                    WHERE template_potential >= ?
                """

                params = [min_potential]

                if stack:
                    query += " AND ? = ANY(stack_suggestions)"
                    params.append(stack)

                query += " ORDER BY template_potential DESC, quality_score DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)

                return [dict(row) for row in cursor.fetchall()]

    # Search caching methods
    def cache_search_results(self, query: str, stack_filter: Optional[str],
                           results: List[Dict], ttl_seconds: int = 3600) -> bool:
        """
        Cache search results.

        Args:
            query: Search query
            stack_filter: Optional stack filter
            results: Search results to cache
            ttl_seconds: Cache TTL

        Returns:
            True if successful
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

                try:
                    cursor.execute("""
                        INSERT INTO search_results
                        (query, stack_filter, results, result_count, expires_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        query,
                        stack_filter,
                        json.dumps(results),
                        len(results),
                        expires_at
                    ))

                    conn.commit()
                    return True

                except Exception as e:
                    logger.error(f"Failed to cache search results: {e}")
                    return False

    def get_cached_search_results(self, query: str, stack_filter: Optional[str]) -> Optional[List[Dict]]:
        """
        Get cached search results.

        Args:
            query: Search query
            stack_filter: Optional stack filter

        Returns:
            Cached results or None
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT results FROM search_results
                    WHERE query = ? AND stack_filter IS ?
                      AND expires_at > CURRENT_TIMESTAMP
                    ORDER BY created_at DESC LIMIT 1
                """, (query, stack_filter))

                row = cursor.fetchone()
                if row:
                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError:
                        return None

        return None

    # Validation caching methods
    def cache_validation_result(self, template_name: str, stack: str,
                              validation_result: Dict, ttl_seconds: int = 86400) -> bool:
        """
        Cache template validation result.

        Args:
            template_name: Template name
            stack: Template stack
            validation_result: Validation result to cache
            ttl_seconds: Cache TTL

        Returns:
            True if successful
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

                try:
                    cursor.execute("""
                        INSERT INTO validation_results
                        (template_name, stack, validation_result, is_valid, issues, warnings,
                         quality_score, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        template_name,
                        stack,
                        json.dumps(validation_result),
                        validation_result.get('valid', False),
                        json.dumps(validation_result.get('issues', [])),
                        json.dumps(validation_result.get('warnings', [])),
                        validation_result.get('quality_score', 0.0),
                        expires_at
                    ))

                    conn.commit()
                    return True

                except Exception as e:
                    logger.error(f"Failed to cache validation result: {e}")
                    return False

    def get_cached_validation_result(self, template_name: str, stack: str) -> Optional[Dict]:
        """
        Get cached validation result.

        Args:
            template_name: Template name
            stack: Template stack

        Returns:
            Cached validation result or None
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT validation_result FROM validation_results
                    WHERE template_name = ? AND stack = ?
                      AND expires_at > CURRENT_TIMESTAMP
                    ORDER BY validated_at DESC LIMIT 1
                """, (template_name, stack))

                row = cursor.fetchone()
                if row:
                    try:
                        return json.loads(row[0])
                    except json.JSONDecodeError:
                        return None

        return None

    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            try:
                self._local.connection.close()
            except:
                pass
            delattr(self._local, 'connection')

    def __del__(self):
        """Cleanup on destruction."""
        self.close()
