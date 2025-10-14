"""
Caching utilities for Template Heaven.

This module provides a simple file-based caching system for template
metadata and downloaded content with TTL support.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta

from .logger import get_logger
from .sqlite_cache import SQLiteCache

logger = get_logger(__name__)


class Cache:
    """
    Advanced caching system with SQLite backend and file fallback.

    Provides high-performance caching for template metadata, search results,
    repository data, and validation results using SQLite as primary storage
    with file-based fallback for compatibility.

    Features:
    - SQLite-based metadata storage
    - Template and repository caching
    - Search result caching
    - Validation result caching
    - Automatic cleanup of expired entries
    - Thread-safe operations

    Attributes:
        cache_dir: Directory for cache files and SQLite database
        default_ttl: Default time-to-live in seconds
        max_size: Maximum cache size in bytes (optional, legacy)
        sqlite_cache: SQLite cache backend instance
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        default_ttl: int = 3600,  # 1 hour
        max_size: Optional[int] = None
    ):
        """
        Initialize cache with SQLite backend.

        Args:
            cache_dir: Directory to store cache files and SQLite database
            default_ttl: Default TTL in seconds
            max_size: Maximum cache size in bytes (optional, legacy)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size = max_size

        # Initialize SQLite cache
        db_path = self.cache_dir / "cache.db"
        self.sqlite_cache = SQLiteCache(db_path, max_age_days=30)

        logger.debug(f"Advanced cache initialized: {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get cache file path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Hash the key to create a safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, cache_path: Path, ttl: int) -> bool:
        """
        Check if cache entry is expired.
        
        Args:
            cache_path: Path to cache file
            ttl: Time-to-live in seconds
            
        Returns:
            True if expired
        """
        if not cache_path.exists():
            return True
        
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_time = file_time + timedelta(seconds=ttl)
        return datetime.now() > expiry_time
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache using SQLite backend with file fallback.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        # Try SQLite cache first
        try:
            result = self.sqlite_cache.get(key)
            if result is not None:
                logger.debug(f"SQLite cache hit: {key}")
                return result
        except Exception as e:
            logger.warning(f"SQLite cache error for {key}: {e}")

        # Fallback to file-based cache
        cache_path = self._get_cache_path(key)
        if not cache_path.exists() or self._is_expired(cache_path, self.default_ttl):
            logger.debug(f"Cache miss: {key}")
            return default

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"File cache hit: {key}")
            return data
        except (pickle.PickleError, OSError) as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache using SQLite backend with file fallback.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful
        """
        ttl_seconds = ttl or self.default_ttl

        # Try SQLite cache first
        try:
            if self.sqlite_cache.set(key, value, ttl_seconds):
                logger.debug(f"SQLite cache set: {key} (TTL: {ttl_seconds}s)")
                return True
        except Exception as e:
            logger.warning(f"SQLite cache error for {key}: {e}")

        # Fallback to file-based cache
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"File cache set: {key} (TTL: {ttl_seconds}s)")
            return True
        except (pickle.PickleError, OSError) as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache using SQLite backend with file fallback.

        Args:
            key: Cache key

        Returns:
            True if deleted or didn't exist
        """
        # Try SQLite cache first
        try:
            if self.sqlite_cache.delete(key):
                logger.debug(f"SQLite cache deleted: {key}")
                return True
        except Exception as e:
            logger.warning(f"SQLite cache delete error for {key}: {e}")

        # Fallback to file-based cache
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"File cache deleted: {key}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries from both SQLite and file caches.

        Returns:
            True if successful
        """
        success = True

        # Clear SQLite cache
        try:
            if not self.sqlite_cache.clear():
                success = False
        except Exception as e:
            logger.error(f"Failed to clear SQLite cache: {e}")
            success = False

        # Clear file cache
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            logger.info("File cache cleared")
        except OSError as e:
            logger.error(f"Failed to clear file cache: {e}")
            success = False

        if success:
            logger.info("All caches cleared")
        return success
    
    def cleanup(self) -> int:
        """
        Clean up expired cache entries from both SQLite and file caches.

        Returns:
            Number of entries cleaned up
        """
        cleaned = 0

        # Clean up SQLite cache
        try:
            cleaned += self.sqlite_cache.cleanup_expired()
        except Exception as e:
            logger.error(f"Failed to cleanup SQLite cache: {e}")

        # Clean up file cache
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                if self._is_expired(cache_file, self.default_ttl):
                    cache_file.unlink()
                    cleaned += 1

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired cache entries")

        except OSError as e:
            logger.error(f"Failed to cleanup file cache: {e}")

        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics from both SQLite and file caches.

        Returns:
            Dictionary with cache statistics
        """
        stats = {}

        # Get SQLite cache stats
        try:
            sqlite_stats = self.sqlite_cache.get_stats()
            stats.update({
                "sqlite_" + k: v for k, v in sqlite_stats.items()
            })
        except Exception as e:
            logger.error(f"Failed to get SQLite cache stats: {e}")

        # Get file cache stats
        total_files = 0
        total_size = 0
        expired_files = 0

        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                total_files += 1
                total_size += cache_file.stat().st_size

                if self._is_expired(cache_file, self.default_ttl):
                    expired_files += 1
        except OSError as e:
            logger.error(f"Failed to get file cache stats: {e}")

        stats.update({
            "file_total_entries": total_files,
            "file_expired_entries": expired_files,
            "file_active_entries": total_files - expired_files,
            "file_total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "max_size": self.max_size,
        })

        return stats

    # SQLite-specific methods for advanced caching features

    def store_template_metadata(self, template_data: Dict[str, Any]) -> bool:
        """
        Store template metadata in SQLite cache.

        Args:
            template_data: Template metadata dictionary

        Returns:
            True if successful
        """
        try:
            return self.sqlite_cache.store_template_metadata(template_data)
        except Exception as e:
            logger.error(f"Failed to store template metadata: {e}")
            return False

    def get_template_metadata(self, name: str, stack: str) -> Optional[Dict[str, Any]]:
        """
        Get template metadata from SQLite cache.

        Args:
            name: Template name
            stack: Template stack

        Returns:
            Template metadata or None
        """
        try:
            return self.sqlite_cache.get_template_metadata(name, stack)
        except Exception as e:
            logger.error(f"Failed to get template metadata: {e}")
            return None

    def store_repository_metadata(self, repo_data: Dict[str, Any]) -> bool:
        """
        Store repository metadata in SQLite cache.

        Args:
            repo_data: Repository metadata from GitHub API

        Returns:
            True if successful
        """
        try:
            return self.sqlite_cache.store_repository_metadata(repo_data)
        except Exception as e:
            logger.error(f"Failed to store repository metadata: {e}")
            return False

    def find_template_candidates(self, stack: Optional[str] = None,
                               min_potential: float = 0.5, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find template candidate repositories from SQLite cache.

        Args:
            stack: Optional stack filter
            min_potential: Minimum template potential score
            limit: Maximum results

        Returns:
            List of repository candidates
        """
        try:
            return self.sqlite_cache.find_template_candidates(stack, min_potential, limit)
        except Exception as e:
            logger.error(f"Failed to find template candidates: {e}")
            return []

    def cache_search_results(self, query: str, stack_filter: Optional[str],
                           results: List[Dict], ttl_seconds: int = 3600) -> bool:
        """
        Cache search results in SQLite.

        Args:
            query: Search query
            stack_filter: Optional stack filter
            results: Search results to cache
            ttl_seconds: Cache TTL

        Returns:
            True if successful
        """
        try:
            return self.sqlite_cache.cache_search_results(query, stack_filter, results, ttl_seconds)
        except Exception as e:
            logger.error(f"Failed to cache search results: {e}")
            return False

    def get_cached_search_results(self, query: str, stack_filter: Optional[str]) -> Optional[List[Dict]]:
        """
        Get cached search results from SQLite.

        Args:
            query: Search query
            stack_filter: Optional stack filter

        Returns:
            Cached results or None
        """
        try:
            return self.sqlite_cache.get_cached_search_results(query, stack_filter)
        except Exception as e:
            logger.error(f"Failed to get cached search results: {e}")
            return None

    def cache_validation_result(self, template_name: str, stack: str,
                              validation_result: Dict, ttl_seconds: int = 86400) -> bool:
        """
        Cache template validation result in SQLite.

        Args:
            template_name: Template name
            stack: Template stack
            validation_result: Validation result to cache
            ttl_seconds: Cache TTL

        Returns:
            True if successful
        """
        try:
            return self.sqlite_cache.cache_validation_result(template_name, stack, validation_result, ttl_seconds)
        except Exception as e:
            logger.error(f"Failed to cache validation result: {e}")
            return False

    def get_cached_validation_result(self, template_name: str, stack: str) -> Optional[Dict]:
        """
        Get cached validation result from SQLite.

        Args:
            template_name: Template name
            stack: Template stack

        Returns:
            Cached validation result or None
        """
        try:
            return self.sqlite_cache.get_cached_validation_result(template_name, stack)
        except Exception as e:
            logger.error(f"Failed to get cached validation result: {e}")
            return None
    
    def get_size(self) -> int:
        """
        Get total cache size in bytes.
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                total_size += cache_file.stat().st_size
        except OSError as e:
            logger.error(f"Failed to get cache size: {e}")
        
        return total_size
    
    def is_full(self) -> bool:
        """
        Check if cache is full (if max_size is set).
        
        Returns:
            True if cache is full
        """
        if self.max_size is None:
            return False
        
        return self.get_size() >= self.max_size


class JSONCache(Cache):
    """
    JSON-based cache for human-readable data.
    
    Extends the base Cache class to store data in JSON format
    instead of pickle, making cache files human-readable.
    """
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from JSON cache.
        
        Args:
            key: Cache key
            default: Default value if not found or expired
            
        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists() or self._is_expired(cache_path, self.default_ttl):
            logger.debug(f"JSON cache miss: {key}")
            return default
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"JSON cache hit: {key}")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load JSON cache entry {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in JSON cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)
        ttl = ttl or self.default_ttl
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, indent=2, default=str)
            logger.debug(f"JSON cache set: {key} (TTL: {ttl}s)")
            return True
        except (TypeError, OSError) as e:
            logger.error(f"Failed to cache {key} as JSON: {e}")
            return False
