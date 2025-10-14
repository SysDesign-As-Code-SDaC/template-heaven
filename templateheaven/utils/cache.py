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

logger = get_logger(__name__)


class Cache:
    """
    Simple file-based cache with TTL support.
    
    Provides caching functionality for template metadata, search results,
    and other data that can be reused across sessions.
    
    Attributes:
        cache_dir: Directory to store cache files
        default_ttl: Default time-to-live in seconds
        max_size: Maximum cache size in bytes (optional)
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        default_ttl: int = 3600,  # 1 hour
        max_size: Optional[int] = None
    ):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default TTL in seconds
            max_size: Maximum cache size in bytes (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size = max_size
        
        logger.debug(f"Cache initialized: {self.cache_dir}")
    
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
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found or expired
            
        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists() or self._is_expired(cache_path, self.default_ttl):
            logger.debug(f"Cache miss: {key}")
            return default
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit: {key}")
            return data
        except (pickle.PickleError, OSError) as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)
        ttl = ttl or self.default_ttl
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except (pickle.PickleError, OSError) as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted or didn't exist
        """
        cache_path = self._get_cache_path(key)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Cache deleted: {key}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            logger.info("Cache cleared")
            return True
        except OSError as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def cleanup(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                if self._is_expired(cache_file, self.default_ttl):
                    cache_file.unlink()
                    cleaned += 1
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired cache entries")
            
            return cleaned
        except OSError as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
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
            logger.error(f"Failed to get cache stats: {e}")
        
        return {
            "total_entries": total_files,
            "expired_entries": expired_files,
            "active_entries": total_files - expired_files,
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
            "max_size": self.max_size,
        }
    
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
