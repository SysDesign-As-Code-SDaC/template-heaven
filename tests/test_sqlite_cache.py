"""
Tests for SQLite Cache System.

This module contains comprehensive tests for the SQLite-based caching system
including metadata storage, search caching, and validation result caching.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from templateheaven.utils.sqlite_cache import SQLiteCache


class TestSQLiteCache:
    """Test SQLiteCache functionality."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create a test SQLite cache instance."""
        db_path = temp_cache_dir / "test_cache.db"
        return SQLiteCache(db_path, max_age_days=7)

    def test_initialization(self, cache, temp_cache_dir):
        """Test cache initialization."""
        assert cache.db_path == temp_cache_dir / "test_cache.db"
        assert cache.max_age_days == 7
        assert cache.db_path.exists()

    def test_basic_cache_operations(self, cache):
        """Test basic get/set operations."""
        # Test set and get
        assert cache.set("test_key", {"data": "value"}, ttl_seconds=3600)
        result = cache.get("test_key")
        assert result == {"data": "value"}

        # Test get with default
        assert cache.get("nonexistent_key", "default") == "default"

        # Test delete
        assert cache.delete("test_key")
        assert cache.get("test_key") is None

        # Test clear
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_expiration(self, cache):
        """Test cache entry expiration."""
        # Set entry with short TTL
        cache.set("short_ttl", "value", ttl_seconds=1)
        assert cache.get("short_ttl") == "value"

        # Wait for expiration (simulate by manually cleaning)
        import time
        time.sleep(2)
        cache.cleanup_expired()

        # Entry should be expired
        assert cache.get("short_ttl") is None

    def test_template_metadata_storage(self, cache):
        """Test template metadata storage and retrieval."""
        template_data = {
            "name": "test-template",
            "stack": "frontend",
            "source_url": "https://github.com/user/repo",
            "stars": 150,
            "forks": 25,
            "quality_score": 0.85
        }

        # Store metadata
        assert cache.store_template_metadata(template_data)

        # Retrieve metadata
        result = cache.get_template_metadata("test-template", "frontend")
        assert result is not None
        assert result["name"] == "test-template"
        assert result["stack"] == "frontend"
        assert result["stars"] == 150

        # Test non-existent template
        assert cache.get_template_metadata("nonexistent", "frontend") is None

    def test_repository_metadata_storage(self, cache):
        """Test repository metadata storage and retrieval."""
        repo_data = {
            "id": 12345,
            "name": "test-repo",
            "full_name": "user/test-repo",
            "description": "A test repository",
            "html_url": "https://github.com/user/test-repo",
            "stargazers_count": 100,
            "forks_count": 20,
            "language": "JavaScript",
            "topics": ["react", "template"],
            "license": {"name": "MIT"},
            "template_potential": 0.8,
            "quality_score": 0.9,
            "stack_suggestions": ["frontend"],
            "analysis_data": {"test": "data"}
        }

        # Store repository metadata
        assert cache.store_repository_metadata(repo_data)

        # Retrieve by GitHub ID
        result = cache.get_repository_metadata(12345)
        assert result is not None
        assert result["name"] == "test-repo"
        assert result["template_potential"] == 0.8
        assert json.loads(result["topics"]) == ["react", "template"]

        # Test non-existent repo
        assert cache.get_repository_metadata(99999) is None

    def test_template_candidate_search(self, cache):
        """Test finding template candidates."""
        # Store multiple repositories with different potentials
        repos = [
            {
                "id": 1, "name": "high-potential", "full_name": "user/high",
                "stargazers_count": 500, "template_potential": 0.9,
                "stack_suggestions": ["frontend"]
            },
            {
                "id": 2, "name": "medium-potential", "full_name": "user/medium",
                "stargazers_count": 200, "template_potential": 0.7,
                "stack_suggestions": ["backend"]
            },
            {
                "id": 3, "name": "low-potential", "full_name": "user/low",
                "stargazers_count": 50, "template_potential": 0.3,
                "stack_suggestions": ["frontend"]
            }
        ]

        for repo in repos:
            cache.store_repository_metadata(repo)

        # Find all candidates above threshold
        candidates = cache.find_template_candidates(min_potential=0.5)
        assert len(candidates) == 2

        # Sort by potential (should be descending)
        assert candidates[0]["template_potential"] >= candidates[1]["template_potential"]

        # Find candidates for specific stack
        frontend_candidates = cache.find_template_candidates(stack="frontend", min_potential=0.5)
        assert len(frontend_candidates) == 1
        assert frontend_candidates[0]["name"] == "high-potential"

    def test_search_result_caching(self, cache):
        """Test search result caching."""
        query = "react template"
        stack_filter = "frontend"
        results = [
            {"name": "repo1", "score": 0.9},
            {"name": "repo2", "score": 0.8}
        ]

        # Cache search results
        assert cache.cache_search_results(query, stack_filter, results, ttl_seconds=3600)

        # Retrieve cached results
        cached = cache.get_cached_search_results(query, stack_filter)
        assert cached == results

        # Test non-existent search
        assert cache.get_cached_search_results("nonexistent", None) is None

    def test_validation_result_caching(self, cache):
        """Test validation result caching."""
        template_name = "test-template"
        stack = "frontend"
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": ["Minor issue"],
            "quality_score": 8.5
        }

        # Cache validation result
        assert cache.cache_validation_result(template_name, stack, validation_result, ttl_seconds=86400)

        # Retrieve cached result
        cached = cache.get_cached_validation_result(template_name, stack)
        assert cached == validation_result

        # Test non-existent validation
        assert cache.get_cached_validation_result("nonexistent", "frontend") is None

    def test_cache_stats(self, cache):
        """Test cache statistics."""
        # Add some data
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.store_template_metadata({
            "name": "template1",
            "stack": "frontend",
            "stars": 100
        })

        cache.store_repository_metadata({
            "id": 123,
            "name": "repo1",
            "full_name": "user/repo1",
            "template_potential": 0.8
        })

        stats = cache.get_stats()

        # Check basic stats exist
        assert "sqlite_total_entries" in stats
        assert "sqlite_template_count" in stats
        assert "sqlite_repo_count" in stats
        assert stats["sqlite_template_count"] >= 1
        assert stats["sqlite_repo_count"] >= 1

    def test_cleanup_expired_entries(self, cache):
        """Test cleanup of expired entries."""
        # Set entries with different TTLs
        cache.set("short_ttl", "value1", ttl_seconds=1)
        cache.set("long_ttl", "value2", ttl_seconds=86400)

        # Both should be retrievable initially
        assert cache.get("short_ttl") == "value1"
        assert cache.get("long_ttl") == "value2"

        # Wait and cleanup
        import time
        time.sleep(2)
        removed = cache.cleanup_expired()

        # Short TTL entry should be removed
        assert removed >= 1
        assert cache.get("short_ttl") is None
        assert cache.get("long_ttl") == "value2"

    def test_thread_safety(self, cache):
        """Test thread-safe operations."""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                # Each worker does operations
                cache.set(f"worker_{worker_id}_key", f"worker_{worker_id}_value")
                result = cache.get(f"worker_{worker_id}_key")
                results.append((worker_id, result))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 10

        for worker_id, result in results:
            assert result == f"worker_{worker_id}_value"

    def test_error_handling(self, cache):
        """Test error handling for invalid operations."""
        # Test with invalid JSON (should not crash)
        cache.set("bad_json", set([1, 2, 3]))  # sets are not JSON serializable

        # Should handle gracefully
        result = cache.get("bad_json")
        assert result is None  # Should not crash, just return None

    def test_database_connection_handling(self, cache):
        """Test database connection handling."""
        # Test multiple operations
        for i in range(100):
            cache.set(f"test_key_{i}", f"test_value_{i}")
            result = cache.get(f"test_key_{i}")
            assert result == f"test_value_{i}"

        # Cleanup
        for i in range(100):
            cache.delete(f"test_key_{i}")

    def test_large_data_handling(self, cache):
        """Test handling of large data."""
        # Create large data structure
        large_data = {
            "big_list": list(range(1000)),
            "nested": {
                "data": "x" * 10000  # 10KB string
            }
        }

        # Should handle large data
        assert cache.set("large_data", large_data)
        result = cache.get("large_data")
        assert result == large_data

    def test_unicode_handling(self, cache):
        """Test Unicode string handling."""
        unicode_data = {
            "message": "Hello 世界 🌍",
            "emojis": "🚀⭐🔥",
            "special": "ñáéíóú"
        }

        assert cache.set("unicode_data", unicode_data)
        result = cache.get("unicode_data")
        assert result == unicode_data
