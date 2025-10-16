"""
Pytest configuration and shared fixtures for comprehensive testing.

This conftest.py provides:
- Global fixtures for common test data
- Pytest configuration and hooks
- Shared utilities for all test modules
- Database and API client fixtures
- Mock and patch utilities
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from faker import Faker
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Initialize Faker for test data generation
fake = Faker()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that test API endpoints"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def temp_file(temp_dir):
    """Create a temporary file for individual tests."""
    def _temp_file(content: str = "", suffix: str = ".txt") -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return Path(path)
    return _temp_file


@pytest.fixture(scope="session")
def faker():
    """Provide Faker instance for generating test data."""
    return fake


@pytest.fixture(scope="function")
def fake_user(faker):
    """Generate fake user data."""
    return {
        "id": faker.uuid4(),
        "name": faker.name(),
        "email": faker.email(),
        "username": faker.user_name(),
        "password": faker.password(),
        "created_at": faker.date_time_this_year(),
    }


@pytest.fixture(scope="function")
def fake_company(faker):
    """Generate fake company data."""
    return {
        "id": faker.uuid4(),
        "name": faker.company(),
        "domain": faker.domain_name(),
        "address": faker.address(),
        "phone": faker.phone_number(),
    }


@pytest.fixture(scope="function")
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/testdb",
        "REDIS_URL": "redis://localhost:6379/0",
        "SECRET_KEY": "test-secret-key",
        "DEBUG": "True",
        "API_KEY": "test-api-key",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(scope="function")
def mock_requests():
    """Mock HTTP requests for testing."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post, \
         patch('requests.put') as mock_put, \
         patch('requests.delete') as mock_delete:

        # Configure default responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "success"}

        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": 1, "created": True}

        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {"updated": True}

        mock_delete.return_value.status_code = 204

        yield {
            "get": mock_get,
            "post": mock_post,
            "put": mock_put,
            "delete": mock_delete,
        }


@pytest.fixture(scope="function")
def mock_httpx():
    """Mock HTTPX async client for testing."""
    async def mock_response(*args, **kwargs):
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"status": "success"}
        mock.text = '{"status": "success"}'
        return mock

    with patch('httpx.AsyncClient.get', side_effect=mock_response), \
         patch('httpx.AsyncClient.post', side_effect=mock_response), \
         patch('httpx.AsyncClient.put', side_effect=mock_response), \
         patch('httpx.AsyncClient.delete', side_effect=mock_response):

        yield


@pytest.fixture(scope="session")
def db_engine():
    """Create a test database engine."""
    # Use SQLite for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a database session for testing."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def mock_db_session():
    """Mock database session for testing."""
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    mock_session.query = MagicMock()
    yield mock_session


@pytest.fixture(scope="function")
def sample_data():
    """Provide sample test data."""
    return {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
        ],
        "products": [
            {"id": 1, "name": "Widget A", "price": 10.99},
            {"id": 2, "name": "Widget B", "price": 15.49},
            {"id": 3, "name": "Widget C", "price": 8.75},
        ],
        "orders": [
            {"id": 1, "user_id": 1, "product_id": 1, "quantity": 2},
            {"id": 2, "user_id": 2, "product_id": 2, "quantity": 1},
        ]
    }


@pytest.fixture(scope="function")
def mock_logger():
    """Mock logger for testing."""
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture(scope="function")
def mock_cache():
    """Mock cache for testing."""
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    yield mock_cache


@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    yield mock_redis


@pytest.fixture(scope="function")
def mock_s3():
    """Mock S3 client for testing."""
    mock_s3 = MagicMock()
    mock_s3.upload_file.return_value = True
    mock_s3.download_file.return_value = True
    mock_s3.delete_object.return_value = True
    mock_s3.list_objects_v2.return_value = {"Contents": []}
    yield mock_s3


@pytest.fixture(scope="function")
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"Test execution time: {end_time - start_time:.4f} seconds")


@pytest.fixture(scope="function")
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = final_memory - initial_memory
    print(f"Memory delta: {memory_delta:.2f} MB")


# Async fixtures
@pytest_asyncio.fixture(scope="function")
async def async_client():
    """Create an async HTTP client for testing."""
    async with AsyncClient(base_url="http://testserver") as client:
        yield client


@pytest_asyncio.fixture(scope="function")
async def mock_async_session():
    """Mock async database session."""
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    mock_session.query = MagicMock()

    # Make methods async
    async def async_add(item):
        return mock_session.add(item)

    async def async_commit():
        return mock_session.commit()

    mock_session.add = async_add
    mock_session.commit = async_commit

    yield mock_session


# Custom pytest hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Mark integration tests
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark e2e tests
        if "test_e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Mark performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information."""
    if hasattr(terminalreporter, 'stats'):
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        skipped = len(terminalreporter.stats.get('skipped', []))

        terminalreporter.write_sep(
            f"\nTest Summary: {passed} passed, {failed} failed, {skipped} skipped\n"
        )