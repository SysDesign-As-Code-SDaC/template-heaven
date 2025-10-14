"""
Pytest configuration and fixtures for comprehensive testing.

This module provides shared fixtures and configuration for all test types:
- Unit tests
- Integration tests  
- End-to-end tests
- Performance tests
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.database import get_db, Base
from app.main import app
from app.core.models import User, Post
from app.core.security import create_access_token


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Test settings with in-memory database."""
    settings = get_settings()
    settings.DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    settings.SECRET_KEY = "test-secret-key"
    settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    settings.ENVIRONMENT = "test"
    return settings


@pytest.fixture(scope="session")
async def test_engine(test_settings):
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.DATABASE_URL,
        echo=False,
        future=True
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Clean up
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def client(db_session: AsyncSession) -> Generator[TestClient, None, None]:
    """Create a test client with database session override."""
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


# User fixtures
@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
        "is_active": True
    }


@pytest.fixture
async def test_user(db_session: AsyncSession, test_user_data):
    """Create a test user in the database."""
    from app.core.security import get_password_hash
    
    user = User(
        email=test_user_data["email"],
        hashed_password=get_password_hash(test_user_data["password"]),
        full_name=test_user_data["full_name"],
        is_active=test_user_data["is_active"]
    )
    
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    return user


@pytest.fixture
def test_user_token(test_user):
    """Create an access token for the test user."""
    return create_access_token(data={"sub": test_user.email})


@pytest.fixture
def authenticated_headers(test_user_token):
    """Headers with authentication token."""
    return {"Authorization": f"Bearer {test_user_token}"}


# Post fixtures
@pytest.fixture
def test_post_data():
    """Sample post data for testing."""
    return {
        "title": "Test Post",
        "content": "This is a test post content.",
        "is_published": True
    }


@pytest.fixture
async def test_post(db_session: AsyncSession, test_user, test_post_data):
    """Create a test post in the database."""
    post = Post(
        title=test_post_data["title"],
        content=test_post_data["content"],
        is_published=test_post_data["is_published"],
        author_id=test_user.id
    )
    
    db_session.add(post)
    await db_session.commit()
    await db_session.refresh(post)
    
    return post


# Mock fixtures
@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.exists.return_value = False
    return mock


@pytest.fixture
def mock_email_service():
    """Mock email service."""
    mock = AsyncMock()
    mock.send_email.return_value = True
    mock.send_verification_email.return_value = True
    mock.send_password_reset_email.return_value = True
    return mock


@pytest.fixture
def mock_file_storage():
    """Mock file storage service."""
    mock = AsyncMock()
    mock.upload_file.return_value = "https://example.com/file.jpg"
    mock.delete_file.return_value = True
    mock.get_file_url.return_value = "https://example.com/file.jpg"
    return mock


# Database fixtures for integration tests
@pytest.fixture
async def sample_users(db_session: AsyncSession):
    """Create multiple test users."""
    from app.core.security import get_password_hash
    
    users_data = [
        {
            "email": "user1@example.com",
            "password": "password123",
            "full_name": "User One",
            "is_active": True
        },
        {
            "email": "user2@example.com", 
            "password": "password123",
            "full_name": "User Two",
            "is_active": True
        },
        {
            "email": "inactive@example.com",
            "password": "password123", 
            "full_name": "Inactive User",
            "is_active": False
        }
    ]
    
    users = []
    for user_data in users_data:
        user = User(
            email=user_data["email"],
            hashed_password=get_password_hash(user_data["password"]),
            full_name=user_data["full_name"],
            is_active=user_data["is_active"]
        )
        db_session.add(user)
        users.append(user)
    
    await db_session.commit()
    
    for user in users:
        await db_session.refresh(user)
    
    return users


@pytest.fixture
async def sample_posts(db_session: AsyncSession, sample_users):
    """Create multiple test posts."""
    posts_data = [
        {
            "title": "First Post",
            "content": "Content of the first post.",
            "is_published": True,
            "author_id": sample_users[0].id
        },
        {
            "title": "Second Post", 
            "content": "Content of the second post.",
            "is_published": True,
            "author_id": sample_users[1].id
        },
        {
            "title": "Draft Post",
            "content": "This is a draft post.",
            "is_published": False,
            "author_id": sample_users[0].id
        }
    ]
    
    posts = []
    for post_data in posts_data:
        post = Post(**post_data)
        db_session.add(post)
        posts.append(post)
    
    await db_session.commit()
    
    for post in posts:
        await db_session.refresh(post)
    
    return posts


# Performance test fixtures
@pytest.fixture
def performance_data():
    """Large dataset for performance testing."""
    return {
        "users": [
            {
                "email": f"user{i}@example.com",
                "password": "password123",
                "full_name": f"User {i}",
                "is_active": True
            }
            for i in range(1000)
        ],
        "posts": [
            {
                "title": f"Post {i}",
                "content": f"Content of post {i}",
                "is_published": True,
                "author_id": (i % 100) + 1  # Distribute across first 100 users
            }
            for i in range(5000)
        ]
    }


# API test fixtures
@pytest.fixture
def api_test_data():
    """Comprehensive API test data."""
    return {
        "valid_user": {
            "email": "api@example.com",
            "password": "ValidPassword123!",
            "full_name": "API Test User"
        },
        "invalid_user": {
            "email": "invalid-email",
            "password": "123",
            "full_name": ""
        },
        "valid_post": {
            "title": "API Test Post",
            "content": "This is a test post for API testing.",
            "is_published": True
        },
        "invalid_post": {
            "title": "",
            "content": "",
            "is_published": "invalid"
        }
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_database(db_session: AsyncSession):
    """Clean up database after each test."""
    yield
    
    # Clean up all tables
    for table in reversed(Base.metadata.sorted_tables):
        await db_session.execute(table.delete())
    await db_session.commit()


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that take longer than 1 second
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
