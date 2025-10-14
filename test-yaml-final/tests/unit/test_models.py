"""
Unit tests for data models.

Tests the core data models including validation, relationships,
and business logic.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from pydantic import ValidationError

from app.core.models import User, Post, UserCreate, UserUpdate, PostCreate, PostUpdate
from app.core.security import get_password_hash


class TestUserModel:
    """Test cases for User model."""
    
    def test_user_creation(self, db_session):
        """Test basic user creation."""
        user = User(
            email="test@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Test User",
            is_active=True
        )
        
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_email_unique(self, db_session):
        """Test that email addresses must be unique."""
        user1 = User(
            email="test@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Test User 1"
        )
        
        user2 = User(
            email="test@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Test User 2"
        )
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_user_defaults(self, db_session):
        """Test user default values."""
        user = User(
            email="test@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Test User"
        )
        
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_str_representation(self, db_session):
        """Test string representation of user."""
        user = User(
            email="test@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Test User"
        )
        
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert str(user) == f"User(id={user.id}, email=test@example.com)"
    
    def test_user_posts_relationship(self, db_session):
        """Test user-posts relationship."""
        user = User(
            email="test@example.com",
            hashed_password=get_password_hash("password123"),
            full_name="Test User"
        )
        
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        post1 = Post(
            title="First Post",
            content="Content of first post",
            author_id=user.id
        )
        
        post2 = Post(
            title="Second Post", 
            content="Content of second post",
            author_id=user.id
        )
        
        db_session.add_all([post1, post2])
        db_session.commit()
        
        # Test relationship
        assert len(user.posts) == 2
        assert post1 in user.posts
        assert post2 in user.posts
        assert post1.author == user
        assert post2.author == user


class TestPostModel:
    """Test cases for Post model."""
    
    def test_post_creation(self, db_session, test_user):
        """Test basic post creation."""
        post = Post(
            title="Test Post",
            content="This is a test post content.",
            is_published=True,
            author_id=test_user.id
        )
        
        db_session.add(post)
        db_session.commit()
        db_session.refresh(post)
        
        assert post.id is not None
        assert post.title == "Test Post"
        assert post.content == "This is a test post content."
        assert post.is_published is True
        assert post.author_id == test_user.id
        assert post.created_at is not None
        assert post.updated_at is not None
    
    def test_post_defaults(self, db_session, test_user):
        """Test post default values."""
        post = Post(
            title="Test Post",
            content="Test content",
            author_id=test_user.id
        )
        
        db_session.add(post)
        db_session.commit()
        db_session.refresh(post)
        
        assert post.is_published is False
        assert post.created_at is not None
        assert post.updated_at is not None
    
    def test_post_author_relationship(self, db_session, test_user):
        """Test post-author relationship."""
        post = Post(
            title="Test Post",
            content="Test content",
            author_id=test_user.id
        )
        
        db_session.add(post)
        db_session.commit()
        db_session.refresh(post)
        
        assert post.author == test_user
        assert post in test_user.posts
    
    def test_post_str_representation(self, db_session, test_user):
        """Test string representation of post."""
        post = Post(
            title="Test Post",
            content="Test content",
            author_id=test_user.id
        )
        
        db_session.add(post)
        db_session.commit()
        db_session.refresh(post)
        
        assert str(post) == f"Post(id={post.id}, title=Test Post)"


class TestUserCreateSchema:
    """Test cases for UserCreate Pydantic schema."""
    
    def test_valid_user_create(self):
        """Test valid user creation data."""
        user_data = {
            "email": "test@example.com",
            "password": "ValidPassword123!",
            "full_name": "Test User"
        }
        
        user = UserCreate(**user_data)
        
        assert user.email == "test@example.com"
        assert user.password == "ValidPassword123!"
        assert user.full_name == "Test User"
    
    def test_invalid_email(self):
        """Test invalid email validation."""
        user_data = {
            "email": "invalid-email",
            "password": "ValidPassword123!",
            "full_name": "Test User"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        
        assert "email" in str(exc_info.value)
    
    def test_weak_password(self):
        """Test weak password validation."""
        user_data = {
            "email": "test@example.com",
            "password": "123",
            "full_name": "Test User"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        
        assert "password" in str(exc_info.value)
    
    def test_empty_full_name(self):
        """Test empty full name validation."""
        user_data = {
            "email": "test@example.com",
            "password": "ValidPassword123!",
            "full_name": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        
        assert "full_name" in str(exc_info.value)


class TestUserUpdateSchema:
    """Test cases for UserUpdate Pydantic schema."""
    
    def test_partial_update(self):
        """Test partial user update."""
        user_data = {
            "full_name": "Updated Name"
        }
        
        user = UserUpdate(**user_data)
        
        assert user.full_name == "Updated Name"
        assert user.email is None
        assert user.password is None
    
    def test_empty_update(self):
        """Test empty user update."""
        user = UserUpdate()
        
        assert user.email is None
        assert user.password is None
        assert user.full_name is None


class TestPostCreateSchema:
    """Test cases for PostCreate Pydantic schema."""
    
    def test_valid_post_create(self):
        """Test valid post creation data."""
        post_data = {
            "title": "Test Post",
            "content": "This is test content.",
            "is_published": True
        }
        
        post = PostCreate(**post_data)
        
        assert post.title == "Test Post"
        assert post.content == "This is test content."
        assert post.is_published is True
    
    def test_empty_title(self):
        """Test empty title validation."""
        post_data = {
            "title": "",
            "content": "Test content",
            "is_published": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PostCreate(**post_data)
        
        assert "title" in str(exc_info.value)
    
    def test_empty_content(self):
        """Test empty content validation."""
        post_data = {
            "title": "Test Post",
            "content": "",
            "is_published": True
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PostCreate(**post_data)
        
        assert "content" in str(exc_info.value)


class TestPostUpdateSchema:
    """Test cases for PostUpdate Pydantic schema."""
    
    def test_partial_update(self):
        """Test partial post update."""
        post_data = {
            "title": "Updated Title"
        }
        
        post = PostUpdate(**post_data)
        
        assert post.title == "Updated Title"
        assert post.content is None
        assert post.is_published is None
    
    def test_empty_update(self):
        """Test empty post update."""
        post = PostUpdate()
        
        assert post.title is None
        assert post.content is None
        assert post.is_published is None


class TestModelValidation:
    """Test cases for model validation and constraints."""
    
    def test_user_email_format(self, db_session):
        """Test email format validation."""
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test..test@example.com",
            "test@example..com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                User(
                    email=email,
                    hashed_password=get_password_hash("password123"),
                    full_name="Test User"
                )
    
    def test_post_title_length(self, db_session, test_user):
        """Test post title length constraints."""
        # Test maximum length
        long_title = "a" * 201  # Assuming max length is 200
        
        with pytest.raises(ValidationError):
            Post(
                title=long_title,
                content="Test content",
                author_id=test_user.id
            )
    
    def test_post_content_length(self, db_session, test_user):
        """Test post content length constraints."""
        # Test maximum length
        long_content = "a" * 10001  # Assuming max length is 10000
        
        with pytest.raises(ValidationError):
            Post(
                title="Test Post",
                content=long_content,
                author_id=test_user.id
            )
