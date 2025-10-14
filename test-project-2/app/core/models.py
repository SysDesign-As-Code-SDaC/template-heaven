"""
SQLAlchemy models for the application.

This module defines the database models using SQLAlchemy ORM with
proper relationships, constraints, and validation.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import re

Base = declarative_base()


class User(Base):
    """User model for authentication and user management."""
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # User information
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    
    # User status
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('ix_users_email_active', 'email', 'is_active'),
        CheckConstraint('length(email) >= 5', name='ck_users_email_length'),
        CheckConstraint('length(full_name) >= 1', name='ck_users_full_name_length'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            raise ValueError('Invalid email format')
        return email.lower()
    
    @validates('full_name')
    def validate_full_name(self, key, full_name):
        """Validate full name."""
        if not full_name or len(full_name.strip()) < 1:
            raise ValueError('Full name cannot be empty')
        return full_name.strip()
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"
    
    def __str__(self):
        return f"User(id={self.id}, email={self.email})"


class Post(Base):
    """Post model for content management."""
    
    __tablename__ = "posts"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Post content
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(String(500), nullable=True)
    
    # Post status
    is_published = Column(Boolean, default=False, nullable=False)
    is_featured = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    slug = Column(String(255), unique=True, index=True, nullable=True)
    tags = Column(String(500), nullable=True)  # Comma-separated tags
    view_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    author = relationship("User", back_populates="posts")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=True)
    
    # Constraints
    __table_args__ = (
        Index('ix_posts_author_created', 'author_id', 'created_at'),
        Index('ix_posts_published_created', 'is_published', 'created_at'),
        Index('ix_posts_slug', 'slug'),
        CheckConstraint('length(title) >= 1', name='ck_posts_title_length'),
        CheckConstraint('length(content) >= 10', name='ck_posts_content_length'),
        CheckConstraint('view_count >= 0', name='ck_posts_view_count_positive'),
    )
    
    @validates('title')
    def validate_title(self, key, title):
        """Validate post title."""
        if not title or len(title.strip()) < 1:
            raise ValueError('Post title cannot be empty')
        return title.strip()
    
    @validates('content')
    def validate_content(self, key, content):
        """Validate post content."""
        if not content or len(content.strip()) < 10:
            raise ValueError('Post content must be at least 10 characters')
        return content.strip()
    
    @validates('slug')
    def validate_slug(self, key, slug):
        """Validate and generate slug."""
        if slug:
            # Ensure slug is URL-safe
            slug = re.sub(r'[^a-zA-Z0-9\-_]', '-', slug.lower())
            slug = re.sub(r'-+', '-', slug).strip('-')
        return slug
    
    def __repr__(self):
        return f"<Post(id={self.id}, title={self.title})>"
    
    def __str__(self):
        return f"Post(id={self.id}, title={self.title})"


class AuditLog(Base):
    """Audit log for tracking changes and actions."""
    
    __tablename__ = "audit_logs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Audit information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(100), nullable=True)
    
    # Change details
    old_values = Column(Text, nullable=True)  # JSON string
    new_values = Column(Text, nullable=True)  # JSON string
    
    # Request context
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        Index('ix_audit_logs_user_created', 'user_id', 'created_at'),
        Index('ix_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('ix_audit_logs_action', 'action', 'created_at'),
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, user_id={self.user_id})>"


class Session(Base):
    """User session management."""
    
    __tablename__ = "sessions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Session information
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Session data
    data = Column(Text, nullable=True)  # JSON string
    
    # Session metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    
    # Constraints
    __table_args__ = (
        Index('ix_sessions_user_active', 'user_id', 'is_active'),
        Index('ix_sessions_expires', 'expires_at'),
    )
    
    def __repr__(self):
        return f"<Session(id={self.id}, session_id={self.session_id}, user_id={self.user_id})>"


class APIKey(Base):
    """API key management for external integrations."""
    
    __tablename__ = "api_keys"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # API key information
    key_name = Column(String(255), nullable=False)
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    key_prefix = Column(String(10), nullable=False)  # First few chars for identification
    
    # Ownership and permissions
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permissions = Column(Text, nullable=True)  # JSON string of permissions
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Constraints
    __table_args__ = (
        Index('ix_api_keys_user_active', 'user_id', 'is_active'),
        Index('ix_api_keys_prefix', 'key_prefix'),
        CheckConstraint('usage_count >= 0', name='ck_api_keys_usage_count_positive'),
    )
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, key_name={self.key_name}, user_id={self.user_id})>"
