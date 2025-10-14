"""
Database models for Template Heaven.

This module defines SQLAlchemy models for all database entities.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey,
    Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .connection import Base


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    # Relationships
    roles: Mapped[List["UserRole"]] = relationship(
        "UserRole", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    api_keys: Mapped[List["APIKey"]] = relationship(
        "APIKey", 
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_users_username", "username"),
        Index("ix_users_email", "email"),
        Index("ix_users_created_at", "created_at"),
    )


class Role(Base):
    """Role model for role-based access control."""
    
    __tablename__ = "roles"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    permissions: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    users: Mapped[List["UserRole"]] = relationship(
        "UserRole", 
        back_populates="role"
    )


class UserRole(Base):
    """Many-to-many relationship between users and roles."""
    
    __tablename__ = "user_roles"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE")
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("roles.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="roles")
    role: Mapped["Role"] = relationship("Role", back_populates="users")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="uq_user_role"),
        Index("ix_user_roles_user_id", "user_id"),
        Index("ix_user_roles_role_id", "role_id"),
    )


class APIKey(Base):
    """API key model for programmatic access."""
    
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index("ix_api_keys_key_hash", "key_hash"),
        Index("ix_api_keys_user_id", "user_id"),
        Index("ix_api_keys_expires_at", "expires_at"),
    )


class Stack(Base):
    """Technology stack model."""
    
    __tablename__ = "stacks"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    technologies: Mapped[Optional[List[str]]] = mapped_column(JSON)
    quality_standards: Mapped[Optional[dict]] = mapped_column(JSON)
    requirements: Mapped[Optional[dict]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    # Relationships
    templates: Mapped[List["Template"]] = relationship(
        "Template", 
        back_populates="stack"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_stacks_name", "name"),
        Index("ix_stacks_is_active", "is_active"),
    )


class Template(Base):
    """Template model for project templates."""
    
    __tablename__ = "templates"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    stack_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("stacks.id", ondelete="CASCADE")
    )
    description: Mapped[Optional[str]] = mapped_column(Text)
    path: Mapped[Optional[str]] = mapped_column(String(500))
    upstream_url: Mapped[Optional[str]] = mapped_column(String(500))
    version: Mapped[str] = mapped_column(String(20), default="0.1.0")
    author: Mapped[Optional[str]] = mapped_column(String(100))
    license: Mapped[Optional[str]] = mapped_column(String(50))
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    technologies: Mapped[Optional[List[str]]] = mapped_column(JSON)
    features: Mapped[Optional[List[str]]] = mapped_column(JSON)
    dependencies: Mapped[Optional[dict]] = mapped_column(JSON)
    min_python_version: Mapped[Optional[str]] = mapped_column(String(10))
    min_node_version: Mapped[Optional[str]] = mapped_column(String(10))
    stars: Mapped[int] = mapped_column(Integer, default=0)
    forks: Mapped[int] = mapped_column(Integer, default=0)
    growth_rate: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    last_synced: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    stack: Mapped["Stack"] = relationship("Stack", back_populates="templates")
    downloads: Mapped[List["TemplateDownload"]] = relationship(
        "TemplateDownload", 
        back_populates="template",
        cascade="all, delete-orphan"
    )
    
    # Indexes and constraints
    __table_args__ = (
        UniqueConstraint("name", "stack_id", name="uq_template_name_stack"),
        CheckConstraint("quality_score >= 0.0 AND quality_score <= 1.0", name="ck_template_quality_score"),
        Index("ix_templates_name", "name"),
        Index("ix_templates_stack_id", "stack_id"),
        Index("ix_templates_quality_score", "quality_score"),
        Index("ix_templates_stars", "stars"),
        Index("ix_templates_is_active", "is_active"),
        Index("ix_templates_created_at", "created_at"),
        Index("ix_templates_updated_at", "updated_at"),
    )


class TemplateDownload(Base):
    """Template download tracking model."""
    
    __tablename__ = "template_downloads"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    template_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("templates.id", ondelete="CASCADE")
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL")
    )
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    format: Mapped[str] = mapped_column(String(10), default="zip")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    template: Mapped["Template"] = relationship("Template", back_populates="downloads")
    user: Mapped[Optional["User"]] = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("ix_template_downloads_template_id", "template_id"),
        Index("ix_template_downloads_user_id", "user_id"),
        Index("ix_template_downloads_created_at", "created_at"),
    )


class SearchQuery(Base):
    """Search query tracking model."""
    
    __tablename__ = "search_queries"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    query: Mapped[str] = mapped_column(String(200), nullable=False)
    stack: Mapped[Optional[str]] = mapped_column(String(50))
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL")
    )
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    results_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index("ix_search_queries_query", "query"),
        Index("ix_search_queries_stack", "stack"),
        Index("ix_search_queries_created_at", "created_at"),
    )
