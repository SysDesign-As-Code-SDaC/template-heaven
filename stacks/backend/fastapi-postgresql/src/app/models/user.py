"""
User Model

SQLAlchemy model for user management with authentication support.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func

from ..core.database import Base


class User(Base):
    """
    User database model.

    Represents a user in the system with authentication and profile information.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Profile fields
    bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    website = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)

    def __repr__(self) -> str:
        """String representation of the user."""
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

    @property
    def is_verified(self) -> bool:
        """Check if user email is verified."""
        return self.email_verified

    @property
    def display_name(self) -> str:
        """Get display name (full name or username)."""
        return self.full_name or self.username

    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()

    def activate(self) -> None:
        """Activate the user account."""
        self.is_active = True
        self.updated_at = datetime.utcnow()

    def deactivate(self) -> None:
        """Deactivate the user account."""
        self.is_active = False
        self.updated_at = datetime.utcnow()

    def verify_email(self) -> None:
        """Mark email as verified."""
        self.email_verified = True
        self.updated_at = datetime.utcnow()

    def make_superuser(self) -> None:
        """Grant superuser privileges."""
        self.is_superuser = True
        self.updated_at = datetime.utcnow()

    def remove_superuser(self) -> None:
        """Remove superuser privileges."""
        self.is_superuser = False
        self.updated_at = datetime.utcnow()


class UserSession(Base):
    """
    User session model for tracking active sessions.
    """

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    session_token = Column(String(500), unique=True, nullable=False)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:
        """String representation of the session."""
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.utcnow() > self.expires_at

    def deactivate(self) -> None:
        """Deactivate the session."""
        self.is_active = False


class PasswordResetToken(Base):
    """
    Password reset token model.
    """

    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    token = Column(String(500), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        """String representation of the token."""
        return f"<PasswordResetToken(id={self.id}, user_id={self.user_id}, used={self.used})>"

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the token is valid (not used and not expired)."""
        return not self.used and not self.is_expired

    def mark_used(self) -> None:
        """Mark the token as used."""
        self.used = True


class EmailVerificationToken(Base):
    """
    Email verification token model.
    """

    __tablename__ = "email_verification_tokens"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    token = Column(String(500), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        """String representation of the token."""
        return f"<EmailVerificationToken(id={self.id}, user_id={self.user_id}, used={self.used})>"

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the token is valid (not used and not expired)."""
        return not self.used and not self.is_expired

    def mark_used(self) -> None:
        """Mark the token as used."""
        self.used = True
