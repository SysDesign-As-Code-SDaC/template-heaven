"""
User Pydantic Schemas

Request and response models for user management operations.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator


class UserBase(BaseModel):
    """Base user schema with common fields."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8)

    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        from ..core.security import validate_password_strength
        is_valid, error_msg = validate_password_strength(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v

    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v.lower()


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = None


class UserInDBBase(UserBase):
    """Base schema for user data from database."""
    id: int
    is_active: bool
    is_superuser: bool
    email_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]

    class Config:
        """Pydantic configuration."""
        from_attributes = True


class User(UserInDBBase):
    """Schema for user response (without sensitive data)."""
    pass


class UserInDB(UserInDBBase):
    """Schema for user data with sensitive information."""
    hashed_password: str


class UserPublic(BaseModel):
    """Public user information (no sensitive data)."""
    id: int
    username: str
    full_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    website: Optional[str]
    location: Optional[str]
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        """Pydantic configuration."""
        from_attributes = True


# Authentication schemas
class Token(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload schema."""
    sub: Optional[str] = None
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    type: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request schema."""
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., min_length=1)


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    token: str
    new_password: str = Field(..., min_length=8)

    @validator('new_password')
    def validate_password(cls, v):
        """Validate password strength."""
        from ..core.security import validate_password_strength
        is_valid, error_msg = validate_password_strength(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v


class ChangePasswordRequest(BaseModel):
    """Change password request schema."""
    current_password: str
    new_password: str = Field(..., min_length=8)

    @validator('new_password')
    def validate_password(cls, v):
        """Validate password strength."""
        from ..core.security import validate_password_strength
        is_valid, error_msg = validate_password_strength(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v


class EmailVerificationRequest(BaseModel):
    """Email verification request schema."""
    token: str


# Admin schemas
class UserAdminUpdate(BaseModel):
    """Admin schema for updating user information."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    email_verified: Optional[bool] = None


class UserAdminCreate(UserCreate):
    """Admin schema for creating users."""
    is_active: bool = True
    is_superuser: bool = False
    email_verified: bool = False


class UsersList(BaseModel):
    """Response schema for list of users."""
    users: List[User]
    total: int
    page: int
    per_page: int
    pages: int


# Profile schemas
class ProfileUpdate(BaseModel):
    """Schema for updating user profile."""
    full_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=500)
    website: Optional[str] = Field(None, max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = None


# Statistics schemas
class UserStats(BaseModel):
    """User statistics schema."""
    total_users: int
    active_users: int
    verified_users: int
    superusers: int
    recent_registrations: int  # Last 30 days
    login_activity: int  # Last 24 hours
