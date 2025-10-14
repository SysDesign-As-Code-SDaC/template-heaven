"""
Authentication API endpoints.

This module provides authentication-related endpoints including
login, logout, token refresh, and password management.
"""

from datetime import timedelta
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field

from app.core.database import get_db
from app.core.models import User
from app.core.security import (
    verify_password, get_password_hash, create_access_token, create_refresh_token,
    verify_token, generate_password_reset_token, verify_password_reset_token,
    generate_email_verification_token, verify_email_verification_token,
    validate_password_strength
)
from app.core.config import get_settings
from app.core.exceptions import AuthenticationError, ValidationError, NotFoundError
from app.api.dependencies import get_current_user, get_optional_current_user
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()


# Pydantic models
class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data model."""
    access_token: str
    token_type: str = "bearer"


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str = Field(..., min_length=8)


class UserRegister(BaseModel):
    """User registration model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)


class PasswordReset(BaseModel):
    """Password reset model."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""
    token: str
    new_password: str = Field(..., min_length=8)


class PasswordChange(BaseModel):
    """Password change model."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class EmailVerification(BaseModel):
    """Email verification model."""
    token: str


# Authentication endpoints
@router.post("/login", response_model=Token, summary="User login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> Token:
    """
    Authenticate user and return access and refresh tokens.
    
    Args:
        form_data: OAuth2 password form data
        db: Database session
        
    Returns:
        Token response with access and refresh tokens
        
    Raises:
        AuthenticationError: If authentication fails
    """
    # Get user by email
    from sqlalchemy import select
    result = await db.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise AuthenticationError("Invalid email or password")
    
    if not user.is_active:
        raise AuthenticationError("User account is disabled")
    
    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        raise AuthenticationError("Invalid email or password")
    
    # Update last login
    from sqlalchemy import update
    await db.execute(
        update(User)
        .where(User.id == user.id)
        .values(last_login=db.execute("SELECT NOW()").scalar())
    )
    await db.commit()
    
    # Create tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user.email, "user_id": user.id},
        expires_delta=refresh_token_expires
    )
    
    logger.info(f"User {user.email} logged in successfully")
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/register", response_model=Dict[str, str], summary="User registration")
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        ValidationError: If validation fails
        ConflictError: If user already exists
    """
    # Validate password strength
    is_valid, errors = validate_password_strength(user_data.password)
    if not is_valid:
        raise ValidationError(f"Password validation failed: {', '.join(errors)}")
    
    # Check if user already exists
    from sqlalchemy import select
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=True,
        is_verified=False
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Generate email verification token
    verification_token = generate_email_verification_token(user_data.email)
    
    # In a real application, you would send an email here
    logger.info(f"User {user_data.email} registered successfully")
    
    return {
        "message": "User registered successfully",
        "verification_token": verification_token  # Only for development
    }


@router.post("/refresh", response_model=TokenData, summary="Refresh access token")
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
) -> TokenData:
    """
    Refresh access token using refresh token.
    
    Args:
        refresh_token: Refresh token
        db: Database session
        
    Returns:
        New access token
        
    Raises:
        AuthenticationError: If refresh token is invalid
    """
    try:
        # Verify refresh token
        payload = verify_token(refresh_token, token_type="refresh")
        user_email = payload.get("sub")
        
        if not user_email:
            raise AuthenticationError("Invalid refresh token")
        
        # Get user
        from sqlalchemy import select
        result = await db.execute(
            select(User).where(User.email == user_email)
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id},
            expires_delta=access_token_expires
        )
        
        return TokenData(access_token=access_token)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise AuthenticationError("Token refresh failed")


@router.post("/logout", summary="User logout")
async def logout(
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Logout user (invalidate tokens on client side).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    # In a real application, you would invalidate the token
    # by adding it to a blacklist or updating the user's session
    
    logger.info(f"User {current_user.email} logged out")
    
    return {"message": "Logged out successfully"}


@router.post("/password-reset", summary="Request password reset")
async def request_password_reset(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Request password reset token.
    
    Args:
        reset_data: Password reset request data
        db: Database session
        
    Returns:
        Success message
    """
    # Check if user exists
    from sqlalchemy import select
    result = await db.execute(
        select(User).where(User.email == reset_data.email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        # Don't reveal if user exists or not
        return {"message": "If the email exists, a password reset link has been sent"}
    
    # Generate password reset token
    reset_token = generate_password_reset_token(reset_data.email)
    
    # In a real application, you would send an email here
    logger.info(f"Password reset requested for {reset_data.email}")
    
    return {
        "message": "If the email exists, a password reset link has been sent",
        "reset_token": reset_token  # Only for development
    }


@router.post("/password-reset/confirm", summary="Confirm password reset")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Confirm password reset with token.
    
    Args:
        reset_data: Password reset confirmation data
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        ValidationError: If validation fails
        AuthenticationError: If token is invalid
    """
    # Validate new password strength
    is_valid, errors = validate_password_strength(reset_data.new_password)
    if not is_valid:
        raise ValidationError(f"Password validation failed: {', '.join(errors)}")
    
    # Verify reset token
    try:
        email = verify_password_reset_token(reset_data.token)
    except HTTPException:
        raise AuthenticationError("Invalid or expired reset token")
    
    # Get user
    from sqlalchemy import select, update
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise NotFoundError("User not found")
    
    # Update password
    hashed_password = get_password_hash(reset_data.new_password)
    await db.execute(
        update(User)
        .where(User.id == user.id)
        .values(hashed_password=hashed_password)
    )
    await db.commit()
    
    logger.info(f"Password reset completed for {email}")
    
    return {"message": "Password reset successfully"}


@router.post("/change-password", summary="Change password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Change user password.
    
    Args:
        password_data: Password change data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        ValidationError: If validation fails
        AuthenticationError: If current password is incorrect
    """
    # Validate new password strength
    is_valid, errors = validate_password_strength(password_data.new_password)
    if not is_valid:
        raise ValidationError(f"Password validation failed: {', '.join(errors)}")
    
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise AuthenticationError("Current password is incorrect")
    
    # Update password
    from sqlalchemy import update
    hashed_password = get_password_hash(password_data.new_password)
    await db.execute(
        update(User)
        .where(User.id == current_user.id)
        .values(hashed_password=hashed_password)
    )
    await db.commit()
    
    logger.info(f"Password changed for user {current_user.email}")
    
    return {"message": "Password changed successfully"}


@router.post("/verify-email", summary="Verify email address")
async def verify_email(
    verification_data: EmailVerification,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Verify email address with token.
    
    Args:
        verification_data: Email verification data
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        AuthenticationError: If token is invalid
        NotFoundError: If user is not found
    """
    # Verify email verification token
    try:
        email = verify_email_verification_token(verification_data.token)
    except HTTPException:
        raise AuthenticationError("Invalid or expired verification token")
    
    # Get user
    from sqlalchemy import select, update
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise NotFoundError("User not found")
    
    # Update user verification status
    await db.execute(
        update(User)
        .where(User.id == user.id)
        .values(is_verified=True)
    )
    await db.commit()
    
    logger.info(f"Email verified for user {email}")
    
    return {"message": "Email verified successfully"}


@router.get("/me", summary="Get current user")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "is_superuser": current_user.is_superuser,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }
