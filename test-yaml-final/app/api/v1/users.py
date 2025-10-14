"""
User management API endpoints.

This module provides user management endpoints including
user CRUD operations and profile management.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from pydantic import BaseModel, EmailStr, Field

from app.core.database import get_db
from app.core.models import User
from app.core.security import get_password_hash, validate_password_strength
from app.core.exceptions import ValidationError, NotFoundError, ConflictError
from app.api.dependencies import (
    get_current_user, get_current_superuser, get_user_by_id,
    get_pagination_params, require_permissions
)
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Pydantic models
class UserResponse(BaseModel):
    """User response model."""
    id: int
    email: str
    full_name: str
    is_active: bool
    is_verified: bool
    is_superuser: bool
    created_at: str
    last_login: Optional[str] = None
    
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)
    is_active: bool = True
    is_superuser: bool = False


class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_superuser: Optional[bool] = None


class UserProfileUpdate(BaseModel):
    """User profile update model."""
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)


class PasswordUpdate(BaseModel):
    """Password update model."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class UserListResponse(BaseModel):
    """User list response model."""
    users: List[UserResponse]
    total: int
    page: int
    size: int
    pages: int


# User management endpoints
@router.get("/me", response_model=UserResponse, summary="Get current user profile")
async def get_my_profile(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """
    Get current user's profile.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User profile information
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse, summary="Update current user profile")
async def update_my_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Update current user's profile.
    
    Args:
        profile_data: Profile update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated user profile
        
    Raises:
        ValidationError: If validation fails
    """
    update_data = profile_data.dict(exclude_unset=True)
    
    if not update_data:
        return UserResponse.from_orm(current_user)
    
    # Update user
    await db.execute(
        update(User)
        .where(User.id == current_user.id)
        .values(**update_data)
    )
    await db.commit()
    
    # Refresh user from database
    await db.refresh(current_user)
    
    logger.info(f"User {current_user.email} updated their profile")
    
    return UserResponse.from_orm(current_user)


@router.delete("/me", summary="Delete current user account")
async def delete_my_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete current user's account.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
    """
    # Soft delete by deactivating the user
    await db.execute(
        update(User)
        .where(User.id == current_user.id)
        .values(is_active=False)
    )
    await db.commit()
    
    logger.info(f"User {current_user.email} deleted their account")
    
    return {"message": "Account deleted successfully"}


# Admin endpoints
@router.get("/", response_model=UserListResponse, summary="List all users (Admin)")
async def list_users(
    pagination: Dict[str, int] = Depends(get_pagination_params),
    search: Optional[str] = Query(None, description="Search by email or name"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    is_verified: Optional[bool] = Query(None, description="Filter by verification status"),
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> UserListResponse:
    """
    List all users with pagination and filtering (Admin only).
    
    Args:
        pagination: Pagination parameters
        search: Search query
        is_active: Filter by active status
        is_verified: Filter by verification status
        current_user: Current superuser
        db: Database session
        
    Returns:
        Paginated list of users
    """
    # Build query
    query = select(User)
    
    # Apply filters
    if search:
        query = query.where(
            (User.email.ilike(f"%{search}%")) |
            (User.full_name.ilike(f"%{search}%"))
        )
    
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    
    if is_verified is not None:
        query = query.where(User.is_verified == is_verified)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.offset(pagination["offset"]).limit(pagination["size"])
    query = query.order_by(User.created_at.desc())
    
    # Execute query
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Calculate pages
    pages = (total + pagination["size"] - 1) // pagination["size"]
    
    return UserListResponse(
        users=[UserResponse.from_orm(user) for user in users],
        total=total,
        page=pagination["page"],
        size=pagination["size"],
        pages=pages
    )


@router.get("/{user_id}", response_model=UserResponse, summary="Get user by ID (Admin)")
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Get user by ID (Admin only).
    
    Args:
        user_id: User ID
        current_user: Current superuser
        db: Database session
        
    Returns:
        User information
        
    Raises:
        NotFoundError: If user is not found
    """
    user = await get_user_by_id(user_id, db)
    return UserResponse.from_orm(user)


@router.post("/", response_model=UserResponse, summary="Create new user (Admin)")
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Create a new user (Admin only).
    
    Args:
        user_data: User creation data
        current_user: Current superuser
        db: Database session
        
    Returns:
        Created user information
        
    Raises:
        ValidationError: If validation fails
        ConflictError: If user already exists
    """
    # Validate password strength
    is_valid, errors = validate_password_strength(user_data.password)
    if not is_valid:
        raise ValidationError(f"Password validation failed: {', '.join(errors)}")
    
    # Check if user already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise ConflictError("User with this email already exists")
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        is_active=user_data.is_active,
        is_superuser=user_data.is_superuser,
        is_verified=False
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"Admin {current_user.email} created user {new_user.email}")
    
    return UserResponse.from_orm(new_user)


@router.put("/{user_id}", response_model=UserResponse, summary="Update user (Admin)")
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    Update user (Admin only).
    
    Args:
        user_id: User ID
        user_data: User update data
        current_user: Current superuser
        db: Database session
        
    Returns:
        Updated user information
        
    Raises:
        NotFoundError: If user is not found
        ConflictError: If email already exists
    """
    # Get user
    user = await get_user_by_id(user_id, db)
    
    # Check if email is being changed and if it already exists
    if user_data.email and user_data.email != user.email:
        result = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise ConflictError("User with this email already exists")
    
    # Update user
    update_data = user_data.dict(exclude_unset=True)
    
    if update_data:
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(**update_data)
        )
        await db.commit()
        await db.refresh(user)
    
    logger.info(f"Admin {current_user.email} updated user {user.email}")
    
    return UserResponse.from_orm(user)


@router.delete("/{user_id}", summary="Delete user (Admin)")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete user (Admin only).
    
    Args:
        user_id: User ID
        current_user: Current superuser
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        NotFoundError: If user is not found
        ValidationError: If trying to delete self
    """
    # Prevent self-deletion
    if user_id == current_user.id:
        raise ValidationError("Cannot delete your own account")
    
    # Get user
    user = await get_user_by_id(user_id, db)
    
    # Soft delete by deactivating
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(is_active=False)
    )
    await db.commit()
    
    logger.info(f"Admin {current_user.email} deleted user {user.email}")
    
    return {"message": "User deleted successfully"}


@router.post("/{user_id}/activate", summary="Activate user (Admin)")
async def activate_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Activate user account (Admin only).
    
    Args:
        user_id: User ID
        current_user: Current superuser
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        NotFoundError: If user is not found
    """
    user = await get_user_by_id(user_id, db)
    
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(is_active=True)
    )
    await db.commit()
    
    logger.info(f"Admin {current_user.email} activated user {user.email}")
    
    return {"message": "User activated successfully"}


@router.post("/{user_id}/deactivate", summary="Deactivate user (Admin)")
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Deactivate user account (Admin only).
    
    Args:
        user_id: User ID
        current_user: Current superuser
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        NotFoundError: If user is not found
        ValidationError: If trying to deactivate self
    """
    # Prevent self-deactivation
    if user_id == current_user.id:
        raise ValidationError("Cannot deactivate your own account")
    
    user = await get_user_by_id(user_id, db)
    
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(is_active=False)
    )
    await db.commit()
    
    logger.info(f"Admin {current_user.email} deactivated user {user.email}")
    
    return {"message": "User deactivated successfully"}
