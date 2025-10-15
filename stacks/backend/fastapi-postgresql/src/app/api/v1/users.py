"""
Users API Routes

User management endpoints for admins and user profile operations.
"""

import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...services.user_service import UserService
from ...schemas.user import (
    User,
    UserPublic,
    UserUpdate,
    UserAdminUpdate,
    UserAdminCreate,
    UsersList,
    UserStats,
    ProfileUpdate
)
from .auth import get_current_active_user, get_current_superuser

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/me", response_model=User)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get current user's full profile information.

    Returns detailed user information including private fields.
    """
    return current_user


@router.put("/me", response_model=User)
async def update_current_user_profile(
    user_data: ProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update current user's profile information.

    Only allows updating profile-related fields.
    """
    user_service = UserService(db)

    # Convert ProfileUpdate to UserUpdate
    update_data = UserUpdate(**user_data.model_dump())

    try:
        updated_user = await user_service.update_user(current_user.id, update_data)

        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        logger.info(f"User profile updated: {current_user.username}")

        return User.from_orm(updated_user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/stats", response_model=UserStats)
async def get_user_statistics(
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get user statistics and analytics.

    Admin endpoint for user management insights.
    """
    user_service = UserService(db)

    try:
        stats = await user_service.get_user_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get user statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.get("/", response_model=UsersList)
async def get_users(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    search: str = Query(None, description="Search query for username, email, or name"),
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get paginated list of users.

    Admin endpoint for user management.
    Supports search and pagination.
    """
    user_service = UserService(db)

    try:
        result = await user_service.get_users_list(
            page=page,
            per_page=per_page,
            search=search
        )

        return UsersList(
            users=[User.from_orm(user) for user in result["users"]],
            total=result["total"],
            page=result["page"],
            per_page=result["per_page"],
            pages=result["pages"]
        )

    except Exception as e:
        logger.error(f"Failed to get users list: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user_admin(
    user_data: UserAdminCreate,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Create a new user account (admin only).

    Admin endpoint to create users with full control over account settings.
    """
    user_service = UserService(db)

    try:
        user = await user_service.create_user(user_data)
        logger.info(f"Admin created user: {user.username} by {current_user.username}")

        return User.from_orm(user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Admin user creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed"
        )


@router.get("/{user_id}", response_model=User)
async def get_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get user by ID.

    Admin endpoint to retrieve detailed user information.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return User.from_orm(user)


@router.put("/{user_id}", response_model=User)
async def update_user_admin(
    user_id: int,
    user_data: UserAdminUpdate,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update user information (admin only).

    Admin endpoint with full control over user account settings.
    """
    user_service = UserService(db)

    try:
        updated_user = await user_service.admin_update_user(user_id, user_data)

        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        logger.info(f"Admin updated user: {updated_user.username} by {current_user.username}")

        return User.from_orm(updated_user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{user_id}")
async def delete_user_admin(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Delete/deactivate user account (admin only).

    Admin endpoint to deactivate user accounts.
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    user_service = UserService(db)

    try:
        success = await user_service.delete_user(user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        logger.info(f"Admin deleted user ID: {user_id} by {current_user.username}")

        return {"message": "User account deactivated"}

    except Exception as e:
        logger.error(f"Admin user deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deletion failed"
        )


@router.post("/bulk-deactivate")
async def bulk_deactivate_users(
    user_ids: List[int],
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Bulk deactivate multiple user accounts.

    Admin endpoint for bulk user management operations.
    """
    if current_user.id in user_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )

    user_service = UserService(db)

    try:
        deactivated_count = await user_service.bulk_deactivate_users(user_ids)

        logger.info(f"Admin bulk deactivated {deactivated_count} users by {current_user.username}")

        return {
            "message": f"Successfully deactivated {deactivated_count} user accounts",
            "deactivated_count": deactivated_count
        }

    except Exception as e:
        logger.error(f"Bulk deactivation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk deactivation failed"
        )


@router.post("/cleanup-tokens")
async def cleanup_expired_tokens(
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Clean up expired tokens and sessions.

    Admin maintenance endpoint for database cleanup.
    """
    user_service = UserService(db)

    try:
        cleanup_result = await user_service.cleanup_expired_tokens()

        logger.info(f"Admin cleanup completed: {cleanup_result}")

        return {
            "message": "Token cleanup completed",
            "cleaned_up": cleanup_result
        }

    except Exception as e:
        logger.error(f"Token cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token cleanup failed"
        )


@router.get("/public/{user_id}", response_model=UserPublic)
async def get_user_public_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get public user profile information.

    Public endpoint that returns limited user information.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserPublic.from_orm(user)


@router.get("/public/by-username/{username}", response_model=UserPublic)
async def get_user_public_profile_by_username(
    username: str,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get public user profile by username.

    Public endpoint for finding users by username.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_username(username)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserPublic.from_orm(user)
