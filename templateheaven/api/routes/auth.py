"""
Authentication routes for Template Heaven API.

This module provides authentication endpoints including login,
token refresh, and user management.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Form, Path
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

from ...core.models import APIResponse
from ...database.models import User, UserRole
from ...services.auth_service import auth_service
from ..dependencies import (
    get_settings, get_current_user, require_auth, get_request_id
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request_id: str = Depends(get_request_id)
):
    """
    Authenticate user and return access token.
    
    Authenticates a user with username/password and returns a JWT
    access token for API authentication.
    """
    try:
        # Use the imported auth service
        
        # Authenticate user
        user = await auth_service.authenticate_user(
            username=form_data.username,
            password=form_data.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        token_data = {
            "sub": user.username,
            "user_id": str(user.id),
            "roles": [role.role.name for role in user.roles]
        }
        access_token = auth_service.create_access_token(token_data)
        
        logger.info(f"User {user.username} logged in successfully")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": get_settings().access_token_expire_minutes * 60,
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "roles": [role.role.name for role in user.roles]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/auth/refresh")
async def refresh_token(
    current_user: User = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Refresh access token.
    
    Creates a new access token for the authenticated user,
    extending their session.
    """
    try:
        # Use the imported auth service
        
        # Create new access token
        access_token = await auth_service.create_access_token(
            data={"sub": current_user.username, "user_id": current_user.id}
        )
        
        logger.info(f"Token refreshed for user {current_user.username}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": get_settings().access_token_expire_minutes * 60
        }
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.get("/auth/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Get current user information.
    
    Returns detailed information about the currently authenticated user.
    """
    return current_user


@router.post("/auth/logout")
async def logout(
    current_user: User = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Logout user.
    
    Invalidates the current user's session and logs the logout event.
    """
    try:
        # Use the imported auth service
        
        # Log logout event
        await auth_service.logout_user(current_user.id)
        
        logger.info(f"User {current_user.username} logged out")
        
        return APIResponse(
            success=True,
            message="Successfully logged out"
        )
        
    except Exception as e:
        logger.error(f"Logout failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/auth/register")
async def register_user(
    username: str = Form(..., min_length=3, max_length=50),
    email: str = Form(...),
    password: str = Form(..., min_length=8),
    full_name: Optional[str] = Form(None),
    request_id: str = Depends(get_request_id)
):
    """
    Register a new user.
    
    Creates a new user account with the provided information.
    """
    try:
        # Use the imported auth service
        
        # Check if user already exists
        existing_user = await auth_service.get_user_by_username(username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        existing_email = await auth_service.get_user_by_email(email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = await auth_service.create_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
            roles=["user"]  # Default role
        )
        
        logger.info(f"New user registered: {username}")
        
        return APIResponse(
            success=True,
            message="User registered successfully",
            data={
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email,
                "roles": [role.role.name for role in user.roles]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration failed"
        )


@router.post("/auth/change-password")
async def change_password(
    current_password: str = Form(...),
    new_password: str = Form(..., min_length=8),
    current_user: User = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Change user password.
    
    Changes the password for the currently authenticated user.
    """
    try:
        # Use the imported auth service
        
        # Verify current password
        if not await auth_service.verify_password(current_password, current_user.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        await auth_service.update_password(current_user.id, new_password)
        
        logger.info(f"Password changed for user {current_user.username}")
        
        return APIResponse(
            success=True,
            message="Password changed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/auth/reset-password")
async def request_password_reset(
    email: str = Form(...),
    request_id: str = Depends(get_request_id)
):
    """
    Request password reset.
    
    Initiates a password reset process for the user with the given email.
    """
    try:
        # Use the imported auth service
        
        # Check if user exists
        user = await auth_service.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists or not
            return APIResponse(
                success=True,
                message="If the email exists, a password reset link has been sent"
            )
        
        # Generate reset token
        reset_token = await auth_service.create_password_reset_token(user.id)
        
        # Send reset email (in a real implementation)
        # await email_service.send_password_reset_email(user.email, reset_token)
        
        logger.info(f"Password reset requested for user {user.username}")
        
        return APIResponse(
            success=True,
            message="If the email exists, a password reset link has been sent"
        )
        
    except Exception as e:
        logger.error(f"Password reset request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )


@router.post("/auth/reset-password/confirm")
async def confirm_password_reset(
    token: str = Form(...),
    new_password: str = Form(..., min_length=8),
    request_id: str = Depends(get_request_id)
):
    """
    Confirm password reset.
    
    Completes the password reset process using the provided token.
    """
    try:
        # Use the imported auth service
        
        # Verify reset token
        user_id = await auth_service.verify_password_reset_token(token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Update password
        await auth_service.update_password(user_id, new_password)
        
        # Invalidate reset token
        await auth_service.invalidate_password_reset_token(token)
        
        logger.info(f"Password reset completed for user {user_id}")
        
        return APIResponse(
            success=True,
            message="Password reset successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset confirmation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset confirmation failed"
        )


@router.get("/auth/sessions")
async def get_user_sessions(
    current_user: User = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Get user sessions.
    
    Returns information about the current user's active sessions.
    """
    try:
        # Use the imported auth service
        
        # Get user sessions
        sessions = await auth_service.get_user_sessions(current_user.id)
        
        return APIResponse(
            success=True,
            message="User sessions retrieved",
            data={"sessions": sessions}
        )
        
    except Exception as e:
        logger.error(f"Failed to get user sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user sessions"
        )


@router.delete("/auth/sessions/{session_id}")
async def revoke_session(
    session_id: str = Path(..., description="Session ID"),
    current_user: User = Depends(require_auth),
    request_id: str = Depends(get_request_id)
):
    """
    Revoke a user session.
    
    Revokes a specific session for the current user.
    """
    try:
        # Use the imported auth service
        
        # Revoke session
        success = await auth_service.revoke_session(session_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        logger.info(f"Session {session_id} revoked for user {current_user.username}")
        
        return APIResponse(
            success=True,
            message="Session revoked successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke session"
        )
