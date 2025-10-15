"""
Authentication API Routes

JWT-based authentication endpoints for login, logout, token refresh, and password management.
"""

import logging
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.security import (
    verify_token,
    verify_refresh_token,
    create_token_pair,
    validate_password_strength
)
from ...services.user_service import UserService
from ...schemas.user import (
    User,
    UserCreate,
    Token,
    LoginRequest,
    RefreshTokenRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    ChangePasswordRequest,
    EmailVerificationRequest
)

logger = logging.getLogger(__name__)

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user_id = verify_token(token)
    if user_id is None:
        raise credentials_exception

    user_service = UserService(db)
    user = await user_service.get_user_by_id(int(user_id))

    if user is None or not user.is_active:
        raise credentials_exception

    return User.from_orm(user)


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_user)) -> User:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Register a new user account.

    Creates a new user with the provided information and sends an email verification.
    """
    user_service = UserService(db)

    try:
        user = await user_service.create_user(user_data)
        logger.info(f"User registered: {user.username} from {request.client.host}")

        return User.from_orm(user)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    Authenticate user and return access/refresh tokens.

    Supports login with username or email.
    """
    user_service = UserService(db)

    login_data = LoginRequest(
        username_or_email=form_data.username,
        password=form_data.password
    )

    user = await user_service.authenticate_user(login_data)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create session
    session_token = await user_service.create_session(
        user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )

    # Create token pair
    token_data = create_token_pair(user.id)

    logger.info(f"User logged in: {user.username} from {request.client.host}")

    return token_data


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Refresh access token using refresh token.

    Returns new access and refresh token pair.
    """
    user_id = verify_refresh_token(refresh_data.refresh_token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    user_service = UserService(db)
    user = await user_service.get_user_by_id(int(user_id))

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Create new token pair
    token_data = create_token_pair(user.id)

    logger.info(f"Token refreshed for user: {user.username}")

    return token_data


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Logout user and invalidate session.

    Requires valid access token.
    """
    # Note: In a production system, you might want to maintain a blacklist
    # of invalidated tokens. For simplicity, we rely on token expiration.

    logger.info(f"User logged out: {current_user.username} from {request.client.host}")

    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)) -> Any:
    """
    Get current user information.

    Returns the current authenticated user's profile information.
    """
    return current_user


@router.put("/me", response_model=User)
async def update_user_me(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update current user's profile information.
    """
    user_service = UserService(db)

    try:
        updated_user = await user_service.update_user(current_user.id, user_data)

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


@router.post("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Change current user's password.

    Requires current password verification.
    """
    user_service = UserService(db)

    try:
        success = await user_service.change_password(current_user.id, password_data)

        if success:
            logger.info(f"Password changed for user: {current_user.username}")
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/password-reset-request")
async def request_password_reset(
    reset_data: PasswordResetRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Request password reset.

    Sends password reset email if user exists (doesn't reveal if email exists).
    """
    user_service = UserService(db)

    try:
        success = await user_service.reset_password_request(reset_data)

        # Always return success for security (don't reveal if email exists)
        return {"message": "If the email exists, a password reset link has been sent"}

    except Exception as e:
        logger.error(f"Password reset request failed: {e}")
        # Still return success for security
        return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/password-reset-confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Confirm password reset with token.

    Validates token and updates password.
    """
    user_service = UserService(db)

    try:
        success = await user_service.reset_password_confirm(reset_data)

        if success:
            return {"message": "Password reset successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password reset failed"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/verify-email")
async def verify_email(
    verification_data: EmailVerificationRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Verify user email address with token.
    """
    user_service = UserService(db)

    try:
        success = await user_service.verify_email(verification_data.token)

        if success:
            return {"message": "Email verified successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email verification failed"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/resend-verification")
async def resend_verification_email(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Resend email verification token.

    Only works if email is not already verified.
    """
    if current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )

    user_service = UserService(db)

    try:
        token = await user_service.create_email_verification_token(current_user.id)

        # TODO: Send email with verification token
        logger.info(f"Verification email resent for user: {current_user.username}")

        return {"message": "Verification email sent"}

    except Exception as e:
        logger.error(f"Failed to resend verification email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email"
        )


# Password validation endpoint
@router.post("/validate-password")
async def validate_password(password: str) -> Any:
    """
    Validate password strength.

    Returns validation result without creating account.
    """
    is_valid, error_message = validate_password_strength(password)

    return {
        "valid": is_valid,
        "message": error_message if not is_valid else "Password is strong"
    }
