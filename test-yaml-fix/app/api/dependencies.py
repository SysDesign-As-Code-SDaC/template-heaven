"""
API dependencies for authentication and authorization.

This module provides FastAPI dependencies for user authentication,
authorization, and common API functionality.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.models import User, APIKey
from app.core.security import verify_token, verify_api_key
from app.core.exceptions import AuthenticationError, AuthorizationError, NotFoundError
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get the current authenticated user.
    
    Args:
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        Current user
        
    Raises:
        AuthenticationError: If authentication fails
    """
    if not credentials:
        raise AuthenticationError("Authentication required")
    
    try:
        # Verify the token
        payload = verify_token(credentials.credentials)
        user_email = payload.get("sub")
        
        if not user_email:
            raise AuthenticationError("Invalid token payload")
        
        # Get user from database
        result = await db.execute(
            select(User).where(User.email == user_email)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("User account is disabled")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise AuthenticationError("Authentication failed")


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get the current active user.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        Current active user
        
    Raises:
        AuthenticationError: If user is not active
    """
    if not current_user.is_active:
        raise AuthenticationError("User account is disabled")
    
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get the current superuser.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        Current superuser
        
    Raises:
        AuthorizationError: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise AuthorizationError("Superuser access required")
    
    return current_user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get the current user if authenticated, otherwise return None.
    
    Args:
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        Current user or None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except (AuthenticationError, HTTPException):
        return None


async def get_user_by_id(
    user_id: int,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get a user by ID.
    
    Args:
        user_id: User ID
        db: Database session
        
    Returns:
        User
        
    Raises:
        NotFoundError: If user is not found
    """
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise NotFoundError(f"User with ID {user_id} not found")
    
    return user


async def verify_api_key_auth(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Verify API key authentication.
    
    Args:
        request: FastAPI request object
        db: Database session
        
    Returns:
        User associated with the API key
        
    Raises:
        AuthenticationError: If API key is invalid
    """
    # Get API key from header
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise AuthenticationError("API key required")
    
    # Get API key prefix (first 8 characters)
    key_prefix = api_key[:8]
    
    # Find API key in database
    result = await db.execute(
        select(APIKey).where(
            APIKey.key_prefix == key_prefix,
            APIKey.is_active == True
        )
    )
    api_key_record = result.scalar_one_or_none()
    
    if not api_key_record:
        raise AuthenticationError("Invalid API key")
    
    # Verify the API key
    if not verify_api_key(api_key, api_key_record.key_hash):
        raise AuthenticationError("Invalid API key")
    
    # Get the user
    result = await db.execute(
        select(User).where(User.id == api_key_record.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise AuthenticationError("User not found or inactive")
    
    # Update API key usage
    api_key_record.usage_count += 1
    api_key_record.last_used = db.execute("SELECT NOW()").scalar()
    await db.commit()
    
    return user


async def get_pagination_params(
    page: int = 1,
    size: int = 20,
    max_size: int = 100
) -> Dict[str, int]:
    """
    Get pagination parameters.
    
    Args:
        page: Page number (1-based)
        size: Page size
        max_size: Maximum allowed page size
        
    Returns:
        Dictionary with pagination parameters
        
    Raises:
        HTTPException: If parameters are invalid
    """
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page number must be greater than 0"
        )
    
    if size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be greater than 0"
        )
    
    if size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Page size cannot exceed {max_size}"
        )
    
    return {
        "page": page,
        "size": size,
        "offset": (page - 1) * size
    }


async def get_rate_limit_info(request: Request) -> Dict[str, Any]:
    """
    Get rate limit information from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dictionary with rate limit information
    """
    # Get client IP
    client_ip = request.client.host
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    
    # Get user agent
    user_agent = request.headers.get("user-agent", "")
    
    return {
        "client_ip": client_ip,
        "user_agent": user_agent,
        "endpoint": request.url.path,
        "method": request.method
    }


async def log_request(
    request: Request,
    current_user: Optional[User] = Depends(get_optional_current_user)
) -> None:
    """
    Log API request for audit purposes.
    
    Args:
        request: FastAPI request object
        current_user: Current user (if authenticated)
    """
    # This would typically log to an audit system
    # For now, we'll just log to the application logger
    logger.info(
        f"API Request: {request.method} {request.url.path} - "
        f"User: {current_user.email if current_user else 'Anonymous'} - "
        f"IP: {request.client.host}"
    )


def require_permissions(permissions: list[str]):
    """
    Decorator to require specific permissions.
    
    Args:
        permissions: List of required permissions
        
    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        # This would check user permissions
        # For now, we'll just check if user is active
        if not current_user.is_active:
            raise AuthorizationError("User account is disabled")
        
        # In a real implementation, you would check the user's permissions
        # against the required permissions list
        
        return current_user
    
    return permission_checker


def require_roles(roles: list[str]):
    """
    Decorator to require specific roles.
    
    Args:
        roles: List of required roles
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        # This would check user roles
        # For now, we'll just check if user is a superuser for admin roles
        if "admin" in roles and not current_user.is_superuser:
            raise AuthorizationError("Admin role required")
        
        return current_user
    
    return role_checker
