"""
Security and Authentication Utilities

Provides JWT token management, password hashing, and authentication utilities.
"""

import secrets
from datetime import datetime, timedelta
from typing import Any, Union, Optional

from jose import jwt
from passlib.context import CryptContext

from .config import get_settings

settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.

    Args:
        subject: Token subject (usually user ID)
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "iat": datetime.utcnow(),
        "type": "access",
    }

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT refresh token.

    Args:
        subject: Token subject (usually user ID)
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "iat": datetime.utcnow(),
        "type": "refresh",
    }

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """
    Verify JWT token and return subject.

    Args:
        token: JWT token to verify

    Returns:
        Token subject if valid, None if invalid
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_type: str = payload.get("type")
        subject: str = payload.get("sub")

        if token_type != "access":
            return None

        return subject
    except jwt.JWTError:
        return None


def verify_refresh_token(token: str) -> Optional[str]:
    """
    Verify JWT refresh token and return subject.

    Args:
        token: JWT refresh token to verify

    Returns:
        Token subject if valid, None if invalid
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_type: str = payload.get("type")
        subject: str = payload.get("sub")

        if token_type != "refresh":
            return None

        return subject
    except jwt.JWTError:
        return None


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches hash
    """
    return pwd_context.verify(plain_password, hashed_password)


def generate_verification_token() -> str:
    """
    Generate secure verification token.

    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(32)


def generate_password_reset_token() -> str:
    """
    Generate secure password reset token.

    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(32)


def generate_session_token() -> str:
    """
    Generate secure session token.

    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(64)


def create_token_pair(user_id: Union[str, Any]) -> dict:
    """
    Create access and refresh token pair.

    Args:
        user_id: User ID for token subject

    Returns:
        Dictionary containing access_token and refresh_token
    """
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token(user_id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


def decode_token(token: str) -> Optional[dict]:
    """
    Decode JWT token without verification (for debugging).

    Args:
        token: JWT token to decode

    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM], options={"verify_exp": False}
        )
        return payload
    except jwt.JWTError:
        return None


def get_token_expiration(token: str) -> Optional[datetime]:
    """
    Get token expiration time.

    Args:
        token: JWT token

    Returns:
        Expiration datetime or None if invalid
    """
    payload = decode_token(token)
    if payload and "exp" in payload:
        return datetime.fromtimestamp(payload["exp"])
    return None


def is_token_expired(token: str) -> bool:
    """
    Check if token is expired.

    Args:
        token: JWT token

    Returns:
        True if token is expired or invalid
    """
    expiration = get_token_expiration(token)
    if expiration:
        return datetime.utcnow() > expiration
    return True


def sanitize_token(token: str) -> str:
    """
    Sanitize token for logging (show only first/last few characters).

    Args:
        token: JWT token

    Returns:
        Sanitized token string
    """
    if len(token) <= 20:
        return token

    return f"{token[:8]}...{token[-8:]}"


# Password strength validation
def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password strength requirements.

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"

    # Check for common weak passwords
    weak_passwords = ["password", "123456", "qwerty", "abc123", "password123"]
    if password.lower() in weak_passwords:
        return False, "Password is too common, please choose a stronger password"

    return True, ""


# CSRF protection utilities
def generate_csrf_token() -> str:
    """
    Generate CSRF protection token.

    Returns:
        Secure random CSRF token
    """
    return secrets.token_hex(32)


def validate_csrf_token(session_token: str, request_token: str) -> bool:
    """
    Validate CSRF token.

    Args:
        session_token: Token stored in session
        request_token: Token from request

    Returns:
        True if tokens match
    """
    if not session_token or not request_token:
        return False

    return secrets.compare_digest(session_token, request_token)
