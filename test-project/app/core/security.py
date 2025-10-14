"""
Security utilities for authentication and authorization.

This module provides JWT token handling, password hashing,
and security-related utilities.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status
import secrets
import hashlib

from app.core.config import get_settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Get settings
settings = get_settings()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        JWT refresh token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        token_type: Expected token type ("access" or "refresh")
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key and its hash.
    
    Returns:
        Tuple of (api_key, api_key_hash)
    """
    # Generate a secure random API key
    api_key = secrets.token_urlsafe(32)
    
    # Create a hash for storage
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    return api_key, api_key_hash


def verify_api_key(api_key: str, api_key_hash: str) -> bool:
    """
    Verify an API key against its hash.
    
    Args:
        api_key: Plain API key
        api_key_hash: Hashed API key
        
    Returns:
        True if API key matches, False otherwise
    """
    computed_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return secrets.compare_digest(computed_hash, api_key_hash)


def generate_password_reset_token(email: str) -> str:
    """
    Generate a password reset token.
    
    Args:
        email: User's email address
        
    Returns:
        Password reset token
    """
    data = {
        "email": email,
        "purpose": "password_reset",
        "exp": datetime.utcnow() + timedelta(hours=1)  # 1 hour expiration
    }
    
    return jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_password_reset_token(token: str) -> str:
    """
    Verify a password reset token and return the email.
    
    Args:
        token: Password reset token
        
    Returns:
        Email address from the token
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Check purpose
        if payload.get("purpose") != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token purpose"
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token has expired"
            )
        
        return payload.get("email")
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token"
        )


def generate_email_verification_token(email: str) -> str:
    """
    Generate an email verification token.
    
    Args:
        email: User's email address
        
    Returns:
        Email verification token
    """
    data = {
        "email": email,
        "purpose": "email_verification",
        "exp": datetime.utcnow() + timedelta(days=7)  # 7 days expiration
    }
    
    return jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_email_verification_token(token: str) -> str:
    """
    Verify an email verification token and return the email.
    
    Args:
        token: Email verification token
        
    Returns:
        Email address from the token
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Check purpose
        if payload.get("purpose") != "email_verification":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token purpose"
            )
        
        # Check expiration
        exp = payload.get("exp")
        if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token has expired"
            )
        
        return payload.get("email")
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token"
        )


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*(),.?\":{}|<>" for c in password):
        errors.append("Password must contain at least one special character")
    
    return len(errors) == 0, errors


def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent XSS and other attacks.
    
    Args:
        input_string: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not input_string:
        return ""
    
    # Truncate if too long
    if len(input_string) > max_length:
        input_string = input_string[:max_length]
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\r', '\n']
    for char in dangerous_chars:
        input_string = input_string.replace(char, '')
    
    return input_string.strip()


def generate_csrf_token() -> str:
    """
    Generate a CSRF token.
    
    Returns:
        CSRF token string
    """
    return secrets.token_urlsafe(32)


def verify_csrf_token(token: str, session_token: str) -> bool:
    """
    Verify a CSRF token.
    
    Args:
        token: Token to verify
        session_token: Token from session
        
    Returns:
        True if tokens match, False otherwise
    """
    return secrets.compare_digest(token, session_token)
