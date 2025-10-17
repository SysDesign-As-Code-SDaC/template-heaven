"""
Authentication and authorization management for MCP SDK Template.

Provides JWT-based authentication, password hashing, and role-based access control
for MCP server and client applications.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import secrets

from .config import SecuritySettings
from .exceptions import MCPAuthenticationError, MCPAuthorizationError, MCPSDKError


logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    GUEST = "guest"


class AuthenticationManager:
    """
    Authentication and authorization management.
    
    Provides JWT token management, password hashing, user authentication,
    and role-based access control.
    """
    
    def __init__(self, settings: SecuritySettings):
        """
        Initialize authentication manager.
        
        Args:
            settings: Security configuration settings
        """
        self.settings = settings
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize authentication manager.
        
        Raises:
            MCPSDKError: If initialization fails
        """
        try:
            logger.info("Initializing authentication manager...")
            
            # Validate secret key
            if self.settings.secret_key == "mcp-sdk-secret-key-change-in-production":
                logger.warning("Using default secret key - change in production!")
            
            # Test password hashing
            test_hash = self.hash_password("test_password")
            if not self.verify_password("test_password", test_hash):
                raise MCPSDKError("Password hashing test failed")
            
            self._initialized = True
            logger.info("Authentication manager initialization complete")
            
        except Exception as e:
            logger.error(f"Authentication manager initialization failed: {e}")
            raise MCPSDKError(f"Authentication initialization failed: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
            
        Returns:
            True if password matches hash
        """
        return self.pwd_context.verify(password, hashed)
    
    def create_access_token(
        self,
        user_id: str,
        username: str,
        roles: List[UserRole],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User identifier
            username: Username
            roles: List of user roles
            expires_delta: Optional token expiration time
            
        Returns:
            Encoded JWT token
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                hours=self.settings.jwt_expiration_hours
            )
        
        to_encode = {
            "sub": user_id,
            "username": username,
            "roles": [role.value for role in roles],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.secret_key,
            algorithm=self.settings.jwt_algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT refresh token.
        
        Args:
            user_id: User identifier
            expires_delta: Optional token expiration time
            
        Returns:
            Encoded JWT refresh token
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=30)  # Refresh tokens last longer
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.secret_key,
            algorithm=self.settings.jwt_algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            MCPAuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.jwt_algorithm]
            )
            
            # Check token type
            if payload.get("type") != "access":
                raise MCPAuthenticationError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                raise MCPAuthenticationError("Token has expired")
            
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise MCPAuthenticationError("Invalid token")
    
    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT refresh token.
        
        Args:
            token: JWT refresh token string
            
        Returns:
            Decoded token payload
            
        Raises:
            MCPAuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.jwt_algorithm]
            )
            
            # Check token type
            if payload.get("type") != "refresh":
                raise MCPAuthenticationError("Invalid refresh token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                raise MCPAuthenticationError("Refresh token has expired")
            
            return payload
            
        except JWTError as e:
            logger.warning(f"Refresh token verification failed: {e}")
            raise MCPAuthenticationError("Invalid refresh token")
    
    def check_permission(
        self,
        user_roles: List[UserRole],
        required_role: UserRole,
        operation: Optional[str] = None
    ) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_roles: List of user roles
            required_role: Required role for operation
            operation: Optional operation name for logging
            
        Returns:
            True if user has permission
            
        Raises:
            MCPAuthorizationError: If user lacks permission
        """
        # Define role hierarchy
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.READONLY: 1,
            UserRole.USER: 2,
            UserRole.ADMIN: 3
        }
        
        # Check if user has any role with sufficient privileges
        user_max_level = max(
            (role_hierarchy.get(role, 0) for role in user_roles),
            default=0
        )
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_max_level >= required_level:
            return True
        
        # Log authorization failure
        logger.warning(
            f"Authorization failed for operation '{operation}': "
            f"user roles {[r.value for r in user_roles]} "
            f"insufficient for required role {required_role.value}"
        )
        
        raise MCPAuthorizationError(
            f"Insufficient permissions for operation '{operation}'",
            operation=operation,
            resource=required_role.value
        )
    
    def generate_password_reset_token(self, user_id: str) -> str:
        """
        Generate password reset token.
        
        Args:
            user_id: User identifier
            
        Returns:
            Password reset token
        """
        expire = datetime.utcnow() + timedelta(hours=1)  # Reset tokens expire in 1 hour
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "password_reset"
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.secret_key,
            algorithm=self.settings.jwt_algorithm
        )
        
        return encoded_jwt
    
    def verify_password_reset_token(self, token: str) -> str:
        """
        Verify password reset token and return user ID.
        
        Args:
            token: Password reset token
            
        Returns:
            User ID from token
            
        Raises:
            MCPAuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.jwt_algorithm]
            )
            
            # Check token type
            if payload.get("type") != "password_reset":
                raise MCPAuthenticationError("Invalid password reset token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                raise MCPAuthenticationError("Password reset token has expired")
            
            return payload.get("sub")
            
        except JWTError as e:
            logger.warning(f"Password reset token verification failed: {e}")
            raise MCPAuthenticationError("Invalid password reset token")
    
    def generate_api_key(self) -> str:
        """
        Generate a secure API key.
        
        Returns:
            Secure API key string
        """
        return secrets.token_urlsafe(32)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Validation results with strength score and feedback
        """
        feedback = []
        score = 0
        
        # Length check
        if len(password) >= self.settings.password_min_length:
            score += 1
        else:
            feedback.append(f"Password must be at least {self.settings.password_min_length} characters long")
        
        # Character variety checks
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Password should contain uppercase letters")
        
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Password should contain lowercase letters")
        
        if any(c.isdigit() for c in password):
            score += 1
        else:
            feedback.append("Password should contain numbers")
        
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            feedback.append("Password should contain special characters")
        
        # Determine strength level
        if score <= 2:
            strength = "weak"
        elif score <= 3:
            strength = "medium"
        elif score <= 4:
            strength = "strong"
        else:
            strength = "very_strong"
        
        return {
            "score": score,
            "max_score": 5,
            "strength": strength,
            "is_valid": len(feedback) == 0,
            "feedback": feedback
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if authentication manager is initialized."""
        return self._initialized
