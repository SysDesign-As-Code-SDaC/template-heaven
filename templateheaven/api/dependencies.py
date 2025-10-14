"""
API dependencies for Template Heaven.

This module provides dependency injection for the FastAPI application,
including settings, authentication, and service dependencies.
"""

from functools import lru_cache
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import Field
from pydantic_settings import BaseSettings

# Import will be done locally to avoid circular imports
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = Field(default="Template Heaven API", description="Application name")
    app_description: str = Field(default="Template management and discovery service", description="Application description")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Log level")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    
    # CORS
    cors_origins: list = Field(default=["*"], description="CORS allowed origins")
    trusted_hosts: Optional[list] = Field(default=None, description="Trusted hosts")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./templateheaven.db", description="Database URL")
    redis_url: Optional[str] = Field(default=None, description="Redis URL")
    
    # GitHub API
    github_token: Optional[str] = Field(default=None, description="GitHub API token")
    github_rate_limit: int = Field(default=5000, description="GitHub API rate limit")
    
    # Cache
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Cache max size")
    
    # Templates
    templates_dir: str = Field(default="./templates", description="Templates directory")
    stacks_dir: str = Field(default="./stacks", description="Stacks directory")
    
    # Documentation
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return getattr(self, key, default)


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings)
) -> Optional[User]:
    """Get current authenticated user."""
    if not credentials:
        return None
    
    try:
        # In a real implementation, you would validate the JWT token here
        # For now, we'll create a mock user based on the token
        token = credentials.credentials
        
        # Mock user creation (replace with actual JWT validation)
        from ..database.models import User, UserRole
        
        if token == "admin-token":
            return User(
                id="admin-1",
                username="admin",
                email="admin@templateheaven.dev",
                role=UserRole.ADMIN
            )
        elif token == "user-token":
            return User(
                id="user-1",
                username="user",
                email="user@templateheaven.dev",
                role=UserRole.USER
            )
        else:
            return None
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None


def get_optional_user(
    current_user = Depends(get_current_user)
):
    """Get optional current user (doesn't raise exception if not authenticated)."""
    return current_user


def require_auth(
    current_user = Depends(get_current_user)
):
    """Require authentication (raises exception if not authenticated)."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_admin(
    current_user: User = Depends(require_auth)
) -> User:
    """Require admin role."""
    if current_user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


def get_api_key(request: Request, settings: Settings = Depends(get_settings)) -> Optional[str]:
    """Get API key from request headers."""
    return request.headers.get(settings.api_key_header)


def require_api_key(
    api_key: Optional[str] = Depends(get_api_key),
    settings: Settings = Depends(get_settings)
) -> str:
    """Require valid API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # In a real implementation, you would validate the API key against a database
    # For now, we'll use a simple check
    valid_api_keys = ["templateheaven-api-key", "admin-api-key"]
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key


def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


def get_user_agent(request: Request) -> str:
    """Get user agent from request headers."""
    return request.headers.get("user-agent", "unknown")


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check for forwarded headers first
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    if hasattr(request.client, "host"):
        return request.client.host
    
    return "unknown"


class ServiceContainer:
    """Service container for dependency injection."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
    
    def register(self, name: str, service: Any) -> None:
        """Register a service."""
        self._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    def get(self, name: str) -> Any:
        """Get a service."""
        if name not in self._services:
            raise ValueError(f"Service not found: {name}")
        return self._services[name]
    
    def has(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._services


# Global service container
service_container = ServiceContainer()


def get_service_container() -> ServiceContainer:
    """Get the global service container."""
    return service_container


def get_service(service_name: str):
    """Get a service from the container."""
    def _get_service(container: ServiceContainer = Depends(get_service_container)):
        return container.get(service_name)
    return _get_service
