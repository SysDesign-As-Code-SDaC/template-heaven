"""
MCP Server Configuration

Centralized configuration management for the Template Heaven MCP server.
Supports environment variables and validation.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class MCPConfig(BaseSettings):
    """
    Configuration for the Template Heaven MCP server.

    All settings can be overridden with environment variables.
    """

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 3000
    debug: bool = False

    # WebSocket Configuration
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    close_timeout: float = 5.0

    # Request Configuration
    request_timeout: float = 30.0
    max_message_size: int = 1024 * 1024  # 1MB

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: float = 60.0  # seconds

    # Service URLs (Microservices)
    template_service_url: str = "http://template-service:8001"
    validation_service_url: str = "http://validation-service:8002"
    generation_service_url: str = "http://generation-service:8003"
    analysis_service_url: str = "http://analysis-service:8004"
    user_service_url: str = "http://user-service:8005"
    sync_service_url: str = "http://sync-service:8006"
    api_gateway_url: str = "http://api-gateway:8007"

    # Database Configuration
    database_url: str = "postgresql+asyncpg://mcp:mcp@localhost:5432/mcp_db"
    redis_url: str = "redis://localhost:6379/0"

    # Security Configuration
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json" if not debug else "console"

    # Monitoring Configuration
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_health_checks: bool = True

    # Cache Configuration
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    cache_max_size: int = 1000

    # File Upload Configuration
    upload_max_size: int = 10 * 1024 * 1024  # 10MB
    upload_allowed_extensions: list = ["json", "yaml", "yml", "md", "txt"]

    # Template Heaven Specific
    template_storage_path: str = "/app/templates"
    project_storage_path: str = "/app/projects"
    max_concurrent_generations: int = 5

    # External Service Configuration
    github_token: Optional[str] = None
    docker_registry: str = "docker.io"
    container_registry_token: Optional[str] = None

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
config = MCPConfig()


def get_config() -> MCPConfig:
    """Get the global configuration instance."""
    return config


def create_test_config() -> MCPConfig:
    """Create configuration for testing."""
    test_config = MCPConfig()
    test_config.debug = True
    test_config.database_url = "sqlite+aiosqlite:///./test.db"
    test_config.redis_url = "redis://localhost:6379/1"
    test_config.template_service_url = "http://localhost:8001"
    test_config.validation_service_url = "http://localhost:8002"
    test_config.generation_service_url = "http://localhost:8003"
    return test_config
