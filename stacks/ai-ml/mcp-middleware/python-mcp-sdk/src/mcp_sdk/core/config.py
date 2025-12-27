"""
Configuration management for MCP SDK Template.

Provides centralized configuration using Pydantic settings with environment
variable support and validation.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://mcp_user:mcp_pass@localhost:5432/mcp_db",
        description="Database connection URL"
    )
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=30, description="Maximum database connection overflow")
    pool_timeout: int = Field(default=30, description="Database connection timeout")
    pool_recycle: int = Field(default=3600, description="Database connection recycle time")
    
    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    max_connections: int = Field(default=20, description="Maximum Redis connections")
    socket_timeout: int = Field(default=5, description="Redis socket timeout")
    socket_connect_timeout: int = Field(default=5, description="Redis connection timeout")
    
    class Config:
        env_prefix = "REDIS_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(
        default="mcp-sdk-secret-key-change-in-production",
        description="Secret key for JWT tokens and encryption"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")
    password_min_length: int = Field(default=8, description="Minimum password length")
    max_login_attempts: int = Field(default=5, description="Maximum login attempts before lockout")
    lockout_duration_minutes: int = Field(default=30, description="Account lockout duration")
    
    class Config:
        env_prefix = "SECURITY_"


class MCPSettings(BaseSettings):
    """MCP-specific configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="MCP server host")
    port: int = Field(default=8000, description="MCP server port")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    max_connections: int = Field(default=100, description="Maximum concurrent connections")
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    request_timeout: int = Field(default=60, description="Request timeout in seconds")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    class Config:
        env_prefix = "MCP_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    log_requests: bool = Field(default=True, description="Log all requests")
    log_responses: bool = Field(default=False, description="Log all responses")
    
    class Config:
        env_prefix = "MONITORING_"


class Settings(PydanticBaseSettings):
    """
    Main application settings.
    
    Combines all configuration sections and provides environment-based
    configuration with validation.
    """
    
    # Application metadata
    app_name: str = Field(default="Python MCP SDK Template", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_description: str = Field(
        default="Production-ready MCP server and client template",
        description="Application description"
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    
    # Configuration sections
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    cors_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed CORS methods"
    )
    cors_headers: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ['development', 'staging', 'production', 'testing']
        if v.lower() not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v.lower()
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings: Application configuration settings
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from environment.
    
    Returns:
        Settings: Reloaded application configuration settings
    """
    global settings
    settings = Settings()
    return settings
