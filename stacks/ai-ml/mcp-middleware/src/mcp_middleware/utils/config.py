"""
Configuration management for MCP Middleware.

This module handles environment variables, settings validation, and configuration
management for the MCP middleware application.
"""

import os
from pathlib import Path
from typing import List, Optional
import secrets

class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # Server Configuration
        self.host: str = os.getenv("MCP_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("MCP_PORT", "8000"))
        self.workers: int = int(os.getenv("MCP_WORKERS", "4"))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"

        # Security
        self.secret_key: str = os.getenv("MCP_SECRET_KEY", secrets.token_hex(32))
        self.allowed_origins: List[str] = os.getenv(
            "MCP_ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:5173"
        ).split(",")

        # Database
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            "sqlite:///./mcp_servers.db"
        )

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_format: str = os.getenv("LOG_FORMAT", "json")

        # Rate Limiting
        self.rate_limit_requests_per_minute: int = int(os.getenv(
            "RATE_LIMIT_REQUESTS_PER_MINUTE", "100"
        ))
        self.rate_limit_burst_size: int = int(os.getenv(
            "RATE_LIMIT_BURST_SIZE", "20"
        ))

        # Paths
        self.base_dir: Path = Path(__file__).parent.parent.parent.parent
        self.config_dir: Path = self.base_dir / "config"
        self.data_dir: Path = self.base_dir / "data"
        self.logs_dir: Path = self.base_dir / "logs"

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Validate configuration settings."""
        # Validate host/port
        if not (0 <= self.port <= 65535):
            raise ValueError(f"Invalid port number: {self.port}")

        # Validate database URL
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")

        # Validate secret key
        if len(self.secret_key) < 32:
            raise ValueError("MCP_SECRET_KEY must be at least 32 characters")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid LOG_LEVEL: {self.log_level}")

    def get_database_config(self) -> dict:
        """Get database-specific configuration."""
        return {
            "url": self.database_url,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
        }

    def get_redis_config(self) -> Optional[dict]:
        """Get Redis configuration if available."""
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            return {
                "url": redis_url,
                "decode_responses": True,
                "socket_connect_timeout": 5,
                "socket_timeout": 5,
            }
        return None

    def get_cors_config(self) -> dict:
        """Get CORS configuration."""
        return {
            "allow_origins": [origin.strip() for origin in self.allowed_origins],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    def get_rate_limit_config(self) -> dict:
        """Get rate limiting configuration."""
        return {
            "requests_per_minute": self.rate_limit_requests_per_minute,
            "burst_size": self.rate_limit_burst_size,
        }

# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    raise RuntimeError(f"Configuration validation failed: {e}")
