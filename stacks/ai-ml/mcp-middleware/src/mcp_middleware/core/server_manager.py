"""
MCP Server Manager.

This module manages the lifecycle of MCP servers, including registration,
health monitoring, and communication with individual server instances.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from ..servers.base import BaseMCPServer
from ..servers.filesystem import FilesystemServer
from ..servers.database import DatabaseServer
from ..servers.web import WebServer
from ..servers.git import GitServer
from ..servers.api import APIServer
from ..servers.search import SearchServer
from ..servers.execution import CodeExecutionServer
from ..servers.vector import VectorSearchServer
from ..servers.clickup import ClickUpServer
from ..utils.config import settings

logger = logging.getLogger(__name__)

class ServerManager:
    """Manages MCP server instances and their configurations."""

    def __init__(self):
        self.servers: Dict[str, BaseMCPServer] = {}
        self.server_configs: Dict[str, Dict] = {}
        self.server_types = {
            "filesystem": FilesystemServer,
            "database": DatabaseServer,
            "web": WebServer,
            "git": GitServer,
            "api": APIServer,
            "search": SearchServer,
            "execution": CodeExecutionServer,
            "vector": VectorSearchServer,
            "clickup": ClickUpServer
        }

    async def initialize(self):
        """Initialize the server manager and load existing configurations."""
        logger.info("Initializing MCP Server Manager")

        # Load server configurations from storage
        await self._load_server_configs()

        # Initialize configured servers
        for server_name, config in self.server_configs.items():
            if config.get("enabled", True):
                try:
                    await self._initialize_server(server_name, config)
                except Exception as e:
                    logger.error(f"Failed to initialize server {server_name}: {e}")

        logger.info(f"Server Manager initialized with {len(self.servers)} servers")

    async def shutdown(self):
        """Shutdown all MCP servers gracefully."""
        logger.info("Shutting down MCP Server Manager")

        shutdown_tasks = []
        for server_name, server in self.servers.items():
            shutdown_tasks.append(self._shutdown_server(server_name, server))

        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self.servers.clear()

        logger.info("Server Manager shutdown complete")

    async def add_server(self, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new MCP server configuration.

        Args:
            server_config: Server configuration dictionary

        Returns:
            Server creation result

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        self._validate_server_config(server_config)

        server_name = server_config["name"]

        if server_name in self.servers:
            raise ValueError(f"Server '{server_name}' already exists")

        # Store configuration
        self.server_configs[server_name] = server_config
        await self._save_server_config(server_name, server_config)

        # Initialize server if enabled
        if server_config.get("enabled", True):
            await self._initialize_server(server_name, server_config)

        logger.info(f"Added MCP server: {server_name}")
        return {
            "status": "success",
            "server": server_name,
            "message": "Server added successfully"
        }

    async def remove_server(self, server_name: str) -> Dict[str, Any]:
        """
        Remove an MCP server configuration.

        Args:
            server_name: Name of the server to remove

        Returns:
            Server removal result

        Raises:
            ValueError: If server doesn't exist
        """
        if server_name not in self.server_configs:
            raise ValueError(f"Server '{server_name}' not found")

        # Shutdown server if running
        if server_name in self.servers:
            await self._shutdown_server(server_name, self.servers[server_name])
            del self.servers[server_name]

        # Remove configuration
        del self.server_configs[server_name]
        await self._delete_server_config(server_name)

        logger.info(f"Removed MCP server: {server_name}")
        return {
            "status": "success",
            "server": server_name,
            "message": "Server removed successfully"
        }

    async def list_servers(self) -> List[Dict[str, Any]]:
        """
        List all configured MCP servers with their status.

        Returns:
            List of server information
        """
        servers = []

        for server_name, config in self.server_configs.items():
            server_info = {
                "name": server_name,
                "type": config.get("type"),
                "version": config.get("version", "1.0"),
                "enabled": config.get("enabled", True),
                "status": "running" if server_name in self.servers else "stopped",
                "config": config.get("config", {}),
                "created_at": config.get("created_at"),
                "last_health_check": None
            }

            # Add health status if server is running
            if server_name in self.servers:
                health_status = await self.servers[server_name].health_check()
                server_info["last_health_check"] = health_status

            servers.append(server_info)

        return servers

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all servers.

        Returns:
            Health status information
        """
        overall_health = True
        server_health = {}

        for server_name, server in self.servers.items():
            try:
                health = await server.health_check()
                server_health[server_name] = health
                if not health.get("healthy", False):
                    overall_health = False
            except Exception as e:
                logger.error(f"Health check failed for {server_name}: {e}")
                server_health[server_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                overall_health = False

        return {
            "overall": overall_health,
            "servers": server_health,
            "timestamp": datetime.utcnow().isoformat(),
            "total_servers": len(self.servers),
            "active_servers": len([s for s in server_health.values() if s.get("healthy")])
        }

    async def get_server(self, server_name: str) -> Optional[BaseMCPServer]:
        """
        Get a server instance by name.

        Args:
            server_name: Name of the server

        Returns:
            Server instance or None if not found
        """
        return self.servers.get(server_name)

    def _validate_server_config(self, config: Dict[str, Any]):
        """Validate server configuration."""
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        if config["type"] not in self.server_types:
            raise ValueError(f"Unsupported server type: {config['type']}")

        # Validate server-specific configuration
        server_class = self.server_types[config["type"]]
        if hasattr(server_class, "validate_config"):
            server_class.validate_config(config.get("config", {}))

    async def _initialize_server(self, server_name: str, config: Dict[str, Any]):
        """Initialize a single MCP server."""
        server_type = config["type"]
        server_class = self.server_types[server_type]

        logger.info(f"Initializing {server_type} server: {server_name}")

        server = server_class(
            name=server_name,
            config=config.get("config", {}),
            auth=config.get("authentication", {})
        )

        await server.initialize()
        self.servers[server_name] = server

        logger.info(f"Server {server_name} initialized successfully")

    async def _shutdown_server(self, server_name: str, server: BaseMCPServer):
        """Shutdown a single MCP server."""
        try:
            logger.info(f"Shutting down server: {server_name}")
            await server.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down server {server_name}: {e}")

    async def _load_server_configs(self):
        """Load server configurations from storage."""
        # In a real implementation, this would load from a database
        # For now, we'll use a simple JSON file
        config_file = settings.CONFIG_DIR / "servers.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.server_configs = json.load(f)
                logger.info(f"Loaded {len(self.server_configs)} server configurations")
            except Exception as e:
                logger.error(f"Failed to load server configurations: {e}")
                self.server_configs = {}

    async def _save_server_config(self, server_name: str, config: Dict[str, Any]):
        """Save server configuration to storage."""
        config_file = settings.CONFIG_DIR / "servers.json"

        # Add timestamp
        config["created_at"] = datetime.utcnow().isoformat()

        try:
            # Ensure config directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w') as f:
                json.dump(self.server_configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save server configuration: {e}")

    async def _delete_server_config(self, server_name: str):
        """Delete server configuration from storage."""
        config_file = settings.CONFIG_DIR / "servers.json"

        try:
            if config_file.exists():
                with open(config_file, 'w') as f:
                    json.dump(self.server_configs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to delete server configuration: {e}")
