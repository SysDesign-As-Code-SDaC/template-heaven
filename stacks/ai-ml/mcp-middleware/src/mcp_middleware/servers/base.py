"""
Base MCP Server Interface.

This module defines the base interface that all MCP servers must implement,
providing a consistent way to interact with different types of MCP servers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseMCPServer(ABC):
    """Abstract base class for all MCP servers."""

    def __init__(self, name: str, config: Dict[str, Any], auth: Dict[str, Any]):
        """
        Initialize the MCP server.

        Args:
            name: Server name/identifier
            config: Server-specific configuration
            auth: Authentication configuration
        """
        self.name = name
        self.config = config
        self.auth = auth
        self.initialized = False
        self.created_at = datetime.utcnow()

    @abstractmethod
    async def initialize(self):
        """Initialize the server and establish connections."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the server and cleanup resources."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the server.

        Returns:
            Health status information
        """
        return {
            "healthy": True,
            "timestamp": datetime.utcnow().isoformat(),
            "server": self.name
        }

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools provided by this server.

        Returns:
            List of tool definitions
        """
        return []

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific tool with given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        raise NotImplementedError(f"Tool '{tool_name}' not implemented")

    @abstractmethod
    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List available resources provided by this server.

        Returns:
            List of resource definitions
        """
        return []

    @abstractmethod
    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """
        Read a specific resource.

        Args:
            resource_uri: URI of the resource to read

        Returns:
            Resource content
        """
        raise NotImplementedError(f"Resource '{resource_uri}' not found")

    @classmethod
    def validate_config(cls, config: Dict[str, Any]):
        """
        Validate server configuration.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Default implementation - subclasses can override
        pass

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and capabilities."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "initialized": self.initialized,
            "created_at": self.created_at.isoformat(),
            "capabilities": self._get_capabilities()
        }

    def _get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities."""
        return {
            "tools": len(self.list_tools()),
            "resources": len(self.list_resources()),
            "authentication": bool(self.auth),
            "health_check": True
        }

    def _handle_error(self, operation: str, error: Exception) -> Dict[str, Any]:
        """Handle and format errors consistently."""
        error_msg = f"Error in {operation}: {str(error)}"
        logger.error(error_msg, exc_info=True)

        return {
            "error": True,
            "message": error_msg,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _validate_auth(self, request_auth: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate authentication for operations.

        Args:
            request_auth: Authentication from request

        Returns:
            True if authenticated, False otherwise
        """
        if not self.auth:
            return True  # No auth required

        if not request_auth:
            return False

        auth_type = self.auth.get("type")
        if auth_type == "bearer":
            return request_auth.get("token") == self.auth.get("token")

        # Add other auth types as needed
        return False
