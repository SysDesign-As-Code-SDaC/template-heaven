"""
Python MCP SDK Template

A production-ready, containerized template for building Model Context Protocol (MCP)
servers and clients using the official Python SDK.

Based on: https://github.com/modelcontextprotocol/python-sdk

Features:
- Complete MCP server implementation
- MCP client examples
- Containerized deployment
- Database integration
- Authentication & security
- Monitoring & observability
- Comprehensive testing
- Development tools

Usage:
    from mcp_sdk import MCPServer, MCPClient
    
    # Create MCP server
    server = MCPServer()
    await server.start()
    
    # Create MCP client
    client = MCPClient("http://localhost:8000")
    await client.connect()
"""

__version__ = "1.0.0"
__author__ = "Template Heaven"
__email__ = "dev@templateheaven.dev"
__license__ = "MIT"

from .core.server import MCPServer
from .core.client import MCPClient
from .core.config import Settings
from .core.exceptions import MCPSDKError, MCPConnectionError, MCPValidationError

__all__ = [
    "MCPServer",
    "MCPClient", 
    "Settings",
    "MCPSDKError",
    "MCPConnectionError",
    "MCPValidationError",
]
