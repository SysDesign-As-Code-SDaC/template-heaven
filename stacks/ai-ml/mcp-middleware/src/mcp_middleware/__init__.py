"""
MCP Middleware - A unified interface for MCP (Model Context Protocol) servers.

This package provides a containerized middleware application that can accept
and manage multiple MCP servers, providing AI assistants with access to
external tools and data sources through a standardized protocol.
"""

__version__ = "1.0.0"
__author__ = "Template Heaven"
__description__ = "MCP Middleware for AI assistants"

from .main import app
from .core.server_manager import ServerManager
from .core.protocol_handler import ProtocolHandler
from .servers.base import BaseMCPServer
from .servers.filesystem import FilesystemServer
from .servers.database import DatabaseServer
from .servers.web import WebServer
from .servers.git import GitServer
from .servers.api import APIServer
from .servers.search import SearchServer
from .servers.execution import CodeExecutionServer
from .servers.vector import VectorSearchServer
from .servers.clickup import ClickUpServer

__all__ = [
    "app",
    "ServerManager",
    "ProtocolHandler",
    "BaseMCPServer",
    "FilesystemServer",
    "DatabaseServer",
    "WebServer",
    "GitServer",
    "APIServer",
    "SearchServer",
    "CodeExecutionServer",
    "VectorSearchServer",
    "ClickUpServer",
    "__version__"
]
