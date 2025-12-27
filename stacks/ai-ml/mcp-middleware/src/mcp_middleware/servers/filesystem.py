"""
Filesystem MCP Server.

This server provides file system operations through the MCP protocol,
allowing AI assistants to read, write, and manage files and directories.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import mimetypes

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class FilesystemServer(BaseMCPServer):
    """MCP server for filesystem operations."""

    def __init__(self, name: str, config: Dict[str, Any], auth: Dict[str, Any]):
        super().__init__(name, config, auth)
        self.root_path = Path(config.get("root_path", "/tmp")).resolve()
        self.allowed_operations = config.get("allowed_operations", ["read", "list"])
        self.max_file_size = config.get("max_file_size", "10MB")
        self.max_file_size_bytes = self._parse_file_size(self.max_file_size)

        # Ensure root path exists
        self.root_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_config(cls, config: Dict[str, Any]):
        """Validate filesystem server configuration."""
        root_path = config.get("root_path")
        if root_path and not os.path.isabs(root_path):
            raise ValueError("root_path must be an absolute path")

        allowed_ops = config.get("allowed_operations", [])
        valid_ops = ["read", "write", "list", "delete", "create"]
        for op in allowed_ops:
            if op not in valid_ops:
                raise ValueError(f"Invalid operation: {op}")

    async def initialize(self):
        """Initialize the filesystem server."""
        try:
            # Test access to root directory
            if not self.root_path.exists():
                raise FileNotFoundError(f"Root path does not exist: {self.root_path}")

            if not os.access(self.root_path, os.R_OK):
                raise PermissionError(f"No read access to root path: {self.root_path}")

            if "write" in self.allowed_operations and not os.access(self.root_path, os.W_OK):
                raise PermissionError(f"No write access to root path: {self.root_path}")

            self.initialized = True
            logger.info(f"Filesystem server {self.name} initialized with root: {self.root_path}")

        except Exception as e:
            logger.error(f"Failed to initialize filesystem server {self.name}: {e}")
            raise

    async def shutdown(self):
        """Shutdown the filesystem server."""
        self.initialized = False
        logger.info(f"Filesystem server {self.name} shutdown")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic filesystem operations
            test_file = self.root_path / ".health_check"
            if "write" in self.allowed_operations:
                test_file.write_text("health_check")
                test_file.unlink()

            return {
                "healthy": True,
                "timestamp": datetime.utcnow().isoformat(),
                "server": self.name,
                "root_path": str(self.root_path),
                "operations": self.allowed_operations
            }
        except Exception as e:
            return {
                "healthy": False,
                "timestamp": datetime.utcnow().isoformat(),
                "server": self.name,
                "error": str(e)
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available filesystem tools."""
        tools = []

        if "read" in self.allowed_operations:
            tools.append({
                "name": "read_file",
                "description": "Read the contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (relative to root)"
                        }
                    },
                    "required": ["path"]
                }
            })

        if "list" in self.allowed_operations:
            tools.append({
                "name": "list_directory",
                "description": "List contents of a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to list (relative to root)",
                            "default": "."
                        }
                    }
                }
            })

        if "write" in self.allowed_operations:
            tools.append({
                "name": "write_file",
                "description": "Write content to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write (relative to root)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            })

        if "create" in self.allowed_operations:
            tools.append({
                "name": "create_directory",
                "description": "Create a new directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to create (relative to root)"
                        }
                    },
                    "required": ["path"]
                }
            })

        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a filesystem tool."""
        try:
            if tool_name == "read_file":
                return await self._read_file(arguments)
            elif tool_name == "list_directory":
                return await self._list_directory(arguments)
            elif tool_name == "write_file":
                return await self._write_file(arguments)
            elif tool_name == "create_directory":
                return await self._create_directory(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            return self._handle_error(f"call_tool_{tool_name}", e)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List filesystem resources."""
        return [
            {
                "uri": f"file://{self.root_path}",
                "mimeType": "inode/directory",
                "description": f"Root directory: {self.root_path}"
            }
        ]

    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Read a filesystem resource."""
        if not resource_uri.startswith(f"file://{self.root_path}"):
            raise ValueError(f"Resource URI not within root path: {resource_uri}")

        path = Path(resource_uri.replace(f"file://{self.root_path}", "").lstrip("/"))
        full_path = (self.root_path / path).resolve()

        # Security check: ensure path is within root
        if not str(full_path).startswith(str(self.root_path)):
            raise ValueError("Access denied: path outside root directory")

        if not full_path.exists():
            raise FileNotFoundError(f"Resource not found: {resource_uri}")

        if full_path.is_file():
            content = full_path.read_text()
            mime_type, _ = mimetypes.guess_type(str(full_path))
            return {
                "contents": [
                    {
                        "uri": resource_uri,
                        "mimeType": mime_type or "text/plain",
                        "text": content
                    }
                ]
            }
        else:
            raise ValueError("Resource is not a file")

    async def _read_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Read a file."""
        path = arguments["path"]
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if full_path.stat().st_size > self.max_file_size_bytes:
            raise ValueError(f"File too large: {full_path.stat().st_size} bytes")

        content = full_path.read_text()
        return {
            "content": content,
            "path": path,
            "size": len(content),
            "encoding": "utf-8"
        }

    async def _list_directory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents."""
        path = arguments.get("path", ".")
        full_path = self._resolve_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not full_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        items = []
        for item in full_path.iterdir():
            stat = item.stat()
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else 0,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return {
            "path": path,
            "items": items,
            "count": len(items)
        }

    async def _write_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file."""
        if "write" not in self.allowed_operations:
            raise PermissionError("Write operation not allowed")

        path = arguments["path"]
        content = arguments["content"]

        full_path = self._resolve_path(path)

        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Check file size limit
        if len(content) > self.max_file_size_bytes:
            raise ValueError(f"Content too large: {len(content)} bytes")

        full_path.write_text(content)

        return {
            "path": path,
            "size": len(content),
            "written": True
        }

    async def _create_directory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new directory."""
        if "create" not in self.allowed_operations:
            raise PermissionError("Create operation not allowed")

        path = arguments["path"]
        full_path = self._resolve_path(path)

        full_path.mkdir(parents=True, exist_ok=True)

        return {
            "path": path,
            "created": True
        }

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to root, with security checks."""
        # Normalize path to prevent directory traversal
        path = os.path.normpath(path)

        # Prevent absolute paths and parent directory traversal
        if os.path.isabs(path) or ".." in path:
            raise ValueError(f"Invalid path: {path}")

        full_path = (self.root_path / path).resolve()

        # Final security check: ensure path is within root
        if not str(full_path).startswith(str(self.root_path)):
            raise ValueError(f"Access denied: path outside root directory")

        return full_path

    def _parse_file_size(self, size_str: str) -> int:
        """Parse file size string (e.g., '10MB') to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
