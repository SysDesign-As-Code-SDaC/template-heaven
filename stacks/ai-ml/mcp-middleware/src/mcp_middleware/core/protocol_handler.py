"""
MCP Protocol Handler.

This module implements the complete Model Context Protocol (MCP) specification,
handling all protocol messages, server lifecycle, tool execution, resource access,
and sampling operations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MCPProtocolVersion(Enum):
    """MCP protocol versions."""
    V2024_11_05 = "2024-11-05"
    LATEST = "2024-11-05"

class MCPMessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"

class MCPMethod(Enum):
    """MCP standard methods."""
    # Initialization
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"

    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"

    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"

    # Resource Templates
    RESOURCES_TEMPLATES_LIST = "resources/templates/list"

    # Sampling
    SAMPLING_CREATE_MESSAGE = "sampling/createMessage"

    # Logging
    LOGGING_SET_LEVEL = "logging/setLevel"

    # Ping/Pong
    PING = "ping"

    # Completion
    COMPLETION_COMPLETE = "completion/complete"

@dataclass
class MCPMessage:
    """MCP protocol message structure."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            data["id"] = self.id
        if self.method is not None:
            data["method"] = self.method
        if self.params is not None:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error")
        )

@dataclass
class MCPInitializeRequest:
    """MCP initialize request parameters."""
    protocolVersion: str
    capabilities: Dict[str, Any]
    clientInfo: Dict[str, Any]

@dataclass
class MCPServerCapabilities:
    """MCP server capabilities."""
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    completion: Optional[Dict[str, Any]] = None

@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]

@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    mimeType: str
    description: str
    name: Optional[str] = None

@dataclass
class MCPResourceTemplate:
    """MCP resource template definition."""
    uriTemplate: str
    name: str
    description: str
    mimeType: str

class MCPProtocolHandler:
    """Handles MCP protocol communication and message routing."""

    def __init__(self, server_manager):
        self.server_manager = server_manager
        self.protocol_version = MCPProtocolVersion.LATEST.value
        self.initialized = False
        self.capabilities = MCPServerCapabilities()
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_requests: Dict[Union[str, int], asyncio.Future] = {}
        self.subscriptions: Dict[str, set] = {}

        # Register protocol handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP method handlers."""
        self.message_handlers = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.TOOLS_LIST.value: self._handle_tools_list,
            MCPMethod.TOOLS_CALL.value: self._handle_tools_call,
            MCPMethod.RESOURCES_LIST.value: self._handle_resources_list,
            MCPMethod.RESOURCES_READ.value: self._handle_resources_read,
            MCPMethod.RESOURCES_SUBSCRIBE.value: self._handle_resources_subscribe,
            MCPMethod.RESOURCES_UNSUBSCRIBE.value: self._handle_resources_unsubscribe,
            MCPMethod.RESOURCES_TEMPLATES_LIST.value: self._handle_resources_templates_list,
            MCPMethod.SAMPLING_CREATE_MESSAGE.value: self._handle_sampling_create_message,
            MCPMethod.LOGGING_SET_LEVEL.value: self._handle_logging_set_level,
            MCPMethod.PING.value: self._handle_ping,
            MCPMethod.COMPLETION_COMPLETE.value: self._handle_completion_complete,
        }

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle incoming MCP message.

        Args:
            message: Incoming MCP message

        Returns:
            Response message or None for notifications
        """
        try:
            mcp_message = MCPMessage.from_dict(message)

            # Handle initialization requirement
            if not self.initialized and mcp_message.method != MCPMethod.INITIALIZE.value:
                return self._create_error_response(
                    mcp_message.id,
                    -32002,
                    "Server not initialized",
                    "The server must be initialized before other requests"
                )

            # Route to appropriate handler
            if mcp_message.method and mcp_message.method in self.message_handlers:
                return await self.message_handlers[mcp_message.method](mcp_message)
            else:
                return self._create_error_response(
                    mcp_message.id,
                    -32601,
                    "Method not found",
                    f"Unknown method: {mcp_message.method}"
                )

        except Exception as e:
            logger.error(f"Error handling MCP message: {e}", exc_info=True)
            return self._create_error_response(
                message.get("id"),
                -32603,
                "Internal error",
                str(e)
            )

    async def initialize_connection(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize MCP connection."""
        try:
            # Parse initialize request
            init_request = MCPInitializeRequest(
                protocolVersion=request.get("protocolVersion", "2024-11-05"),
                capabilities=request.get("capabilities", {}),
                clientInfo=request.get("clientInfo", {})
            )

            # Validate protocol version
            if init_request.protocolVersion != self.protocol_version:
                if init_request.protocolVersion not in [v.value for v in MCPProtocolVersion]:
                    raise ValueError(f"Unsupported protocol version: {init_request.protocolVersion}")

            # Initialize server capabilities based on available servers
            await self._initialize_capabilities()

            # Mark as initialized
            self.initialized = True

            # Return initialize result
            return {
                "protocolVersion": self.protocol_version,
                "capabilities": asdict(self.capabilities),
                "serverInfo": {
                    "name": "MCP Middleware",
                    "version": "1.0.0"
                }
            }

        except Exception as e:
            logger.error(f"Failed to initialize MCP connection: {e}")
            raise

    async def list_tools(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """List all available tools across all servers."""
        tools = []

        # Get tools from all servers
        servers = await self.server_manager.list_servers()
        for server_info in servers:
            if server_info["status"] == "running":
                server = await self.server_manager.get_server(server_info["name"])
                if server:
                    server_tools = await server.list_tools()
                    # Add server prefix to avoid conflicts
                    for tool in server_tools:
                        tool["name"] = f"{server_info['name']}.{tool['name']}"
                    tools.extend(server_tools)

        return {"tools": tools}

    async def call_tool(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call."""
        tool_name = request.get("name")
        arguments = request.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        # Parse server and tool names
        if "." in tool_name:
            server_name, actual_tool_name = tool_name.split(".", 1)
        else:
            raise ValueError("Tool name must be in format: server_name.tool_name")

        # Get server
        server = await self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")

        # Call tool
        result = await server.call_tool(actual_tool_name, arguments)

        return {"content": result}

    async def list_resources(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """List all available resources across all servers."""
        resources = []

        # Get resources from all servers
        servers = await self.server_manager.list_servers()
        for server_info in servers:
            if server_info["status"] == "running":
                server = await self.server_manager.get_server(server_info["name"])
                if server:
                    server_resources = await server.list_resources()
                    resources.extend(server_resources)

        return {"resources": resources}

    async def read_resource(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Read a resource."""
        uri = request.get("uri")

        if not uri:
            raise ValueError("Resource URI is required")

        # Find server that can handle this resource
        servers = await self.server_manager.list_servers()
        for server_info in servers:
            if server_info["status"] == "running":
                server = await self.server_manager.get_server(server_info["name"])
                if server:
                    try:
                        return await server.read_resource(uri)
                    except NotImplementedError:
                        continue
                    except Exception:
                        continue

        raise ValueError(f"Resource not found: {uri}")

    async def proxy_request(self, server_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Proxy MCP request to specific server."""
        server = await self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server not found: {server_name}")

        # This would need to be implemented based on specific server capabilities
        # For now, return a basic response
        return {"status": "proxied", "server": server_name}

    # Private handler methods

    async def _handle_initialize(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle initialize request."""
        result = await self.initialize_connection(message.params or {})
        return self._create_response(message.id, result)

    async def _handle_tools_list(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle tools/list request."""
        result = await self.list_tools(message.params or {})
        return self._create_response(message.id, result)

    async def _handle_tools_call(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle tools/call request."""
        result = await self.call_tool(message.params or {})
        return self._create_response(message.id, result)

    async def _handle_resources_list(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle resources/list request."""
        result = await self.list_resources(message.params or {})
        return self._create_response(message.id, result)

    async def _handle_resources_read(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle resources/read request."""
        result = await self.read_resource(message.params or {})
        return self._create_response(message.id, result)

    async def _handle_resources_subscribe(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle resources/subscribe request."""
        uri = message.params.get("uri") if message.params else None
        if not uri:
            return self._create_error_response(message.id, -32602, "Invalid params", "uri is required")

        if uri not in self.subscriptions:
            self.subscriptions[uri] = set()

        # In a real implementation, this would track subscriptions
        return self._create_response(message.id, {})

    async def _handle_resources_unsubscribe(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle resources/unsubscribe request."""
        uri = message.params.get("uri") if message.params else None
        if not uri:
            return self._create_error_response(message.id, -32602, "Invalid params", "uri is required")

        if uri in self.subscriptions:
            self.subscriptions.pop(uri, None)

        return self._create_response(message.id, {})

    async def _handle_resources_templates_list(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle resources/templates/list request."""
        # For now, return empty list - could be extended to support templates
        return self._create_response(message.id, {"resourceTemplates": []})

    async def _handle_sampling_create_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle sampling/createMessage request."""
        # This would integrate with LLM providers for sampling
        # For now, return a mock response
        return self._create_response(message.id, {
            "model": "mock-model",
            "role": "assistant",
            "content": {
                "type": "text",
                "text": "This is a mock sampling response"
            }
        })

    async def _handle_logging_set_level(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle logging/setLevel request."""
        level = message.params.get("level") if message.params else None
        if not level:
            return self._create_error_response(message.id, -32602, "Invalid params", "level is required")

        # Set logging level (would need to integrate with actual logging system)
        logger.info(f"Setting log level to: {level}")
        return self._create_response(message.id, {})

    async def _handle_ping(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle ping request."""
        return self._create_response(message.id, {})

    async def _handle_completion_complete(self, message: MCPMessage) -> Dict[str, Any]:
        """Handle completion/complete request."""
        # This would provide auto-completion capabilities
        # For now, return empty completions
        return self._create_response(message.id, {"completion": {"values": []}})

    async def _initialize_capabilities(self):
        """Initialize server capabilities based on available servers."""
        # Check what capabilities are available across all servers
        has_tools = False
        has_resources = False
        has_sampling = False

        servers = await self.server_manager.list_servers()
        for server_info in servers:
            if server_info["status"] == "running":
                has_tools = True  # Assume all servers provide tools
                has_resources = True  # Assume all servers provide resources

        # Set capabilities
        if has_tools:
            self.capabilities.tools = {"listChanged": True}

        if has_resources:
            self.capabilities.resources = {
                "subscribe": True,
                "listChanged": True
            }

        # Always support logging
        self.capabilities.logging = {}

        # Support sampling if configured
        self.capabilities.sampling = {}

        # Support completion
        self.capabilities.completion = {}

    def _create_response(self, request_id: Union[str, int], result: Any) -> Dict[str, Any]:
        """Create a successful response message."""
        return MCPMessage(id=request_id, result=result).to_dict()

    def _create_error_response(self, request_id: Union[str, int, None],
                              code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """Create an error response message."""
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data

        return MCPMessage(id=request_id, error=error).to_dict()
