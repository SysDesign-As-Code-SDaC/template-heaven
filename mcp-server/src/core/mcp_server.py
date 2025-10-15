"""
Template Heaven MCP Server

Main MCP server implementation that handles MCP protocol communication and routes
requests to microservices. This server implements the complete MCP specification
and provides middleware capabilities for all Template Heaven services.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed

from .protocol import MCPMessage, MCPRequest, MCPResponse, MCPNotification, MCPError
from .protocol import MCPErrorCode, MCPContent, MCPTool, MCPResource, MCPPrompt
from .service_registry import ServiceRegistry
from .middleware import MiddlewareManager
from .config import MCPConfig

logger = logging.getLogger(__name__)


class TemplateHeavenMCPServer:
    """
    Main MCP server for Template Heaven.

    This server implements the complete MCP protocol and acts as middleware
    for routing requests to specialized microservices.
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self.service_registry = ServiceRegistry(config)
        self.middleware_manager = MiddlewareManager(config)
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        self.request_handlers: Dict[str, Callable] = {}

        # Register core MCP method handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP method handlers."""
        # Core MCP methods
        self.request_handlers["initialize"] = self.handle_initialize
        self.request_handlers["ping"] = self.handle_ping
        self.request_handlers["shutdown"] = self.handle_shutdown

        # Tool methods
        self.request_handlers["tools/list"] = self.handle_tools_list
        self.request_handlers["tools/call"] = self.handle_tools_call

        # Resource methods
        self.request_handlers["resources/list"] = self.handle_resources_list
        self.request_handlers["resources/read"] = self.handle_resources_read
        self.request_handlers["resources/subscribe"] = self.handle_resources_subscribe
        self.request_handlers["resources/unsubscribe"] = self.handle_resources_unsubscribe

        # Prompt methods
        self.request_handlers["prompts/list"] = self.handle_prompts_list
        self.request_handlers["prompts/get"] = self.handle_prompts_get

        # Template Heaven specific methods
        self.request_handlers["template-heaven/templates/list"] = self.handle_templates_list
        self.request_handlers["template-heaven/templates/generate"] = self.handle_templates_generate
        self.request_handlers["template-heaven/templates/validate"] = self.handle_templates_validate
        self.request_handlers["template-heaven/stacks/list"] = self.handle_stacks_list
        self.request_handlers["template-heaven/projects/create"] = self.handle_projects_create

    async def start(self):
        """Start the MCP server."""
        logger.info(f"Starting Template Heaven MCP Server v{__version__}")

        # Initialize service registry
        await self.service_registry.initialize()

        # Start middleware
        await self.middleware_manager.initialize()

        # Start WebSocket server
        server = await websockets.serve(
            self.handle_connection,
            self.config.host,
            self.config.port,
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout,
            close_timeout=self.config.close_timeout,
        )

        logger.info(f"MCP Server listening on ws://{self.config.host}:{self.config.port}")

        # Keep server running
        await server.wait_closed()

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol):
        """Handle a WebSocket connection."""
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = {
            "websocket": websocket,
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "client_info": {},
            "subscriptions": set(),
        }

        logger.info(f"New MCP connection: {connection_id}")

        try:
            async for message_raw in websocket:
                try:
                    # Update activity timestamp
                    self.active_connections[connection_id]["last_activity"] = datetime.utcnow()

                    # Parse message
                    message_data = json.loads(message_raw)
                    message = MCPMessage.from_dict(message_data)

                    # Process message
                    response = await self.process_message(message, connection_id)

                    # Send response if any
                    if response:
                        await websocket.send(response.to_json())

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON message: {e}")
                    error_response = MCPError(
                        code=MCPErrorCode.PARSE_ERROR,
                        message="Invalid JSON"
                    ).to_response(message.id if 'message' in locals() else None)
                    await websocket.send(error_response.to_json())

                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    error_response = MCPError(
                        code=MCPErrorCode.INTERNAL_ERROR,
                        message="Internal server error"
                    ).to_response(message.id if 'message' in locals() else None)
                    await websocket.send(error_response.to_json())

        except ConnectionClosed:
            logger.info(f"MCP connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Connection error for {connection_id}: {e}")
        finally:
            # Clean up connection
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            # Clean up subscriptions
            await self.cleanup_subscriptions(connection_id)

    async def process_message(self, message: MCPMessage, connection_id: str) -> Optional[MCPMessage]:
        """Process an MCP message and return response."""
        # Apply middleware
        message = await self.middleware_manager.process_request(message, connection_id)

        if isinstance(message, MCPRequest):
            # Handle request
            response = await self.handle_request(message, connection_id)
            return response

        elif isinstance(message, MCPNotification):
            # Handle notification (no response expected)
            await self.handle_notification(message, connection_id)
            return None

        else:
            # Invalid message type
            return MCPError(
                code=MCPErrorCode.INVALID_REQUEST,
                message="Invalid message type"
            ).to_response(message.id)

    async def handle_request(self, request: MCPRequest, connection_id: str) -> MCPMessage:
        """Handle an MCP request."""
        handler = self.request_handlers.get(request.method)

        if not handler:
            return MCPError(
                code=MCPErrorCode.METHOD_NOT_FOUND,
                message=f"Method '{request.method}' not found"
            ).to_response(request.id)

        try:
            # Call handler
            result = await handler(request, connection_id)

            # Create response
            response = MCPResponse(
                id=request.id,
                result=result
            )

            return response

        except Exception as e:
            logger.error(f"Handler error for {request.method}: {e}")
            return MCPError(
                code=MCPErrorCode.INTERNAL_ERROR,
                message="Handler execution failed"
            ).to_response(request.id)

    async def handle_notification(self, notification: MCPNotification, connection_id: str):
        """Handle an MCP notification."""
        # Currently, we mainly handle resource subscriptions
        if notification.method == "resources/unsubscribe":
            await self.handle_resources_unsubscribe(notification, connection_id)
        # Other notifications can be handled here as needed

    # Core MCP handlers
    async def handle_initialize(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle initialize request."""
        # Store client capabilities
        self.active_connections[connection_id]["client_info"] = request.params

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True
                },
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                },
                "prompts": {
                    "listChanged": True
                },
                "logging": {},
                "templateHeaven": {
                    "version": __version__,
                    "features": ["templates", "stacks", "projects", "validation"]
                }
            },
            "serverInfo": {
                "name": "template-heaven-mcp",
                "version": __version__
            }
        }

    async def handle_ping(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle ping request."""
        return {}

    async def handle_shutdown(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle shutdown request."""
        logger.info("MCP server shutdown requested")

        # Schedule graceful shutdown
        asyncio.create_task(self.graceful_shutdown())

        return {}

    async def graceful_shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown...")

        # Close all connections
        for connection_id, connection_info in self.active_connections.items():
            try:
                websocket = connection_info["websocket"]
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection {connection_id}: {e}")

        # Shutdown services
        await self.service_registry.shutdown()
        await self.middleware_manager.shutdown()

        logger.info("Graceful shutdown completed")

    # Tool handlers
    async def handle_tools_list(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = await self.service_registry.get_all_tools()
        return {"tools": tools}

    async def handle_tools_call(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name is required")

        # Route to appropriate service
        result = await self.service_registry.call_tool(tool_name, arguments, connection_id)

        return result

    # Resource handlers
    async def handle_resources_list(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle resources/list request."""
        resources = await self.service_registry.get_all_resources()
        return {"resources": resources}

    async def handle_resources_read(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = request.params.get("uri")

        if not uri:
            raise ValueError("Resource URI is required")

        # Route to appropriate service
        result = await self.service_registry.read_resource(uri, connection_id)

        return result

    async def handle_resources_subscribe(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle resources/subscribe request."""
        uri = request.params.get("uri")

        if not uri:
            raise ValueError("Resource URI is required")

        # Add subscription
        self.active_connections[connection_id]["subscriptions"].add(uri)

        return {}

    async def handle_resources_unsubscribe(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle resources/unsubscribe request."""
        uri = request.params.get("uri")

        if not uri:
            raise ValueError("Resource URI is required")

        # Remove subscription
        if uri in self.active_connections[connection_id]["subscriptions"]:
            self.active_connections[connection_id]["subscriptions"].remove(uri)

        return {}

    # Prompt handlers
    async def handle_prompts_list(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle prompts/list request."""
        prompts = await self.service_registry.get_all_prompts()
        return {"prompts": prompts}

    async def handle_prompts_get(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle prompts/get request."""
        prompt_name = request.params.get("name")
        arguments = request.params.get("arguments", {})

        if not prompt_name:
            raise ValueError("Prompt name is required")

        # Route to appropriate service
        result = await self.service_registry.get_prompt(prompt_name, arguments, connection_id)

        return result

    # Template Heaven specific handlers
    async def handle_templates_list(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle template-heaven/templates/list request."""
        stack = request.params.get("stack")
        category = request.params.get("category")

        templates = await self.service_registry.list_templates(stack=stack, category=category)
        return {"templates": templates}

    async def handle_templates_generate(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle template-heaven/templates/generate request."""
        template_name = request.params.get("template")
        destination = request.params.get("destination")
        options = request.params.get("options", {})

        if not template_name or not destination:
            raise ValueError("Template name and destination are required")

        result = await self.service_registry.generate_template(
            template_name, destination, options, connection_id
        )

        return result

    async def handle_templates_validate(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle template-heaven/templates/validate request."""
        template_path = request.params.get("path")
        rules = request.params.get("rules", [])

        if not template_path:
            raise ValueError("Template path is required")

        result = await self.service_registry.validate_template(template_path, rules, connection_id)

        return result

    async def handle_stacks_list(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle template-heaven/stacks/list request."""
        stacks = await self.service_registry.list_stacks()
        return {"stacks": stacks}

    async def handle_projects_create(self, request: MCPRequest, connection_id: str) -> Dict[str, Any]:
        """Handle template-heaven/projects/create request."""
        name = request.params.get("name")
        template = request.params.get("template")
        options = request.params.get("options", {})

        if not name or not template:
            raise ValueError("Project name and template are required")

        result = await self.service_registry.create_project(name, template, options, connection_id)

        return result

    async def cleanup_subscriptions(self, connection_id: str):
        """Clean up subscriptions for a disconnected client."""
        if connection_id in self.active_connections:
            subscriptions = self.active_connections[connection_id]["subscriptions"]
            # Notify services about subscription cleanup
            for uri in subscriptions:
                await self.service_registry.unsubscribe_resource(uri, connection_id)

    async def broadcast_notification(self, notification: MCPNotification):
        """Broadcast notification to all subscribed clients."""
        for connection_id, connection_info in self.active_connections.items():
            try:
                websocket = connection_info["websocket"]
                await websocket.send(notification.to_json())
            except Exception as e:
                logger.error(f"Failed to send notification to {connection_id}: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "total_subscriptions": sum(
                len(conn["subscriptions"]) for conn in self.active_connections.values()
            )
        }
