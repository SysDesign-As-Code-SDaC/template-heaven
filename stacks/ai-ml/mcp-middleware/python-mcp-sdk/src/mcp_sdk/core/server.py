"""
MCP Server implementation for the Python MCP SDK Template.

Provides a production-ready MCP server with comprehensive features including
authentication, monitoring, database integration, and extensible tool support.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from contextlib import asynccontextmanager

from mcp import Server, types
from mcp.server import stdio
from mcp.server.models import InitializationOptions
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import Settings, get_settings
from .exceptions import MCPSDKError, MCPValidationError, MCPInternalError
from .database import DatabaseManager
from .auth import AuthenticationManager
from .monitoring import MonitoringManager


logger = logging.getLogger(__name__)


class MCPServer:
    """
    Production-ready MCP Server implementation.
    
    Features:
    - Full MCP protocol support
    - Authentication and authorization
    - Database integration
    - Monitoring and observability
    - Extensible tool system
    - Health checks and metrics
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize MCP Server.
        
        Args:
            settings: Optional settings override
        """
        self.settings = settings or get_settings()
        self.server = Server(self.settings.app_name)
        self.app = FastAPI(
            title=self.settings.app_name,
            version=self.settings.app_version,
            description=self.settings.app_description
        )
        
        # Core components
        self.db_manager: Optional[DatabaseManager] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.monitoring_manager: Optional[MonitoringManager] = None
        
        # Tool registry
        self.tools: Dict[str, Callable] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, Callable] = {}
        
        # Server state
        self._running = False
        self._startup_complete = False
        
        self._setup_cors()
        self._setup_routes()
        self._setup_mcp_handlers()
    
    def _setup_cors(self) -> None:
        """Setup CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=True,
            allow_methods=self.settings.cors_methods,
            allow_headers=self.settings.cors_headers,
        )
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": self.settings.app_version,
                "environment": self.settings.environment,
                "uptime": self.monitoring_manager.get_uptime() if self.monitoring_manager else 0
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            if not self.monitoring_manager:
                raise HTTPException(status_code=503, detail="Monitoring not available")
            return self.monitoring_manager.get_metrics()
        
        @self.app.get("/tools")
        async def list_tools():
            """List available MCP tools."""
            return {
                "tools": list(self.tools.keys()),
                "count": len(self.tools)
            }
        
        @self.app.get("/resources")
        async def list_resources():
            """List available MCP resources."""
            return {
                "resources": list(self.resources.keys()),
                "count": len(self.resources)
            }
        
        @self.app.get("/prompts")
        async def list_prompts():
            """List available MCP prompts."""
            return {
                "prompts": list(self.prompts.keys()),
                "count": len(self.prompts)
            }
    
    def _setup_mcp_handlers(self) -> None:
        """Setup MCP protocol handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available tools."""
            tools = []
            for name, handler in self.tools.items():
                # Get tool metadata from handler
                metadata = getattr(handler, '__mcp_metadata__', {})
                tools.append(types.Tool(
                    name=name,
                    description=metadata.get('description', f'Tool: {name}'),
                    inputSchema=metadata.get('input_schema', {})
                ))
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            try:
                if name not in self.tools:
                    raise MCPValidationError(f"Tool '{name}' not found")
                
                handler = self.tools[name]
                result = await handler(arguments)
                
                # Convert result to MCP format
                if isinstance(result, str):
                    return [types.TextContent(type="text", text=result)]
                elif isinstance(result, list):
                    return [types.TextContent(type="text", text=str(item)) for item in result]
                else:
                    return [types.TextContent(type="text", text=str(result))]
                    
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                raise MCPInternalError(f"Tool execution failed: {str(e)}")
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            """List available resources."""
            resources = []
            for uri, data in self.resources.items():
                resources.append(types.Resource(
                    uri=uri,
                    name=data.get('name', uri),
                    description=data.get('description', f'Resource: {uri}'),
                    mimeType=data.get('mime_type', 'text/plain')
                ))
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read resource content."""
            if uri not in self.resources:
                raise MCPValidationError(f"Resource '{uri}' not found")
            
            resource = self.resources[uri]
            return resource.get('content', '')
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[types.Prompt]:
            """List available prompts."""
            prompts = []
            for name, handler in self.prompts.items():
                metadata = getattr(handler, '__mcp_metadata__', {})
                prompts.append(types.Prompt(
                    name=name,
                    description=metadata.get('description', f'Prompt: {name}'),
                    arguments=metadata.get('arguments', [])
                ))
            return prompts
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Get prompt content."""
            try:
                if name not in self.prompts:
                    raise MCPValidationError(f"Prompt '{name}' not found")
                
                handler = self.prompts[name]
                result = await handler(arguments)
                
                if isinstance(result, str):
                    return [types.TextContent(type="text", text=result)]
                elif isinstance(result, list):
                    return [types.TextContent(type="text", text=str(item)) for item in result]
                else:
                    return [types.TextContent(type="text", text=str(result))]
                    
            except Exception as e:
                logger.error(f"Error getting prompt {name}: {e}")
                raise MCPInternalError(f"Prompt generation failed: {str(e)}")
    
    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a tool with the MCP server.
        
        Args:
            name: Tool name
            handler: Async function that handles tool calls
            description: Optional tool description
            input_schema: Optional JSON schema for input validation
        """
        # Attach metadata to handler
        handler.__mcp_metadata__ = {
            'description': description or f'Tool: {name}',
            'input_schema': input_schema or {}
        }
        
        self.tools[name] = handler
        logger.info(f"Registered tool: {name}")
    
    def register_resource(
        self,
        uri: str,
        content: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: str = "text/plain"
    ) -> None:
        """
        Register a resource with the MCP server.
        
        Args:
            uri: Resource URI
            content: Resource content
            name: Optional resource name
            description: Optional resource description
            mime_type: MIME type of the resource
        """
        self.resources[uri] = {
            'content': content,
            'name': name or uri,
            'description': description or f'Resource: {uri}',
            'mime_type': mime_type
        }
        logger.info(f"Registered resource: {uri}")
    
    def register_prompt(
        self,
        name: str,
        handler: Callable,
        description: Optional[str] = None,
        arguments: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Register a prompt with the MCP server.
        
        Args:
            name: Prompt name
            handler: Async function that generates prompt content
            description: Optional prompt description
            arguments: Optional list of prompt arguments
        """
        # Attach metadata to handler
        handler.__mcp_metadata__ = {
            'description': description or f'Prompt: {name}',
            'arguments': arguments or []
        }
        
        self.prompts[name] = handler
        logger.info(f"Registered prompt: {name}")
    
    async def initialize(self) -> None:
        """Initialize server components."""
        try:
            logger.info("Initializing MCP Server...")
            
            # Initialize database
            self.db_manager = DatabaseManager(self.settings.database)
            await self.db_manager.initialize()
            
            # Initialize authentication
            self.auth_manager = AuthenticationManager(self.settings.security)
            await self.auth_manager.initialize()
            
            # Initialize monitoring
            self.monitoring_manager = MonitoringManager(self.settings.monitoring)
            await self.monitoring_manager.initialize()
            
            # Register default tools
            self._register_default_tools()
            
            # Register default resources
            self._register_default_resources()
            
            # Register default prompts
            self._register_default_prompts()
            
            self._startup_complete = True
            logger.info("MCP Server initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Server: {e}")
            raise MCPInternalError(f"Server initialization failed: {str(e)}")
    
    def _register_default_tools(self) -> None:
        """Register default tools."""
        
        async def echo_tool(arguments: Dict[str, Any]) -> str:
            """Echo tool for testing."""
            message = arguments.get('message', 'Hello, MCP!')
            return f"Echo: {message}"
        
        self.register_tool(
            name="echo",
            handler=echo_tool,
            description="Echo tool for testing MCP functionality",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    }
                }
            }
        )
        
        async def health_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
            """Health check tool."""
            return {
                "status": "healthy",
                "timestamp": asyncio.get_event_loop().time(),
                "version": self.settings.app_version
            }
        
        self.register_tool(
            name="health",
            handler=health_tool,
            description="Get server health status",
            input_schema={
                "type": "object",
                "properties": {}
            }
        )
    
    def _register_default_resources(self) -> None:
        """Register default resources."""
        
        self.register_resource(
            uri="config://server",
            content=f"""
Server Configuration:
- Name: {self.settings.app_name}
- Version: {self.settings.app_version}
- Environment: {self.settings.environment}
- Host: {self.settings.mcp.host}
- Port: {self.settings.mcp.port}
            """.strip(),
            name="Server Configuration",
            description="Current server configuration",
            mime_type="text/plain"
        )
    
    def _register_default_prompts(self) -> None:
        """Register default prompts."""
        
        async def help_prompt(arguments: Dict[str, Any]) -> str:
            """Help prompt."""
            topic = arguments.get('topic', 'general')
            return f"""
# MCP Server Help - {topic}

This is a Model Context Protocol (MCP) server providing tools, resources, and prompts.

## Available Tools:
{', '.join(self.tools.keys())}

## Available Resources:
{', '.join(self.resources.keys())}

## Available Prompts:
{', '.join(self.prompts.keys())}

For more information, visit: https://modelcontextprotocol.io
            """.strip()
        
        self.register_prompt(
            name="help",
            handler=help_prompt,
            description="Get help information about the MCP server",
            arguments=[
                {
                    "name": "topic",
                    "description": "Help topic",
                    "required": False
                }
            ]
        )
    
    async def start(self) -> None:
        """Start the MCP server."""
        if self._running:
            logger.warning("Server is already running")
            return
        
        try:
            if not self._startup_complete:
                await self.initialize()
            
            logger.info(f"Starting MCP Server on {self.settings.mcp.host}:{self.settings.mcp.port}")
            
            # Start FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host=self.settings.mcp.host,
                port=self.settings.mcp.port,
                log_level=self.settings.mcp.log_level.lower(),
                access_log=self.settings.monitoring.log_requests
            )
            server = uvicorn.Server(config)
            
            self._running = True
            
            # Start server
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start MCP Server: {e}")
            self._running = False
            raise MCPInternalError(f"Server start failed: {str(e)}")
    
    async def stop(self) -> None:
        """Stop the MCP server."""
        if not self._running:
            logger.warning("Server is not running")
            return
        
        try:
            logger.info("Stopping MCP Server...")
            
            # Cleanup components
            if self.monitoring_manager:
                await self.monitoring_manager.cleanup()
            
            if self.db_manager:
                await self.db_manager.cleanup()
            
            self._running = False
            logger.info("MCP Server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP Server: {e}")
            raise MCPInternalError(f"Server stop failed: {str(e)}")
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
    
    @property
    def is_initialized(self) -> bool:
        """Check if server is initialized."""
        return self._startup_complete


async def main():
    """Main entry point for MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = MCPServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
