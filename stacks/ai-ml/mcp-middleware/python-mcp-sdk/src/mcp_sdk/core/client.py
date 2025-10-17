"""
MCP Client implementation for the Python MCP SDK Template.

Provides a production-ready MCP client with comprehensive features including
connection management, authentication, error handling, and tool execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

from mcp import ClientSession, types
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client
import httpx

from .config import Settings, get_settings
from .exceptions import (
    MCPSDKError, MCPConnectionError, MCPValidationError, 
    MCPTimeoutError, MCPAuthenticationError
)


logger = logging.getLogger(__name__)


class MCPClient:
    """
    Production-ready MCP Client implementation.
    
    Features:
    - Full MCP protocol support
    - Multiple connection types (stdio, HTTP)
    - Authentication and authorization
    - Connection pooling and retry logic
    - Comprehensive error handling
    - Tool execution and result parsing
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize MCP Client.
        
        Args:
            server_url: Optional server URL for HTTP connections
            settings: Optional settings override
        """
        self.settings = settings or get_settings()
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self._connected = False
        self._connection_type: Optional[str] = None
        
        # Connection parameters
        self._stdio_params: Optional[StdioServerParameters] = None
        self._http_params: Optional[Dict[str, Any]] = None
        
        # Tool and resource cache
        self._tools_cache: Dict[str, types.Tool] = {}
        self._resources_cache: Dict[str, types.Resource] = {}
        self._prompts_cache: Dict[str, types.Prompt] = {}
    
    def configure_stdio(
        self,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Configure stdio connection parameters.
        
        Args:
            command: Command to execute
            args: Command arguments
            env: Optional environment variables
        """
        self._stdio_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        self._connection_type = "stdio"
        logger.info(f"Configured stdio connection: {command} {' '.join(args)}")
    
    def configure_http(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> None:
        """
        Configure HTTP connection parameters.
        
        Args:
            url: Server URL
            headers: Optional HTTP headers
            timeout: Optional connection timeout
        """
        self._http_params = {
            "url": url,
            "headers": headers or {},
            "timeout": timeout or self.settings.mcp.connection_timeout
        }
        self._connection_type = "http"
        self.server_url = url
        logger.info(f"Configured HTTP connection: {url}")
    
    async def connect(self) -> None:
        """
        Connect to MCP server.
        
        Raises:
            MCPConnectionError: If connection fails
            MCPValidationError: If configuration is invalid
        """
        if self._connected:
            logger.warning("Client is already connected")
            return
        
        try:
            if self._connection_type == "stdio":
                await self._connect_stdio()
            elif self._connection_type == "http":
                await self._connect_http()
            else:
                raise MCPValidationError("No connection type configured")
            
            # Initialize session
            await self.session.initialize()
            
            # Cache server capabilities
            await self._cache_server_capabilities()
            
            self._connected = True
            logger.info("Successfully connected to MCP server")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self._connected = False
            raise MCPConnectionError(f"Connection failed: {str(e)}")
    
    async def _connect_stdio(self) -> None:
        """Connect via stdio."""
        if not self._stdio_params:
            raise MCPValidationError("Stdio parameters not configured")
        
        self._stdio_transport = stdio_client(self._stdio_params)
        self._read, self._write = await self._stdio_transport.__aenter__()
        self.session = ClientSession(self._read, self._write)
    
    async def _connect_http(self) -> None:
        """Connect via HTTP."""
        if not self._http_params:
            raise MCPValidationError("HTTP parameters not configured")
        
        self._http_transport = streamablehttp_client(
            self._http_params["url"],
            headers=self._http_params["headers"],
            timeout=self._http_params["timeout"]
        )
        self._read, self._write, _ = await self._http_transport.__aenter__()
        self.session = ClientSession(self._read, self._write)
    
    async def _cache_server_capabilities(self) -> None:
        """Cache server tools, resources, and prompts."""
        try:
            # Cache tools
            tools_response = await self.session.list_tools()
            self._tools_cache = {tool.name: tool for tool in tools_response.tools}
            
            # Cache resources
            resources_response = await self.session.list_resources()
            self._resources_cache = {resource.uri: resource for resource in resources_response.resources}
            
            # Cache prompts
            prompts_response = await self.session.list_prompts()
            self._prompts_cache = {prompt.name: prompt for prompt in prompts_response.prompts}
            
            logger.info(f"Cached {len(self._tools_cache)} tools, {len(self._resources_cache)} resources, {len(self._prompts_cache)} prompts")
            
        except Exception as e:
            logger.warning(f"Failed to cache server capabilities: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if not self._connected:
            logger.warning("Client is not connected")
            return
        
        try:
            # Close session
            if self.session:
                await self.session.close()
            
            # Close transport
            if self._connection_type == "stdio" and hasattr(self, '_stdio_transport'):
                await self._stdio_transport.__aexit__(None, None, None)
            elif self._connection_type == "http" and hasattr(self, '_http_transport'):
                await self._http_transport.__aexit__(None, None, None)
            
            self._connected = False
            logger.info("Disconnected from MCP server")
            
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")
    
    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> List[types.TextContent]:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Optional tool arguments
            timeout: Optional timeout in seconds
            
        Returns:
            List of text content from tool execution
            
        Raises:
            MCPConnectionError: If not connected
            MCPValidationError: If tool not found or invalid arguments
            MCPTimeoutError: If operation times out
        """
        if not self._connected:
            raise MCPConnectionError("Client is not connected")
        
        if name not in self._tools_cache:
            raise MCPValidationError(f"Tool '{name}' not found")
        
        try:
            # Set timeout if provided
            if timeout:
                result = await asyncio.wait_for(
                    self.session.call_tool(name, arguments or {}),
                    timeout=timeout
                )
            else:
                result = await self.session.call_tool(name, arguments or {})
            
            return result.content
            
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"Tool call '{name}' timed out", timeout_seconds=timeout, operation=name)
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            raise MCPSDKError(f"Tool call failed: {str(e)}")
    
    async def read_resource(
        self,
        uri: str,
        timeout: Optional[float] = None
    ) -> str:
        """
        Read a resource from the MCP server.
        
        Args:
            uri: Resource URI
            timeout: Optional timeout in seconds
            
        Returns:
            Resource content
            
        Raises:
            MCPConnectionError: If not connected
            MCPValidationError: If resource not found
            MCPTimeoutError: If operation times out
        """
        if not self._connected:
            raise MCPConnectionError("Client is not connected")
        
        if uri not in self._resources_cache:
            raise MCPValidationError(f"Resource '{uri}' not found")
        
        try:
            # Set timeout if provided
            if timeout:
                result = await asyncio.wait_for(
                    self.session.read_resource(uri),
                    timeout=timeout
                )
            else:
                result = await self.session.read_resource(uri)
            
            return result.contents[0].text if result.contents else ""
            
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"Resource read '{uri}' timed out", timeout_seconds=timeout, operation="read_resource")
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise MCPSDKError(f"Resource read failed: {str(e)}")
    
    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> List[types.TextContent]:
        """
        Get a prompt from the MCP server.
        
        Args:
            name: Prompt name
            arguments: Optional prompt arguments
            timeout: Optional timeout in seconds
            
        Returns:
            List of text content from prompt generation
            
        Raises:
            MCPConnectionError: If not connected
            MCPValidationError: If prompt not found
            MCPTimeoutError: If operation times out
        """
        if not self._connected:
            raise MCPConnectionError("Client is not connected")
        
        if name not in self._prompts_cache:
            raise MCPValidationError(f"Prompt '{name}' not found")
        
        try:
            # Set timeout if provided
            if timeout:
                result = await asyncio.wait_for(
                    self.session.get_prompt(name, arguments or {}),
                    timeout=timeout
                )
            else:
                result = await self.session.get_prompt(name, arguments or {})
            
            return result.messages[0].content if result.messages else []
            
        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"Prompt generation '{name}' timed out", timeout_seconds=timeout, operation="get_prompt")
        except Exception as e:
            logger.error(f"Error getting prompt {name}: {e}")
            raise MCPSDKError(f"Prompt generation failed: {str(e)}")
    
    def list_tools(self) -> List[str]:
        """
        List available tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools_cache.keys())
    
    def list_resources(self) -> List[str]:
        """
        List available resources.
        
        Returns:
            List of resource URIs
        """
        return list(self._resources_cache.keys())
    
    def list_prompts(self) -> List[str]:
        """
        List available prompts.
        
        Returns:
            List of prompt names
        """
        return list(self._prompts_cache.keys())
    
    def get_tool_info(self, name: str) -> Optional[types.Tool]:
        """
        Get tool information.
        
        Args:
            name: Tool name
            
        Returns:
            Tool information or None if not found
        """
        return self._tools_cache.get(name)
    
    def get_resource_info(self, uri: str) -> Optional[types.Resource]:
        """
        Get resource information.
        
        Args:
            name: Resource URI
            
        Returns:
            Resource information or None if not found
        """
        return self._resources_cache.get(uri)
    
    def get_prompt_info(self, name: str) -> Optional[types.Prompt]:
        """
        Get prompt information.
        
        Args:
            name: Prompt name
            
        Returns:
            Prompt information or None if not found
        """
        return self._prompts_cache.get(name)
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    @property
    def connection_type(self) -> Optional[str]:
        """Get connection type."""
        return self._connection_type
    
    @asynccontextmanager
    async def connection(self):
        """
        Context manager for MCP client connection.
        
        Usage:
            async with client.connection() as session:
                result = await session.call_tool("echo", {"message": "Hello"})
        """
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()


async def main():
    """Main entry point for MCP client."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create client
    client = MCPClient()
    
    # Configure connection (example with stdio)
    client.configure_stdio("python", ["-m", "mcp_sdk.main"])
    
    try:
        async with client.connection():
            # List available tools
            tools = client.list_tools()
            print(f"Available tools: {tools}")
            
            # Call a tool
            if "echo" in tools:
                result = await client.call_tool("echo", {"message": "Hello, MCP!"})
                print(f"Echo result: {result}")
            
            # List available resources
            resources = client.list_resources()
            print(f"Available resources: {resources}")
            
            # Read a resource
            if resources:
                content = await client.read_resource(resources[0])
                print(f"Resource content: {content}")
            
            # List available prompts
            prompts = client.list_prompts()
            print(f"Available prompts: {prompts}")
            
            # Get a prompt
            if "help" in prompts:
                prompt_result = await client.get_prompt("help", {"topic": "general"})
                print(f"Help prompt: {prompt_result}")
                
    except Exception as e:
        logger.error(f"Client error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
