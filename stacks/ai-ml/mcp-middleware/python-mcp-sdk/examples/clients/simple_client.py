"""
Simple MCP Client Example

Demonstrates basic MCP client functionality including connection,
tool calling, resource reading, and prompt generation.
"""

import asyncio
import logging
from typing import Dict, Any

from mcp import ClientSession, types
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def stdio_client_example():
    """Example using stdio connection."""
    print("=== Stdio Client Example ===")
    
    # Configure stdio connection
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_sdk.main", "server"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()
                print("✓ Connected to MCP server via stdio")
                
                # List available tools
                tools_response = await session.list_tools()
                print(f"✓ Available tools: {[tool.name for tool in tools_response.tools]}")
                
                # Call echo tool
                if any(tool.name == "echo" for tool in tools_response.tools):
                    result = await session.call_tool("echo", {"message": "Hello from stdio client!"})
                    print(f"✓ Echo result: {result.content[0].text if result.content else 'No content'}")
                
                # List available resources
                resources_response = await session.list_resources()
                print(f"✓ Available resources: {[resource.uri for resource in resources_response.resources]}")
                
                # Read a resource
                if resources_response.resources:
                    resource = resources_response.resources[0]
                    content = await session.read_resource(resource.uri)
                    print(f"✓ Resource content ({resource.uri}): {content.contents[0].text[:100]}...")
                
                # List available prompts
                prompts_response = await session.list_prompts()
                print(f"✓ Available prompts: {[prompt.name for prompt in prompts_response.prompts]}")
                
                # Get help prompt
                if any(prompt.name == "help" for prompt in prompts_response.prompts):
                    prompt_result = await session.get_prompt("help", {"topic": "stdio_client"})
                    print(f"✓ Help prompt: {prompt_result.messages[0].content[0].text[:100]}...")
                
    except Exception as e:
        print(f"✗ Stdio client error: {e}")


async def http_client_example():
    """Example using HTTP connection."""
    print("\n=== HTTP Client Example ===")
    
    server_url = "http://localhost:8000"
    
    try:
        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()
                print("✓ Connected to MCP server via HTTP")
                
                # List available tools
                tools_response = await session.list_tools()
                print(f"✓ Available tools: {[tool.name for tool in tools_response.tools]}")
                
                # Call health tool
                if any(tool.name == "health" for tool in tools_response.tools):
                    result = await session.call_tool("health", {})
                    print(f"✓ Health result: {result.content[0].text if result.content else 'No content'}")
                
                # List available resources
                resources_response = await session.list_resources()
                print(f"✓ Available resources: {[resource.uri for resource in resources_response.resources]}")
                
                # Read server config resource
                config_resource = next(
                    (r for r in resources_response.resources if "config" in r.uri),
                    None
                )
                if config_resource:
                    content = await session.read_resource(config_resource.uri)
                    print(f"✓ Server config: {content.contents[0].text[:100]}...")
                
                # List available prompts
                prompts_response = await session.list_prompts()
                print(f"✓ Available prompts: {[prompt.name for prompt in prompts_response.prompts]}")
                
                # Get help prompt
                if any(prompt.name == "help" for prompt in prompts_response.prompts):
                    prompt_result = await session.get_prompt("help", {"topic": "http_client"})
                    print(f"✓ Help prompt: {prompt_result.messages[0].content[0].text[:100]}...")
                
    except Exception as e:
        print(f"✗ HTTP client error: {e}")


async def tool_execution_example():
    """Example of tool execution with error handling."""
    print("\n=== Tool Execution Example ===")
    
    server_url = "http://localhost:8000"
    
    try:
        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✓ Connected to MCP server")
                
                # Test valid tool call
                try:
                    result = await session.call_tool("echo", {"message": "Valid tool call"})
                    print(f"✓ Valid tool call result: {result.content[0].text if result.content else 'No content'}")
                except Exception as e:
                    print(f"✗ Valid tool call failed: {e}")
                
                # Test invalid tool call
                try:
                    result = await session.call_tool("nonexistent_tool", {})
                    print(f"✗ Invalid tool call should have failed: {result}")
                except Exception as e:
                    print(f"✓ Invalid tool call correctly failed: {e}")
                
                # Test tool call with invalid arguments
                try:
                    result = await session.call_tool("echo", {"invalid_arg": "test"})
                    print(f"✓ Tool call with invalid args handled: {result.content[0].text if result.content else 'No content'}")
                except Exception as e:
                    print(f"✗ Tool call with invalid args failed: {e}")
                
    except Exception as e:
        print(f"✗ Tool execution example error: {e}")


async def resource_management_example():
    """Example of resource management."""
    print("\n=== Resource Management Example ===")
    
    server_url = "http://localhost:8000"
    
    try:
        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✓ Connected to MCP server")
                
                # List all resources
                resources_response = await session.list_resources()
                print(f"✓ Found {len(resources_response.resources)} resources")
                
                # Read each resource
                for resource in resources_response.resources:
                    try:
                        content = await session.read_resource(resource.uri)
                        text_content = content.contents[0].text if content.contents else ""
                        print(f"✓ Resource '{resource.uri}': {len(text_content)} characters")
                        print(f"  Preview: {text_content[:50]}...")
                    except Exception as e:
                        print(f"✗ Failed to read resource '{resource.uri}': {e}")
                
                # Test reading non-existent resource
                try:
                    content = await session.read_resource("nonexistent://resource")
                    print(f"✗ Reading non-existent resource should have failed: {content}")
                except Exception as e:
                    print(f"✓ Reading non-existent resource correctly failed: {e}")
                
    except Exception as e:
        print(f"✗ Resource management example error: {e}")


async def prompt_generation_example():
    """Example of prompt generation."""
    print("\n=== Prompt Generation Example ===")
    
    server_url = "http://localhost:8000"
    
    try:
        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✓ Connected to MCP server")
                
                # List all prompts
                prompts_response = await session.list_prompts()
                print(f"✓ Found {len(prompts_response.prompts)} prompts")
                
                # Generate each prompt
                for prompt in prompts_response.prompts:
                    try:
                        # Try with different arguments
                        arguments = {"topic": f"example_{prompt.name}"}
                        result = await session.get_prompt(prompt.name, arguments)
                        
                        if result.messages:
                            content = result.messages[0].content[0].text if result.messages[0].content else ""
                            print(f"✓ Prompt '{prompt.name}': {len(content)} characters")
                            print(f"  Preview: {content[:50]}...")
                        else:
                            print(f"✓ Prompt '{prompt.name}': No content generated")
                            
                    except Exception as e:
                        print(f"✗ Failed to generate prompt '{prompt.name}': {e}")
                
                # Test generating non-existent prompt
                try:
                    result = await session.get_prompt("nonexistent_prompt", {})
                    print(f"✗ Generating non-existent prompt should have failed: {result}")
                except Exception as e:
                    print(f"✓ Generating non-existent prompt correctly failed: {e}")
                
    except Exception as e:
        print(f"✗ Prompt generation example error: {e}")


async def main():
    """Run all examples."""
    print("Python MCP SDK Template - Client Examples")
    print("=" * 50)
    
    # Note: These examples assume a running MCP server
    # Start the server first with: python -m mcp_sdk.main server
    
    # Run HTTP examples (requires running server)
    await http_client_example()
    await tool_execution_example()
    await resource_management_example()
    await prompt_generation_example()
    
    # Run stdio example (starts its own server)
    await stdio_client_example()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
