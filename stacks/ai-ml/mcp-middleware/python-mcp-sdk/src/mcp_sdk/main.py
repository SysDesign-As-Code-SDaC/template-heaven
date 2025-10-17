"""
Main entry point for MCP SDK Template.

Provides command-line interface and server startup functionality.
"""

import asyncio
import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .core.server import MCPServer
from .core.client import MCPClient
from .core.config import get_settings
from .core.exceptions import MCPSDKError


# Initialize CLI app
app = typer.Typer(
    name="mcp-sdk",
    help="Python MCP SDK Template - Production Ready MCP Server and Client",
    add_completion=False
)

# Initialize console
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
) -> None:
    """
    Start the MCP server.
    
    Args:
        host: Server host address
        port: Server port
        log_level: Logging level
        debug: Enable debug mode
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        console.print(f"[bold green]Starting MCP Server on {host}:{port}[/bold green]")
        
        # Get settings and override with CLI arguments
        settings = get_settings()
        settings.mcp.host = host
        settings.mcp.port = port
        settings.mcp.log_level = log_level
        settings.mcp.debug = debug
        
        # Create and start server
        server = MCPServer(settings)
        
        # Run server
        asyncio.run(server.start())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Received interrupt signal, shutting down...[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Server error: {e}[/bold red]")
        logger.exception("Server startup failed")
        sys.exit(1)


@app.command()
def client(
    server_url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level")
) -> None:
    """
    Start the MCP client.
    
    Args:
        server_url: MCP server URL
        log_level: Logging level
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        console.print(f"[bold blue]Connecting to MCP Server at {server_url}[/bold blue]")
        
        # Create client
        client = MCPClient(server_url)
        client.configure_http(server_url)
        
        # Run client
        asyncio.run(client_runner(client))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Received interrupt signal, disconnecting...[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Client error: {e}[/bold red]")
        logger.exception("Client startup failed")
        sys.exit(1)


async def client_runner(client: MCPClient) -> None:
    """
    Run the MCP client with interactive features.
    
    Args:
        client: MCP client instance
    """
    try:
        async with client.connection():
            console.print("[green]Connected to MCP server![/green]")
            
            # List available tools
            tools = client.list_tools()
            console.print(f"[blue]Available tools: {', '.join(tools)}[/blue]")
            
            # List available resources
            resources = client.list_resources()
            console.print(f"[blue]Available resources: {', '.join(resources)}[/blue]")
            
            # List available prompts
            prompts = client.list_prompts()
            console.print(f"[blue]Available prompts: {', '.join(prompts)}[/blue]")
            
            # Interactive loop
            while True:
                try:
                    command = typer.prompt("\nEnter command (help, tools, resources, prompts, echo <message>, quit)")
                    
                    if command.lower() in ['quit', 'exit', 'q']:
                        break
                    elif command.lower() == 'help':
                        console.print("""
[bold]Available commands:[/bold]
- help: Show this help message
- tools: List available tools
- resources: List available resources  
- prompts: List available prompts
- echo <message>: Call echo tool with message
- health: Call health tool
- quit: Exit the client
                        """)
                    elif command.lower() == 'tools':
                        tools = client.list_tools()
                        console.print(f"[blue]Tools: {', '.join(tools)}[/blue]")
                    elif command.lower() == 'resources':
                        resources = client.list_resources()
                        console.print(f"[blue]Resources: {', '.join(resources)}[/blue]")
                    elif command.lower() == 'prompts':
                        prompts = client.list_prompts()
                        console.print(f"[blue]Prompts: {', '.join(prompts)}[/blue]")
                    elif command.lower() == 'health':
                        if 'health' in client.list_tools():
                            result = await client.call_tool('health', {})
                            console.print(f"[green]Health result: {result}[/green]")
                        else:
                            console.print("[red]Health tool not available[/red]")
                    elif command.startswith('echo '):
                        message = command[5:]  # Remove 'echo ' prefix
                        if 'echo' in client.list_tools():
                            result = await client.call_tool('echo', {'message': message})
                            console.print(f"[green]Echo result: {result}[/green]")
                        else:
                            console.print("[red]Echo tool not available[/red]")
                    else:
                        console.print(f"[red]Unknown command: {command}[/red]")
                        
                except Exception as e:
                    console.print(f"[red]Command error: {e}[/red]")
                    
    except Exception as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise


@app.command()
def test(
    server_url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level")
) -> None:
    """
    Run basic connectivity tests.
    
    Args:
        server_url: MCP server URL
        log_level: Logging level
    """
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        console.print(f"[bold blue]Testing MCP Server at {server_url}[/bold blue]")
        
        # Run tests
        asyncio.run(run_tests(server_url))
        
    except Exception as e:
        console.print(f"[bold red]Test error: {e}[/bold red]")
        logger.exception("Tests failed")
        sys.exit(1)


async def run_tests(server_url: str) -> None:
    """
    Run basic connectivity and functionality tests.
    
    Args:
        server_url: MCP server URL
    """
    client = MCPClient(server_url)
    client.configure_http(server_url)
    
    try:
        async with client.connection():
            console.print("[green]✓ Connected to server[/green]")
            
            # Test tool listing
            tools = client.list_tools()
            console.print(f"[green]✓ Found {len(tools)} tools: {', '.join(tools)}[/green]")
            
            # Test resource listing
            resources = client.list_resources()
            console.print(f"[green]✓ Found {len(resources)} resources: {', '.join(resources)}[/green]")
            
            # Test prompt listing
            prompts = client.list_prompts()
            console.print(f"[green]✓ Found {len(prompts)} prompts: {', '.join(prompts)}[/green]")
            
            # Test echo tool
            if 'echo' in tools:
                result = await client.call_tool('echo', {'message': 'Test message'})
                console.print(f"[green]✓ Echo tool works: {result}[/green]")
            else:
                console.print("[yellow]⚠ Echo tool not available[/yellow]")
            
            # Test health tool
            if 'health' in tools:
                result = await client.call_tool('health', {})
                console.print(f"[green]✓ Health tool works: {result}[/green]")
            else:
                console.print("[yellow]⚠ Health tool not available[/yellow]")
            
            # Test resource reading
            if resources:
                content = await client.read_resource(resources[0])
                console.print(f"[green]✓ Resource reading works: {len(content)} characters[/green]")
            else:
                console.print("[yellow]⚠ No resources available for testing[/yellow]")
            
            # Test prompt generation
            if 'help' in prompts:
                result = await client.get_prompt('help', {'topic': 'testing'})
                console.print(f"[green]✓ Prompt generation works: {len(result)} messages[/green]")
            else:
                console.print("[yellow]⚠ Help prompt not available[/yellow]")
            
            console.print("\n[bold green]All tests passed![/bold green]")
            
    except Exception as e:
        console.print(f"[red]✗ Test failed: {e}[/red]")
        raise


@app.command()
def info() -> None:
    """Show information about the MCP SDK Template."""
    settings = get_settings()
    
    console.print(f"""
[bold blue]Python MCP SDK Template[/bold blue]
[dim]Production-ready Model Context Protocol server and client[/dim]

[bold]Version:[/bold] {settings.app_version}
[bold]Environment:[/bold] {settings.environment}
[bold]Database:[/bold] {settings.database.url.split('@')[-1] if '@' in settings.database.url else 'hidden'}
[bold]Redis:[/bold] {settings.redis.url.split('@')[-1] if '@' in settings.redis.url else 'hidden'}

[bold]Features:[/bold]
• Full MCP protocol support
• Containerized deployment
• Database integration
• Authentication & security
• Monitoring & observability
• Comprehensive testing
• Development tools

[bold]Quick Start:[/bold]
• Start server: [green]mcp-sdk server[/green]
• Start client: [green]mcp-sdk client[/green]
• Run tests: [green]mcp-sdk test[/green]

[bold]Documentation:[/bold] https://templateheaven.dev/docs/mcp-sdk
[bold]Repository:[/bold] https://github.com/templateheaven/python-mcp-sdk-template
    """)


def main() -> None:
    """Main entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
