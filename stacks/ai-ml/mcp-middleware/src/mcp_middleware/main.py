"""
Main FastAPI application for MCP Middleware.

This module provides the core FastAPI application that serves as a middleware
for MCP (Model Context Protocol) servers, allowing AI assistants to communicate
with external tools and data sources.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .core.server_manager import ServerManager
from .core.protocol_handler import ProtocolHandler
from .middleware.auth import get_current_user
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import RequestLoggingMiddleware
from .utils.config import settings
from .utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Global instances
server_manager = ServerManager()
protocol_handler = ProtocolHandler(server_manager)
metrics_collector = MetricsCollector()

# Security
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MCP Middleware server")
    await server_manager.initialize()
    await metrics_collector.start()

    yield

    # Shutdown
    logger.info("Shutting down MCP Middleware server")
    await server_manager.shutdown()
    await metrics_collector.stop()

# Create FastAPI application
app = FastAPI(
    title="MCP Middleware",
    description="Unified interface for MCP (Model Context Protocol) servers",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = await server_manager.get_health_status()
    metrics_collector.record_request("/health")

    return {
        "status": "healthy" if health_status["overall"] else "unhealthy",
        "timestamp": health_status["timestamp"],
        "servers": health_status["servers"],
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return metrics_collector.get_metrics()

# Server management endpoints
@app.get("/api/servers")
async def list_servers(user: dict = Depends(get_current_user)):
    """List all configured MCP servers."""
    metrics_collector.record_request("/api/servers")
    servers = await server_manager.list_servers()
    return {"servers": servers}

@app.post("/api/servers")
async def add_server(server_config: dict, user: dict = Depends(get_current_user)):
    """Add a new MCP server configuration."""
    metrics_collector.record_request("/api/servers", method="POST")

    try:
        result = await server_manager.add_server(server_config)
        logger.info(f"Added MCP server: {server_config.get('name')}")
        return result
    except Exception as e:
        logger.error(f"Failed to add MCP server: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/servers/{server_name}")
async def remove_server(server_name: str, user: dict = Depends(get_current_user)):
    """Remove an MCP server configuration."""
    metrics_collector.record_request(f"/api/servers/{server_name}", method="DELETE")

    try:
        result = await server_manager.remove_server(server_name)
        logger.info(f"Removed MCP server: {server_name}")
        return result
    except Exception as e:
        logger.error(f"Failed to remove MCP server {server_name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/servers/health")
async def server_health():
    """Get health status of all MCP servers."""
    metrics_collector.record_request("/api/servers/health")
    return await server_manager.get_health_status()

# MCP protocol endpoints
@app.post("/mcp/initialize")
async def mcp_initialize(request: dict):
    """Initialize MCP connection."""
    metrics_collector.record_request("/mcp/initialize", method="POST")

    try:
        response = await protocol_handler.initialize_connection(request)
        return response
    except Exception as e:
        logger.error(f"MCP initialization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/tools/list")
async def mcp_list_tools(request: dict):
    """List available MCP tools."""
    metrics_collector.record_request("/mcp/tools/list", method="POST")

    try:
        response = await protocol_handler.list_tools(request)
        return response
    except Exception as e:
        logger.error(f"MCP tools list failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/tools/call")
async def mcp_call_tool(request: dict):
    """Call an MCP tool."""
    metrics_collector.record_request("/mcp/tools/call", method="POST")

    try:
        response = await protocol_handler.call_tool(request)
        return response
    except Exception as e:
        logger.error(f"MCP tool call failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/resources/list")
async def mcp_list_resources(request: dict):
    """List MCP resources."""
    metrics_collector.record_request("/mcp/resources/list", method="POST")

    try:
        response = await protocol_handler.list_resources(request)
        return response
    except Exception as e:
        logger.error(f"MCP resources list failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/mcp/resources/read")
async def mcp_read_resource(request: dict):
    """Read an MCP resource."""
    metrics_collector.record_request("/mcp/resources/read", method="POST")

    try:
        response = await protocol_handler.read_resource(request)
        return response
    except Exception as e:
        logger.error(f"MCP resource read failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Generic MCP proxy endpoint
@app.post("/api/mcp/{server_name}")
async def proxy_mcp_request(server_name: str, request: dict, user: dict = Depends(get_current_user)):
    """Proxy MCP request to specific server."""
    metrics_collector.record_request(f"/api/mcp/{server_name}", method="POST")

    try:
        response = await protocol_handler.proxy_request(server_name, request)
        return response
    except Exception as e:
        logger.error(f"MCP proxy request to {server_name} failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mcp_middleware.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )
