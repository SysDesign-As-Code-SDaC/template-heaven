"""
Main FastAPI application for Template Heaven.

This module provides the main FastAPI application with all routes,
middleware, and configuration for the Template Heaven API service.
"""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .routes import health, templates_new, search_new, stacks, populate, auth
from .middleware import RateLimitMiddleware
from .dependencies import get_settings
from ..core.models import APIResponse, HealthCheck
from ..utils.logger import get_logger
from ..database.connection import init_database, close_database

logger = get_logger(__name__)

# Global startup time for uptime calculation
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Template Heaven API service...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database initialized successfully")
        
        # Initialize services
        settings = get_settings()
        logger.info(f"📊 Service configuration loaded: {settings.app_name}")
        
        # Health check initialization
        app.state.startup_time = startup_time
        app.state.health_status = "healthy"
        
        logger.info("✅ Template Heaven API service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        app.state.health_status = "unhealthy"
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Template Heaven API service...")
    try:
        await close_database()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
    
    logger.info("✅ Template Heaven API service shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="Template Heaven API",
    version="1.0.0",
    description="API for managing and discovering project templates across various technology stacks.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting Middleware
app.add_middleware(RateLimitMiddleware)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            message="Internal server error",
            data={"error": str(exc) if settings.debug else "An unexpected error occurred"}
        ).dict()
    )

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(templates_new.router, prefix="/api/v1", tags=["Templates"])
app.include_router(search_new.router, prefix="/api/v1", tags=["Search"])
app.include_router(stacks.router, prefix="/api/v1", tags=["Stacks"])
app.include_router(populate.router, prefix="/api/v1", tags=["Population"])
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with basic service information."""
    return {
        "message": "Welcome to Template Heaven API!",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }

# Health check endpoints for Kubernetes/Load Balancers
@app.get("/health/live", summary="Liveness Probe", tags=["Health"])
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "live"}

@app.get("/health/ready", summary="Readiness Probe", tags=["Health"])
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    if app.state.health_status == "healthy":
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready"}
        )

# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema with additional metadata."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Template Heaven API",
        version="1.0.0",
        description="""
        ## Template Heaven API
        
        A comprehensive API for discovering, managing, and deploying project templates
        across 24+ technology stacks.
        
        ### Features
        
        * **Template Management**: CRUD operations for project templates
        * **Search & Discovery**: Advanced search with filtering and sorting
        * **Stack Organization**: Templates organized by technology stacks
        * **GitHub Integration**: Sync templates from GitHub repositories
        * **Authentication**: JWT-based authentication with role-based access
        * **Rate Limiting**: Built-in rate limiting for API protection
        * **Monitoring**: Health checks and metrics endpoints
        
        ### Authentication
        
        Most endpoints require authentication. Use the `/api/v1/auth/login` endpoint
        to obtain a JWT token, then include it in the Authorization header:
        
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        ### Rate Limiting
        
        API requests are rate limited to 100 requests per minute per IP address.
        Rate limit headers are included in responses.
        """,
        routes=app.routes,
    )
    
    # Add additional metadata
    openapi_schema["info"]["contact"] = {
        "name": "Template Heaven Team",
        "email": "support@templateheaven.dev",
        "url": "https://templateheaven.dev"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with Template Heaven branding."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Template Heaven API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "templateheaven.api.main_new:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
