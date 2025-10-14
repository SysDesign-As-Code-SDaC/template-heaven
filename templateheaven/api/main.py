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
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .routes import templates, stacks, search, populate, health, auth
from .middleware import RateLimitMiddleware, LoggingMiddleware, SecurityMiddleware
from .dependencies import get_settings
from ..core.models import APIResponse, HealthCheck
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Global startup time for uptime calculation
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Template Heaven API service...")
    
    # Initialize services
    settings = get_settings()
    logger.info(f"📊 Service configuration loaded: {settings.app_name}")
    
    # Health check initialization
    app.state.startup_time = startup_time
    app.state.health_status = "healthy"
    
    logger.info("✅ Template Heaven API service started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Template Heaven API service...")
    app.state.health_status = "shutting_down"
    logger.info("✅ Template Heaven API service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
        default_response_class=JSONResponse,
    )
    
    # Add middleware
    _add_middleware(app, settings)
    
    # Add routes
    _add_routes(app)
    
    # Add custom OpenAPI schema
    _customize_openapi(app, settings)
    
    return app


def _add_middleware(app: FastAPI, settings) -> None:
    """Add middleware to the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware
    if settings.trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts
        )
    
    # Custom middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)


def _add_routes(app: FastAPI) -> None:
    """Add routes to the FastAPI application."""
    
    # Health and monitoring
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    
    # Authentication
    app.include_router(auth.router, prefix="/api/v1", tags=["authentication"])
    
    # Core API routes
    app.include_router(templates.router, prefix="/api/v1", tags=["templates"])
    app.include_router(stacks.router, prefix="/api/v1", tags=["stacks"])
    app.include_router(search.router, prefix="/api/v1", tags=["search"])
    app.include_router(populate.router, prefix="/api/v1", tags=["population"])
    
    # Root endpoint
    @app.get("/", response_model=APIResponse)
    async def root():
        """Root endpoint with API information."""
        return APIResponse(
            success=True,
            message="Template Heaven API - Template Management Service",
            data={
                "service": "Template Heaven API",
                "version": get_settings().app_version,
                "status": "operational",
                "endpoints": {
                    "docs": "/docs",
                    "health": "/api/v1/health",
                    "templates": "/api/v1/templates",
                    "stacks": "/api/v1/stacks",
                    "search": "/api/v1/search",
                    "population": "/api/v1/populate"
                }
            }
        )


def _customize_openapi(app: FastAPI, settings) -> None:
    """Customize OpenAPI schema."""
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=settings.app_name,
            version=settings.app_version,
            description=settings.app_description,
            routes=app.routes,
        )
        
        # Add custom information
        openapi_schema["info"]["x-logo"] = {
            "url": "https://via.placeholder.com/200x200/4F46E5/FFFFFF?text=TH"
        }
        
        openapi_schema["info"]["contact"] = {
            "name": "Template Heaven Support",
            "email": "support@templateheaven.dev",
            "url": "https://github.com/templateheaven/templateheaven"
        }
        
        openapi_schema["info"]["license"] = {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token for API authentication"
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for service authentication"
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi


# Create the application instance
app = create_app()


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            message="Internal server error",
            errors=[str(exc)],
            request_id=getattr(request.state, "request_id", None)
        ).dict()
    )


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    import uuid
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "templateheaven.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
