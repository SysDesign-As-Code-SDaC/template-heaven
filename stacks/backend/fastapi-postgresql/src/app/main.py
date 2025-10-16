"""
FastAPI Main Application

This module contains the main FastAPI application with all routes,
middleware, and configuration.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from prometheus_fastapi_instrumentator import Instrumentator

from .api.v1.api import api_router
from .core.config import get_settings, Settings
from .core.database import create_db_and_tables
from .utils.logging import setup_logging
from .utils.metrics import setup_metrics

# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)

# Global settings
settings = get_settings()

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown events including:
    - Database initialization
    - Metrics setup
    - Background tasks
    - Cleanup operations
    """
    logger.info("Starting FastAPI PostgreSQL Backend", version=__version__)

    # Startup tasks
    await create_db_and_tables()

    # Setup metrics
    instrumentator = Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    logger.info("Application startup complete")

    yield

    # Shutdown tasks
    logger.info("Application shutdown initiated")
    # Add any cleanup tasks here
    logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add trusted host middleware
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Content Security Policy (customize as needed)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'"
    )

    return response

# Rate limiting middleware (simplified version)
request_counts = {}

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    if not settings.DEBUG:  # Only apply in production
        client_ip = request.client.host if request.client else "unknown"

        # Simple in-memory rate limiting (use Redis for production)
        current_time = request.url.path  # Use path as key for simplicity
        if client_ip in request_counts:
            count, timestamp = request_counts[client_ip]
            if count >= settings.RATE_LIMIT_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
            request_counts[client_ip] = (count + 1, timestamp)
        else:
            request_counts[client_ip] = (1, 0)

    response = await call_next(request)
    return response

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# Health check endpoints
@app.get("/health/live", tags=["health"])
async def liveness_check():
    """Liveness probe endpoint."""
    return {"status": "alive", "service": settings.PROJECT_NAME}

@app.get("/health/ready", tags=["health"])
async def readiness_check():
    """Readiness probe endpoint."""
    # Add database connectivity check here
    return {"status": "ready", "service": settings.PROJECT_NAME}

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health/live"
    }

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with structured logging."""
    logger.error(
        "Unexpected exception occurred",
        exc_info=True,
        path=request.url.path,
        method=request.method
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=None,  # Use our custom logging
    )
