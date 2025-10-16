"""
Main API Router

Combines all API endpoints into a single router with proper versioning.
"""

from fastapi import APIRouter

from .auth import router as auth_router
from .users import router as users_router

api_router = APIRouter()

# Include all API routes with proper prefixes
api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    users_router,
    prefix="/users",
    tags=["users"]
)

# Health check endpoint (available at /api/v1/health)
@api_router.get("/health")
async def api_health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "service": "FastAPI PostgreSQL Backend API",
        "version": "1.0.0"
    }
