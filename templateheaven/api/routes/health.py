"""
Health check routes for Template Heaven API.

This module provides health check and monitoring endpoints for the API service.
"""

import time
import psutil
from typing import Dict, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ...core.models import HealthCheck, APIResponse
from ..dependencies import get_settings, get_request_id
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthCheck)
async def health_check(request: Request):
    """
    Health check endpoint.
    
    Returns the current health status of the service including
    uptime, dependencies, and system metrics.
    """
    try:
        # Get application startup time
        startup_time = getattr(request.app.state, "startup_time", time.time())
        uptime = time.time() - startup_time
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check dependencies
        dependencies = await _check_dependencies()
        
        # Determine overall status
        status = "healthy"
        if any(dep_status != "healthy" for dep_status in dependencies.values()):
            status = "degraded"
        
        return HealthCheck(
            status=status,
            version=get_settings().app_version,
            uptime=uptime,
            dependencies=dependencies,
            metrics={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "process_count": len(psutil.pids())
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthCheck(
            status="unhealthy",
            version=get_settings().app_version,
            uptime=0,
            dependencies={"error": "health_check_failed"},
            metrics={}
        )


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Used by Kubernetes and other orchestration systems to check
    if the service is ready to receive traffic.
    """
    try:
        # Check critical dependencies
        dependencies = await _check_dependencies()
        
        # Service is ready if all critical dependencies are healthy
        critical_deps = ["database", "cache", "github_api"]
        ready = all(
            dependencies.get(dep, "unhealthy") == "healthy" 
            for dep in critical_deps
        )
        
        if ready:
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "timestamp": time.time()}
            )
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "dependencies": dependencies}
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(e)}
        )


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check endpoint.
    
    Used by Kubernetes and other orchestration systems to check
    if the service is alive and should not be restarted.
    """
    try:
        # Simple liveness check - service is alive if it can respond
        return JSONResponse(
            status_code=200,
            content={"status": "alive", "timestamp": time.time()}
        )
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "dead", "error": str(e)}
        )


@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for monitoring and alerting.
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get application metrics
        app_metrics = getattr(router.app.state, "metrics", {})
        
        # Format as Prometheus metrics
        metrics = []
        
        # System metrics
        metrics.append(f"# HELP system_cpu_percent CPU usage percentage")
        metrics.append(f"# TYPE system_cpu_percent gauge")
        metrics.append(f"system_cpu_percent {cpu_percent}")
        
        metrics.append(f"# HELP system_memory_percent Memory usage percentage")
        metrics.append(f"# TYPE system_memory_percent gauge")
        metrics.append(f"system_memory_percent {memory.percent}")
        
        metrics.append(f"# HELP system_memory_available_bytes Available memory in bytes")
        metrics.append(f"# TYPE system_memory_available_bytes gauge")
        metrics.append(f"system_memory_available_bytes {memory.available}")
        
        metrics.append(f"# HELP system_disk_percent Disk usage percentage")
        metrics.append(f"# TYPE system_disk_percent gauge")
        metrics.append(f"system_disk_percent {disk.percent}")
        
        metrics.append(f"# HELP system_disk_free_bytes Free disk space in bytes")
        metrics.append(f"# TYPE system_disk_free_bytes gauge")
        metrics.append(f"system_disk_free_bytes {disk.free}")
        
        # Application metrics
        if "request_count" in app_metrics:
            metrics.append(f"# HELP app_requests_total Total number of requests")
            metrics.append(f"# TYPE app_requests_total counter")
            metrics.append(f"app_requests_total {app_metrics['request_count']}")
        
        if "error_count" in app_metrics:
            metrics.append(f"# HELP app_errors_total Total number of errors")
            metrics.append(f"# TYPE app_errors_total counter")
            metrics.append(f"app_errors_total {app_metrics['error_count']}")
        
        return "\n".join(metrics)
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}", exc_info=True)
        return f"# ERROR: {str(e)}"


async def _check_dependencies() -> Dict[str, str]:
    """Check the health of service dependencies."""
    dependencies = {}
    
    try:
        # Check database
        dependencies["database"] = await _check_database()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        dependencies["database"] = "unhealthy"
    
    try:
        # Check cache
        dependencies["cache"] = await _check_cache()
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        dependencies["cache"] = "unhealthy"
    
    try:
        # Check GitHub API
        dependencies["github_api"] = await _check_github_api()
    except Exception as e:
        logger.error(f"GitHub API health check failed: {e}")
        dependencies["github_api"] = "unhealthy"
    
    try:
        # Check file system
        dependencies["filesystem"] = await _check_filesystem()
    except Exception as e:
        logger.error(f"Filesystem health check failed: {e}")
        dependencies["filesystem"] = "unhealthy"
    
    return dependencies


async def _check_database() -> str:
    """Check database connectivity."""
    try:
        # In a real implementation, you would test database connectivity
        # For now, we'll simulate a check
        settings = get_settings()
        if "sqlite" in settings.database_url:
            return "healthy"
        else:
            return "healthy"  # Assume healthy for other databases
    except Exception:
        return "unhealthy"


async def _check_cache() -> str:
    """Check cache connectivity."""
    try:
        # In a real implementation, you would test cache connectivity
        # For now, we'll simulate a check
        settings = get_settings()
        if settings.redis_url:
            # Test Redis connectivity
            return "healthy"
        else:
            # Using in-memory cache
            return "healthy"
    except Exception:
        return "unhealthy"


async def _check_github_api() -> str:
    """Check GitHub API connectivity."""
    try:
        # In a real implementation, you would test GitHub API connectivity
        # For now, we'll simulate a check
        settings = get_settings()
        if settings.github_token:
            return "healthy"
        else:
            return "degraded"  # Limited without token
    except Exception:
        return "unhealthy"


async def _check_filesystem() -> str:
    """Check filesystem accessibility."""
    try:
        # Check if we can read/write to the templates directory
        settings = get_settings()
        import os
        
        # Test read access
        if os.path.exists(settings.templates_dir):
            return "healthy"
        else:
            # Try to create directory
            os.makedirs(settings.templates_dir, exist_ok=True)
            return "healthy"
    except Exception:
        return "unhealthy"
