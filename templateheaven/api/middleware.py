"""
Custom middleware for Template Heaven API.

This module provides custom middleware for security, logging, rate limiting,
and other cross-cutting concerns.
"""

import time
import uuid
from typing import Callable, Dict, Any
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.models import APIResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and protection."""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        
        # Add request ID if not present
        if not hasattr(request.state, "request_id"):
            request.state.request_id = str(uuid.uuid4())
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content=APIResponse(
                    success=False,
                    message="Request too large",
                    errors=[f"Maximum request size is {self.max_request_size} bytes"]
                ).dict()
            )
        
        # Add security headers
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Request-ID"] = request.state.request_id
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for request/response logging."""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through logging middleware."""
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "response_size": response.headers.get("content-length", "unknown")
                }
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time": process_time
                },
                exc_info=True
            )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting middleware."""
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=APIResponse(
                    success=False,
                    message="Rate limit exceeded",
                    errors=[f"Maximum {self.requests_per_minute} requests per minute allowed"]
                ).dict(),
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Use authenticated user if available
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Fallback to IP address
        return f"ip:{self._get_client_ip(request)}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < minute_ago:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) >= self.requests_per_minute:
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    
    def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < minute_ago:
            client_requests.popleft()
        
        return max(0, self.requests_per_minute - len(client_requests))


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics collection middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through metrics middleware."""
        
        # Increment request count
        self.request_count += 1
        
        # Start timing
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Keep only last 1000 response times
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
            return response
            
        except Exception as e:
            # Increment error count
            self.error_count += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        avg_response_time = 0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count),
            "average_response_time": avg_response_time,
            "recent_response_times": self.response_times[-10:] if self.response_times else []
        }
