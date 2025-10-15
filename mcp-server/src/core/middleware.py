"""
Middleware System

Handles cross-cutting concerns for MCP requests including authentication,
rate limiting, logging, monitoring, and security.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from collections import defaultdict, deque
import asyncio
from datetime import datetime, timedelta

from .config import MCPConfig
from .protocol import MCPMessage, MCPRequest, MCPResponse, MCPError, MCPErrorCode

logger = logging.getLogger(__name__)


class MiddlewareManager:
    """
    Manages middleware components for MCP request processing.

    Middleware components handle cross-cutting concerns like:
    - Authentication and authorization
    - Rate limiting
    - Request logging and monitoring
    - Input validation and sanitization
    - Error handling
    - Caching
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self.middlewares: Dict[str, 'MCPMiddleware'] = {}
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Initialize core middlewares
        self._initialize_middlewares()

    def _initialize_middlewares(self):
        """Initialize all middleware components."""
        self.middlewares["logging"] = LoggingMiddleware(self.config)
        self.middlewares["rate_limiting"] = RateLimitingMiddleware(self.config)
        self.middlewares["authentication"] = AuthenticationMiddleware(self.config)
        self.middlewares["validation"] = ValidationMiddleware(self.config)
        self.middlewares["monitoring"] = MonitoringMiddleware(self.config)
        self.middlewares["security"] = SecurityMiddleware(self.config)

    async def initialize(self):
        """Initialize all middlewares."""
        for middleware in self.middlewares.values():
            await middleware.initialize()
        logger.info("Middleware manager initialized")

    async def shutdown(self):
        """Shutdown all middlewares."""
        for middleware in self.middlewares.values():
            await middleware.shutdown()
        logger.info("Middleware manager shutdown")

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """
        Process a request through all middlewares.

        Args:
            message: The MCP message to process
            connection_id: Unique connection identifier

        Returns:
            Processed message (may be modified or replaced)
        """
        current_message = message

        # Apply request middlewares in order
        for middleware_name in ["logging", "rate_limiting", "authentication", "validation", "security"]:
            if middleware_name in self.middlewares:
                try:
                    current_message = await self.middlewares[middleware_name].process_request(
                        current_message, connection_id
                    )

                    # If middleware returns an error response, stop processing
                    if isinstance(current_message, MCPResponse) and current_message.error:
                        break

                except Exception as e:
                    logger.error(f"Middleware {middleware_name} failed: {e}")
                    current_message = MCPError(
                        code=MCPErrorCode.INTERNAL_ERROR,
                        message=f"Middleware processing failed: {middleware_name}"
                    ).to_response(getattr(message, 'id', None))
                    break

        return current_message

    async def process_response(self, response: MCPMessage, connection_id: str) -> MCPMessage:
        """
        Process a response through middlewares.

        Args:
            response: The MCP response to process
            connection_id: Unique connection identifier

        Returns:
            Processed response
        """
        current_response = response

        # Apply response middlewares
        for middleware_name in ["monitoring", "logging"]:
            if middleware_name in self.middlewares:
                try:
                    current_response = await self.middlewares[middleware_name].process_response(
                        current_response, connection_id
                    )
                except Exception as e:
                    logger.error(f"Response middleware {middleware_name} failed: {e}")

        return current_response


class MCPMiddleware:
    """Base class for MCP middlewares."""

    def __init__(self, config: MCPConfig):
        self.config = config

    async def initialize(self):
        """Initialize the middleware."""
        pass

    async def shutdown(self):
        """Shutdown the middleware."""
        pass

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """Process a request message."""
        return message

    async def process_response(self, response: MCPMessage, connection_id: str) -> MCPMessage:
        """Process a response message."""
        return response


class LoggingMiddleware(MCPMiddleware):
    """Middleware for request/response logging."""

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """Log incoming requests."""
        if isinstance(message, MCPRequest):
            logger.info(
                "MCP Request",
                extra={
                    "connection_id": connection_id,
                    "method": message.method,
                    "request_id": message.id,
                    "params_keys": list(message.params.keys()) if message.params else []
                }
            )
        elif isinstance(message, MCPResponse):
            logger.info(
                "MCP Response",
                extra={
                    "connection_id": connection_id,
                    "response_id": message.id,
                    "has_error": message.error is not None,
                    "result_keys": list(message.result.keys()) if isinstance(message.result, dict) else None
                }
            )

        return message

    async def process_response(self, response: MCPMessage, connection_id: str) -> MCPMessage:
        """Log outgoing responses."""
        # Additional response logging can be added here
        return response


class RateLimitingMiddleware(MCPMiddleware):
    """Middleware for rate limiting."""

    def __init__(self, config: MCPConfig):
        super().__init__(config)
        self.requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.rate_limit_requests))
        self.last_cleanup = time.time()

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """Apply rate limiting."""
        if not isinstance(message, MCPRequest):
            return message

        current_time = time.time()

        # Cleanup old requests periodically
        if current_time - self.last_cleanup > 60:  # Clean up every minute
            self._cleanup_old_requests(current_time)
            self.last_cleanup = current_time

        # Add current request
        self.requests[connection_id].append(current_time)

        # Check rate limit
        if len(self.requests[connection_id]) >= self.config.rate_limit_requests:
            oldest_request = self.requests[connection_id][0]
            if current_time - oldest_request < self.config.rate_limit_window:
                logger.warning(f"Rate limit exceeded for connection {connection_id}")
                return MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message="Rate limit exceeded"
                ).to_response(message.id)

        return message

    def _cleanup_old_requests(self, current_time: float):
        """Clean up old requests from all connections."""
        cutoff_time = current_time - self.config.rate_limit_window

        for connection_id in list(self.requests.keys()):
            # Remove old requests
            while self.requests[connection_id] and self.requests[connection_id][0] < cutoff_time:
                self.requests[connection_id].popleft()

            # Remove empty deques
            if not self.requests[connection_id]:
                del self.requests[connection_id]


class AuthenticationMiddleware(MCPMiddleware):
    """Middleware for authentication and authorization."""

    def __init__(self, config: MCPConfig):
        super().__init__(config)
        self.auth_required_methods = {
            "template-heaven/templates/generate",
            "template-heaven/projects/create",
            "template-heaven/templates/validate"
        }

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """Authenticate requests."""
        if not isinstance(message, MCPRequest):
            return message

        # Check if method requires authentication
        if message.method in self.auth_required_methods:
            # Extract token from params or connection context
            token = self._extract_token(message, connection_id)

            if not token:
                return MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message="Authentication required"
                ).to_response(message.id)

            # Validate token (simplified - in production, verify with user service)
            if not self._validate_token(token):
                return MCPError(
                    code=MCPErrorCode.INTERNAL_ERROR,
                    message="Invalid authentication token"
                ).to_response(message.id)

        return message

    def _extract_token(self, message: MCPRequest, connection_id: str) -> Optional[str]:
        """Extract authentication token from request."""
        # Check params first
        if message.params and "auth_token" in message.params:
            return message.params["auth_token"]

        # Could also check connection context or headers
        # For now, return None (authentication disabled for demo)
        return "demo-token"

    def _validate_token(self, token: str) -> bool:
        """Validate authentication token."""
        # Simplified validation - in production, call user service
        return token == "demo-token"


class ValidationMiddleware(MCPMiddleware):
    """Middleware for request validation."""

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """Validate request parameters."""
        if not isinstance(message, MCPRequest):
            return message

        # Validate method name
        if not message.method or not isinstance(message.method, str):
            return MCPError(
                code=MCPErrorCode.INVALID_REQUEST,
                message="Invalid method name"
            ).to_response(message.id)

        # Validate params
        if message.params is not None and not isinstance(message.params, dict):
            return MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message="Params must be an object"
            ).to_response(message.id)

        # Method-specific validation
        validation_error = self._validate_method_params(message.method, message.params)
        if validation_error:
            return MCPError(
                code=MCPErrorCode.INVALID_PARAMS,
                message=validation_error
            ).to_response(message.id)

        return message

    def _validate_method_params(self, method: str, params: Optional[Dict[str, Any]]) -> Optional[str]:
        """Validate method-specific parameters."""
        if not params:
            return None

        # Template generation validation
        if method == "template-heaven/templates/generate":
            if "template" not in params:
                return "template parameter is required"
            if "destination" not in params:
                return "destination parameter is required"

        # Project creation validation
        elif method == "template-heaven/projects/create":
            if "name" not in params:
                return "name parameter is required"
            if "template" not in params:
                return "template parameter is required"

        return None


class MonitoringMiddleware(MCPMiddleware):
    """Middleware for monitoring and metrics."""

    def __init__(self, config: MCPConfig):
        super().__init__(config)
        self.request_count = 0
        self.error_count = 0
        self.response_times: deque = deque(maxlen=1000)

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """Record request metrics."""
        if isinstance(message, MCPRequest):
            self.request_count += 1
        return message

    async def process_response(self, response: MCPMessage, connection_id: str) -> MCPMessage:
        """Record response metrics."""
        if isinstance(response, MCPResponse):
            if response.error:
                self.error_count += 1
        return response

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0

        return {
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time": avg_response_time,
            "active_connections": len(self.response_times)  # Approximation
        }


class SecurityMiddleware(MCPMiddleware):
    """Middleware for security checks."""

    def __init__(self, config: MCPConfig):
        super().__init__(config)
        self.suspicious_patterns = [
            "../../",  # Path traversal
            "<script",  # XSS
            "javascript:",  # XSS
            "data:text/html",  # XSS
        ]

    async def process_request(self, message: MCPMessage, connection_id: str) -> MCPMessage:
        """Apply security checks."""
        if not isinstance(message, MCPRequest):
            return message

        # Check for suspicious content in params
        if message.params:
            security_issue = self._check_security(message.params)
            if security_issue:
                logger.warning(f"Security issue detected in request: {security_issue}")
                return MCPError(
                    code=MCPErrorCode.INVALID_PARAMS,
                    message="Request contains potentially malicious content"
                ).to_response(message.id)

        return message

    def _check_security(self, params: Dict[str, Any]) -> Optional[str]:
        """Check for security issues in parameters."""
        def check_value(value: Any) -> Optional[str]:
            if isinstance(value, str):
                for pattern in self.suspicious_patterns:
                    if pattern.lower() in value.lower():
                        return f"Suspicious pattern detected: {pattern}"
            elif isinstance(value, dict):
                for v in value.values():
                    issue = check_value(v)
                    if issue:
                        return issue
            elif isinstance(value, list):
                for item in value:
                    issue = check_value(item)
                    if issue:
                        return issue
            return None

        return check_value(params)
