"""
Custom exceptions for MCP SDK Template.

Provides specific exception types for different error scenarios in MCP
server and client operations.
"""

from typing import Optional, Dict, Any


class MCPSDKError(Exception):
    """
    Base exception for all MCP SDK errors.
    
    Attributes:
        message: Error message
        error_code: Optional error code
        details: Optional additional error details
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP SDK error.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary representation.
        
        Returns:
            Dictionary containing error information
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class MCPConnectionError(MCPSDKError):
    """
    Exception raised for MCP connection-related errors.
    
    Used when there are issues establishing or maintaining connections
    between MCP clients and servers.
    """
    
    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP connection error.
        
        Args:
            message: Error message
            host: Optional host that failed to connect
            port: Optional port that failed to connect
            details: Optional additional error details
        """
        self.host = host
        self.port = port
        error_details = details or {}
        if host:
            error_details["host"] = host
        if port:
            error_details["port"] = port
        
        super().__init__(
            message=message,
            error_code="CONNECTION_ERROR",
            details=error_details
        )


class MCPValidationError(MCPSDKError):
    """
    Exception raised for MCP validation errors.
    
    Used when MCP messages, parameters, or data fail validation.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP validation error.
        
        Args:
            message: Error message
            field: Optional field that failed validation
            value: Optional value that failed validation
            details: Optional additional error details
        """
        self.field = field
        self.value = value
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class MCPAuthenticationError(MCPSDKError):
    """
    Exception raised for MCP authentication errors.
    
    Used when authentication fails or credentials are invalid.
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP authentication error.
        
        Args:
            message: Error message
            details: Optional additional error details
        """
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class MCPAuthorizationError(MCPSDKError):
    """
    Exception raised for MCP authorization errors.
    
    Used when a user is authenticated but lacks permission for an operation.
    """
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        operation: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP authorization error.
        
        Args:
            message: Error message
            operation: Optional operation that was denied
            resource: Optional resource that was denied
            details: Optional additional error details
        """
        self.operation = operation
        self.resource = resource
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if resource:
            error_details["resource"] = resource
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            details=error_details
        )


class MCPTimeoutError(MCPSDKError):
    """
    Exception raised for MCP timeout errors.
    
    Used when operations exceed their timeout limits.
    """
    
    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP timeout error.
        
        Args:
            message: Error message
            timeout_seconds: Optional timeout duration in seconds
            operation: Optional operation that timed out
            details: Optional additional error details
        """
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        error_details = details or {}
        if timeout_seconds:
            error_details["timeout_seconds"] = timeout_seconds
        if operation:
            error_details["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details=error_details
        )


class MCPResourceError(MCPSDKError):
    """
    Exception raised for MCP resource-related errors.
    
    Used when there are issues with MCP resources (tools, prompts, etc.).
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP resource error.
        
        Args:
            message: Error message
            resource_type: Optional type of resource (tool, prompt, etc.)
            resource_name: Optional name of the resource
            details: Optional additional error details
        """
        self.resource_type = resource_type
        self.resource_name = resource_name
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_name:
            error_details["resource_name"] = resource_name
        
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            details=error_details
        )


class MCPConfigurationError(MCPSDKError):
    """
    Exception raised for MCP configuration errors.
    
    Used when there are issues with MCP server or client configuration.
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP configuration error.
        
        Args:
            message: Error message
            config_key: Optional configuration key that caused the error
            config_value: Optional configuration value that caused the error
            details: Optional additional error details
        """
        self.config_key = config_key
        self.config_value = config_value
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key
        if config_value is not None:
            error_details["config_value"] = str(config_value)
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=error_details
        )


class MCPInternalError(MCPSDKError):
    """
    Exception raised for internal MCP SDK errors.
    
    Used for unexpected internal errors that should be reported
    but not exposed to clients.
    """
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP internal error.
        
        Args:
            message: Error message
            component: Optional component where the error occurred
            details: Optional additional error details
        """
        self.component = component
        error_details = details or {}
        if component:
            error_details["component"] = component
        
        super().__init__(
            message=message,
            error_code="INTERNAL_ERROR",
            details=error_details
        )
