"""
Custom exceptions for the application.

This module defines custom exception classes for better error handling
and API responses.
"""

from typing import Optional, Dict, Any, List


class BaseAPIException(Exception):
    """Base exception class for API errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)


class ValidationError(BaseAPIException):
    """Exception raised for validation errors."""
    
    def __init__(
        self,
        message: str = "Validation error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "VALIDATION_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=400,
            details=details,
            error_code=error_code
        )


class AuthenticationError(BaseAPIException):
    """Exception raised for authentication errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "AUTHENTICATION_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=401,
            details=details,
            error_code=error_code
        )


class AuthorizationError(BaseAPIException):
    """Exception raised for authorization errors."""
    
    def __init__(
        self,
        message: str = "Access denied",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "AUTHORIZATION_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=403,
            details=details,
            error_code=error_code
        )


class NotFoundError(BaseAPIException):
    """Exception raised when a resource is not found."""
    
    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "NOT_FOUND"
    ):
        super().__init__(
            message=message,
            status_code=404,
            details=details,
            error_code=error_code
        )


class ConflictError(BaseAPIException):
    """Exception raised for resource conflicts."""
    
    def __init__(
        self,
        message: str = "Resource conflict",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "CONFLICT"
    ):
        super().__init__(
            message=message,
            status_code=409,
            details=details,
            error_code=error_code
        )


class InternalServerError(BaseAPIException):
    """Exception raised for internal server errors."""
    
    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "INTERNAL_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
            error_code=error_code
        )


class DatabaseError(BaseAPIException):
    """Exception raised for database errors."""
    
    def __init__(
        self,
        message: str = "Database error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "DATABASE_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
            error_code=error_code
        )


class ExternalServiceError(BaseAPIException):
    """Exception raised for external service errors."""
    
    def __init__(
        self,
        message: str = "External service error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "EXTERNAL_SERVICE_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=502,
            details=details,
            error_code=error_code
        )


class RateLimitError(BaseAPIException):
    """Exception raised for rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "RATE_LIMIT_EXCEEDED"
    ):
        super().__init__(
            message=message,
            status_code=429,
            details=details,
            error_code=error_code
        )


class FileUploadError(BaseAPIException):
    """Exception raised for file upload errors."""
    
    def __init__(
        self,
        message: str = "File upload error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "FILE_UPLOAD_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=400,
            details=details,
            error_code=error_code
        )


class EmailError(BaseAPIException):
    """Exception raised for email sending errors."""
    
    def __init__(
        self,
        message: str = "Email sending error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "EMAIL_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
            error_code=error_code
        )


class CacheError(BaseAPIException):
    """Exception raised for cache errors."""
    
    def __init__(
        self,
        message: str = "Cache error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "CACHE_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
            error_code=error_code
        )


class ConfigurationError(BaseAPIException):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "CONFIGURATION_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
            error_code=error_code
        )


class BusinessLogicError(BaseAPIException):
    """Exception raised for business logic errors."""
    
    def __init__(
        self,
        message: str = "Business logic error",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "BUSINESS_LOGIC_ERROR"
    ):
        super().__init__(
            message=message,
            status_code=400,
            details=details,
            error_code=error_code
        )


class ValidationFieldError:
    """Represents a field validation error."""
    
    def __init__(self, field: str, message: str, code: Optional[str] = None):
        self.field = field
        self.message = message
        self.code = code or "INVALID"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "code": self.code
        }


class ValidationErrors(ValidationError):
    """Exception for multiple validation errors."""
    
    def __init__(
        self,
        errors: List[ValidationFieldError],
        message: str = "Validation errors"
    ):
        self.errors = errors
        details = {
            "errors": [error.to_dict() for error in errors]
        }
        super().__init__(
            message=message,
            details=details,
            error_code="VALIDATION_ERRORS"
        )
