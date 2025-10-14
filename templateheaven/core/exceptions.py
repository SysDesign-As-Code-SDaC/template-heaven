"""
Custom exceptions for Template Heaven.

This module defines custom exception classes for better error handling
and user feedback throughout the application.
"""

from typing import Optional, Dict, Any


class TemplateHeavenError(Exception):
    """
    Base exception class for all Template Heaven errors.
    
    Attributes:
        message: Error message
        error_code: Unique error code for programmatic handling
        details: Additional error details
        suggestion: Suggested action to resolve the error
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.suggestion = suggestion


class TemplateError(TemplateHeavenError):
    """Raised when there's an error with template operations."""
    
    def __init__(
        self,
        message: str,
        template_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "TEMPLATE_ERROR", details, suggestion)
        self.template_name = template_name


class TemplateNotFoundError(TemplateError):
    """Raised when a requested template is not found."""
    
    def __init__(
        self,
        template_name: str,
        available_templates: Optional[list] = None
    ):
        message = f"Template '{template_name}' not found"
        details = {"template_name": template_name}
        
        if available_templates:
            details["available_templates"] = available_templates[:10]  # Show first 10
            suggestion = f"Available templates: {', '.join(available_templates[:5])}"
            if len(available_templates) > 5:
                suggestion += f" and {len(available_templates) - 5} more"
        else:
            suggestion = "Use 'templateheaven list' to see available templates"
        
        super().__init__(message, "TEMPLATE_NOT_FOUND", details, suggestion)


class TemplateValidationError(TemplateError):
    """Raised when template validation fails."""
    
    def __init__(
        self,
        message: str,
        template_name: Optional[str] = None,
        validation_errors: Optional[list] = None
    ):
        details = {"validation_errors": validation_errors or []}
        suggestion = "Check template configuration and try again"
        
        super().__init__(message, "TEMPLATE_VALIDATION_ERROR", details, suggestion)


class ProjectError(TemplateHeavenError):
    """Raised when there's an error with project operations."""
    
    def __init__(
        self,
        message: str,
        project_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "PROJECT_ERROR", details, suggestion)
        self.project_name = project_name


class ProjectAlreadyExistsError(ProjectError):
    """Raised when trying to create a project that already exists."""
    
    def __init__(self, project_name: str, project_path: str):
        message = f"Project '{project_name}' already exists at {project_path}"
        details = {
            "project_name": project_name,
            "project_path": project_path
        }
        suggestion = "Choose a different project name or remove the existing directory"
        
        super().__init__(message, "PROJECT_ALREADY_EXISTS", details, suggestion)


class ProjectCreationError(ProjectError):
    """Raised when project creation fails."""
    
    def __init__(
        self,
        message: str,
        project_name: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__
        
        suggestion = "Check file permissions and disk space, then try again"
        
        super().__init__(message, "PROJECT_CREATION_ERROR", details, suggestion)


class ConfigurationError(TemplateHeavenError):
    """Raised when there's an error with configuration."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "CONFIGURATION_ERROR", details, suggestion)
        self.config_key = config_key


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        valid_values: Optional[list] = None
    ):
        details = {"config_key": config_key}
        if valid_values:
            details["valid_values"] = valid_values
            suggestion = f"Valid values: {', '.join(map(str, valid_values))}"
        else:
            suggestion = "Check the configuration documentation"
        
        super().__init__(message, "INVALID_CONFIGURATION", details, suggestion)


class CacheError(TemplateHeavenError):
    """Raised when there's an error with cache operations."""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "CACHE_ERROR", details, suggestion)
        self.cache_key = cache_key


class NetworkError(TemplateHeavenError):
    """Raised when there's a network-related error."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "NETWORK_ERROR", details, suggestion)
        self.url = url
        self.status_code = status_code


class GitHubAPIError(NetworkError):
    """Raised when there's an error with GitHub API."""
    
    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        rate_limit_info: Optional[Dict[str, Any]] = None
    ):
        details = {"rate_limit_info": rate_limit_info}
        
        if status_code == 403 and rate_limit_info:
            suggestion = "GitHub API rate limit exceeded. Try again later or use a GitHub token"
        elif status_code == 404:
            suggestion = "Repository or resource not found. Check the URL and permissions"
        elif status_code == 401:
            suggestion = "Authentication failed. Check your GitHub token"
        else:
            suggestion = "Check your internet connection and GitHub API status"
        
        super().__init__(message, url, status_code, details, suggestion)


class ValidationError(TemplateHeavenError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "VALIDATION_ERROR", details, suggestion)
        self.field_name = field_name
        self.field_value = field_value


class InvalidProjectNameError(ValidationError):
    """Raised when project name validation fails."""
    
    def __init__(self, project_name: str, reason: str):
        message = f"Invalid project name '{project_name}': {reason}"
        details = {
            "project_name": project_name,
            "reason": reason
        }
        
        if "empty" in reason.lower():
            suggestion = "Project name cannot be empty"
        elif "special" in reason.lower():
            suggestion = "Project name can only contain letters, numbers, hyphens, and underscores"
        elif "length" in reason.lower():
            suggestion = "Project name must be between 1 and 50 characters"
        else:
            suggestion = "Choose a valid project name"
        
        super().__init__(message, "project_name", project_name, details, suggestion)


class DependencyError(TemplateHeavenError):
    """Raised when there's an error with dependencies."""
    
    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "DEPENDENCY_ERROR", details, suggestion)
        self.dependency_name = dependency_name


class MissingDependencyError(DependencyError):
    """Raised when a required dependency is missing."""
    
    def __init__(self, dependency_name: str, install_command: Optional[str] = None):
        message = f"Required dependency '{dependency_name}' is not installed"
        details = {"dependency_name": dependency_name}
        
        if install_command:
            suggestion = f"Install it with: {install_command}"
        else:
            suggestion = f"Install '{dependency_name}' and try again"
        
        super().__init__(message, dependency_name, details, suggestion)


class FileSystemError(TemplateHeavenError):
    """Raised when there's an error with file system operations."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message, "FILESYSTEM_ERROR", details, suggestion)
        self.file_path = file_path
        self.operation = operation


class PermissionError(FileSystemError):
    """Raised when there's a permission error."""
    
    def __init__(self, file_path: str, operation: str):
        message = f"Permission denied: cannot {operation} '{file_path}'"
        details = {
            "file_path": file_path,
            "operation": operation
        }
        suggestion = "Check file permissions or run with appropriate privileges"
        
        super().__init__(message, file_path, operation, details, suggestion)


class DiskSpaceError(FileSystemError):
    """Raised when there's insufficient disk space."""
    
    def __init__(self, required_space: str, available_space: str):
        message = f"Insufficient disk space. Required: {required_space}, Available: {available_space}"
        details = {
            "required_space": required_space,
            "available_space": available_space
        }
        suggestion = "Free up disk space and try again"
        
        super().__init__(message, None, "write", details, suggestion)


class UserCancelledError(TemplateHeavenError):
    """Raised when user cancels an operation."""
    
    def __init__(self, operation: str = "operation"):
        message = f"User cancelled {operation}"
        suggestion = "Operation was cancelled by user"
        
        super().__init__(message, "USER_CANCELLED", {}, suggestion)


class TimeoutError(TemplateHeavenError):
    """Raised when an operation times out."""
    
    def __init__(self, operation: str, timeout_seconds: int):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        details = {
            "operation": operation,
            "timeout_seconds": timeout_seconds
        }
        suggestion = "Try again or increase the timeout value"
        
        super().__init__(message, "TIMEOUT_ERROR", details, suggestion)


# Error handling utilities
def format_error_for_user(error: Exception) -> str:
    """
    Format an error for user-friendly display.
    
    Args:
        error: The exception to format
        
    Returns:
        Formatted error message
    """
    if isinstance(error, TemplateHeavenError):
        message = error.message
        
        if error.suggestion:
            message += f"\n\n💡 Suggestion: {error.suggestion}"
        
        if error.details and isinstance(error.details, dict):
            # Add relevant details for debugging
            if "available_templates" in error.details:
                templates = error.details["available_templates"]
                if templates:
                    message += f"\n\n📋 Available templates: {', '.join(templates)}"
        
        return message
    else:
        # For non-TemplateHeaven errors, provide a generic message
        return f"An unexpected error occurred: {str(error)}"


def get_error_help(error: Exception) -> Optional[str]:
    """
    Get help text for an error.
    
    Args:
        error: The exception
        
    Returns:
        Help text or None
    """
    if isinstance(error, TemplateHeavenError):
        return error.suggestion
    
    # Map common Python errors to helpful suggestions
    error_help_map = {
        FileNotFoundError: "The file or directory was not found. Check the path and try again.",
        PermissionError: "Permission denied. Check file permissions or run with appropriate privileges.",
        ConnectionError: "Network connection failed. Check your internet connection and try again.",
        KeyboardInterrupt: "Operation was cancelled by user.",
    }
    
    return error_help_map.get(type(error))
