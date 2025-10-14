"""
Helper utilities for Template Heaven.

This module provides common helper functions used throughout
the application for formatting, validation, and other utilities.
"""

import re
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    if i == 0:
        return f"{int(size)} {size_names[i]}"
    else:
        return f"{size:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2m 30s")
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    
    minutes = int(seconds // 60)
    seconds = seconds % 60
    
    if minutes == 0:
        return f"{seconds:.1f}s"
    elif minutes < 60:
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"


def validate_project_name(name: str) -> bool:
    """
    Validate project name for safety and compatibility.
    
    Args:
        name: Project name to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If name is invalid
    """
    if not name:
        raise ValueError("Project name cannot be empty")
    
    if len(name) > 50:
        raise ValueError("Project name must be 50 characters or less")
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError("Project name can only contain letters, numbers, hyphens, and underscores")
    
    # Check for reserved names
    reserved_names = {
        'con', 'prn', 'aux', 'nul',
        'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9',
        'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9',
        'node_modules', '.git', '.svn', '.hg',
    }
    
    if name.lower() in reserved_names:
        raise ValueError(f"'{name}' is a reserved name")
    
    # Check for common problematic patterns
    if name.startswith('.') or name.endswith('.'):
        raise ValueError("Project name cannot start or end with a dot")
    
    if name.startswith('-') or name.endswith('-'):
        raise ValueError("Project name cannot start or end with a hyphen")
    
    return True


def sanitize_project_name(name: str) -> str:
    """
    Sanitize project name to make it safe.
    
    Args:
        name: Original project name
        
    Returns:
        Sanitized project name
    """
    # Convert to lowercase and replace spaces with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', name.lower())
    
    # Remove multiple consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)
    
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'project'
    
    # Limit length
    if len(sanitized) > 50:
        sanitized = sanitized[:50].rstrip('-')
    
    return sanitized


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL format
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def extract_github_info(url: str) -> Optional[Dict[str, str]]:
    """
    Extract GitHub repository information from URL.
    
    Args:
        url: GitHub URL
        
    Returns:
        Dictionary with owner and repo, or None if invalid
    """
    # GitHub URL patterns
    patterns = [
        r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$',
        r'git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            return {
                'owner': match.group(1),
                'repo': match.group(2),
                'url': url
            }
    
    return None


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
    """
    Get relative path from base directory.
    
    Args:
        path: Target path
        base: Base directory
        
    Returns:
        Relative path string
    """
    try:
        path = Path(path).resolve()
        base = Path(base).resolve()
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def ensure_path_exists(path: Union[str, Path]) -> Path:
    """
    Ensure path exists, creating directories if necessary.
    
    Args:
        path: Path to ensure exists
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries, with dict2 taking precedence.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Deep merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_list(items: List[str], conjunction: str = "and") -> str:
    """
    Format list of items as a readable string.
    
    Args:
        items: List of items
        conjunction: Conjunction to use (and, or, etc.)
        
    Returns:
        Formatted string
    """
    if not items:
        return ""
    elif len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    else:
        return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def retry_on_exception(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry function on exception with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay * (2 ** attempt))
            else:
                break
    
    raise last_exception


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename
        
    Returns:
        File extension (including dot)
    """
    return Path(filename).suffix


def is_binary_file(filepath: Union[str, Path]) -> bool:
    """
    Check if file is binary by reading first few bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        True if file appears to be binary
    """
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except (OSError, IOError):
        return False


def normalize_line_endings(text: str, target: str = '\n') -> str:
    """
    Normalize line endings in text.
    
    Args:
        text: Text to normalize
        target: Target line ending
        
    Returns:
        Text with normalized line endings
    """
    # Replace all line ending variations with target
    text = re.sub(r'\r\n|\r|\n', target, text)
    return text
