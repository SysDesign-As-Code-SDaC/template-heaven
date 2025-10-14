"""
Version information for Template Heaven package.

This module provides version information for the Template Heaven package.
The version follows Semantic Versioning (SemVer) format: MAJOR.MINOR.PATCH

Version History:
- 0.1.0: Initial MVP release with core CLI functionality
"""

__version__ = "0.1.0"
__version_info__ = tuple(map(int, __version__.split('.')))

# Version metadata
VERSION = __version__
VERSION_INFO = __version_info__

# Build information (set during CI/CD)
BUILD_DATE = None
BUILD_COMMIT = None
BUILD_BRANCH = None

def get_version() -> str:
    """
    Get the current version string.
    
    Returns:
        Version string in format 'MAJOR.MINOR.PATCH'
    """
    return __version__

def get_version_info() -> tuple:
    """
    Get the current version as a tuple.
    
    Returns:
        Tuple of (major, minor, patch) version numbers
    """
    return __version_info__

def get_build_info() -> dict:
    """
    Get build information if available.
    
    Returns:
        Dictionary with build metadata
    """
    return {
        "version": __version__,
        "build_date": BUILD_DATE,
        "build_commit": BUILD_COMMIT,
        "build_branch": BUILD_BRANCH,
    }
