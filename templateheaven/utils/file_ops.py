"""
File operations utilities for Template Heaven.

This module provides safe file operations for template copying,
customization, and project initialization.
"""

import os
import shutil
import stat
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import re

from .logger import get_logger

logger = get_logger(__name__)


class FileOperations:
    """
    Safe file operations for template management.
    
    Provides methods for copying files, creating directories,
    and handling file permissions safely.
    """
    
    def __init__(self):
        """Initialize file operations."""
        self.logger = get_logger(self.__class__.__name__)
    
    def copy_file(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        preserve_permissions: bool = True
    ) -> bool:
        """
        Copy a file safely.
        
        Args:
            src: Source file path
            dst: Destination file path
            preserve_permissions: Whether to preserve file permissions
            
        Returns:
            True if successful
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            OSError: If copy operation fails
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        
        if not src_path.is_file():
            raise ValueError(f"Source is not a file: {src_path}")
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(src_path, dst_path)
            
            if preserve_permissions:
                # Preserve file permissions
                src_stat = src_path.stat()
                dst_path.chmod(src_stat.st_mode)
            
            self.logger.debug(f"Copied file: {src_path} -> {dst_path}")
            return True
            
        except OSError as e:
            self.logger.error(f"Failed to copy file {src_path} to {dst_path}: {e}")
            raise
    
    def copy_directory(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        ignore_patterns: Optional[List[str]] = None
    ) -> bool:
        """
        Copy a directory recursively.
        
        Args:
            src: Source directory path
            dst: Destination directory path
            ignore_patterns: List of patterns to ignore (e.g., ['*.pyc', '__pycache__'])
            
        Returns:
            True if successful
            
        Raises:
            FileNotFoundError: If source directory doesn't exist
            OSError: If copy operation fails
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source directory not found: {src_path}")
        
        if not src_path.is_dir():
            raise ValueError(f"Source is not a directory: {src_path}")
        
        # Default ignore patterns
        if ignore_patterns is None:
            ignore_patterns = [
                '__pycache__',
                '*.pyc',
                '*.pyo',
                '.git',
                '.gitignore',
                '.DS_Store',
                'node_modules',
                '.venv',
                'venv',
                'env',
                '.env',
            ]
        
        def ignore_func(directory, files):
            """Function to determine which files to ignore."""
            ignored = []
            for file in files:
                file_path = Path(directory) / file
                for pattern in ignore_patterns:
                    if file_path.match(pattern) or file == pattern:
                        ignored.append(file)
                        break
            return ignored
        
        try:
            shutil.copytree(
                src_path,
                dst_path,
                ignore=ignore_func,
                dirs_exist_ok=True
            )
            
            self.logger.debug(f"Copied directory: {src_path} -> {dst_path}")
            return True
            
        except OSError as e:
            self.logger.error(f"Failed to copy directory {src_path} to {dst_path}: {e}")
            raise
    
    def create_directory(self, path: Union[str, Path], parents: bool = True) -> bool:
        """
        Create a directory safely.
        
        Args:
            path: Directory path to create
            parents: Whether to create parent directories
            
        Returns:
            True if successful
            
        Raises:
            OSError: If directory creation fails
        """
        dir_path = Path(path)
        
        try:
            dir_path.mkdir(parents=parents, exist_ok=True)
            self.logger.debug(f"Created directory: {dir_path}")
            return True
            
        except OSError as e:
            self.logger.error(f"Failed to create directory {dir_path}: {e}")
            raise
    
    def remove_file(self, path: Union[str, Path]) -> bool:
        """
        Remove a file safely.
        
        Args:
            path: File path to remove
            
        Returns:
            True if successful
        """
        file_path = Path(path)
        
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                self.logger.debug(f"Removed file: {file_path}")
            return True
            
        except OSError as e:
            self.logger.error(f"Failed to remove file {file_path}: {e}")
            return False
    
    def remove_directory(self, path: Union[str, Path]) -> bool:
        """
        Remove a directory and all its contents safely.
        
        Args:
            path: Directory path to remove
            
        Returns:
            True if successful
        """
        dir_path = Path(path)
        
        try:
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)
                self.logger.debug(f"Removed directory: {dir_path}")
            return True
            
        except OSError as e:
            self.logger.error(f"Failed to remove directory {dir_path}: {e}")
            return False
    
    def read_file(self, path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        Read a file safely.
        
        Args:
            path: File path to read
            encoding: File encoding
            
        Returns:
            File contents as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            OSError: If read operation fails
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            self.logger.debug(f"Read file: {file_path}")
            return content
            
        except OSError as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def write_file(
        self,
        path: Union[str, Path],
        content: str,
        encoding: str = 'utf-8',
        create_dirs: bool = True
    ) -> bool:
        """
        Write content to a file safely.
        
        Args:
            path: File path to write
            content: Content to write
            encoding: File encoding
            create_dirs: Whether to create parent directories
            
        Returns:
            True if successful
            
        Raises:
            OSError: If write operation fails
        """
        file_path = Path(path)
        
        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            self.logger.debug(f"Wrote file: {file_path}")
            return True
            
        except OSError as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            raise
    
    def find_files(
        self,
        directory: Union[str, Path],
        pattern: str,
        recursive: bool = True
    ) -> List[Path]:
        """
        Find files matching a pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern (e.g., '*.py', '*.json')
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return []
        
        try:
            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))
            
            # Filter to only include files (not directories)
            files = [f for f in files if f.is_file()]
            
            self.logger.debug(f"Found {len(files)} files matching '{pattern}' in {dir_path}")
            return files
            
        except OSError as e:
            self.logger.error(f"Failed to search for files in {dir_path}: {e}")
            return []
    
    def get_file_size(self, path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            path: File path
            
        Returns:
            File size in bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
            OSError: If stat operation fails
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            return file_path.stat().st_size
        except OSError as e:
            self.logger.error(f"Failed to get file size for {file_path}: {e}")
            raise
    
    def is_safe_path(self, path: Union[str, Path], base_path: Union[str, Path]) -> bool:
        """
        Check if a path is safe (within base path).
        
        Args:
            path: Path to check
            base_path: Base path to check against
            
        Returns:
            True if path is safe
        """
        try:
            path = Path(path).resolve()
            base_path = Path(base_path).resolve()
            
            # Check if path is within base path
            return base_path in path.parents or path == base_path
            
        except (OSError, ValueError):
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename for safe use.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = 'untitled'
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized
