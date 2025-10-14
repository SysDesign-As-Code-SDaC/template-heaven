"""
Configuration management for Template Heaven.

This module provides a comprehensive configuration system for managing
user settings, preferences, and system configuration stored in YAML format.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


class Config:
    """
    Configuration manager for Template Heaven.
    
    Manages user settings stored in ~/.templateheaven/config.yaml with
    support for default values, validation, and automatic directory creation.
    
    Attributes:
        config_dir: Path to configuration directory
        config_file: Path to configuration file
        cache_dir: Path to template cache directory
        _config: Internal configuration dictionary
        
    Configuration keys:
        - cache_dir: Template cache location
        - default_author: Default project author
        - default_license: Default license type
        - github_token: GitHub API token (optional)
        - auto_update: Enable automatic template updates
        - ui_theme: UI theme preference (for future web UI)
        - log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "cache_dir": "~/.templateheaven/cache",
        "default_author": None,
        "default_license": "MIT",
        "github_token": None,
        "auto_update": True,
        "ui_theme": "dark",
        "log_level": "INFO",
        "package_managers": {
            "python": "pip",
            "node": "npm",
            "rust": "cargo",
            "go": "go",
        },
        "editor_preferences": {
            "default_editor": None,
            "open_after_create": True,
        },
        "template_preferences": {
            "include_git": True,
            "include_readme": True,
            "include_license": True,
            "include_contributing": False,
        },
    }
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Custom configuration directory path (optional)
            
        Raises:
            OSError: If configuration directory cannot be created
            yaml.YAMLError: If existing config file is invalid
        """
        if config_dir is None:
            config_dir = Path.home() / ".templateheaven"
        else:
            config_dir = Path(config_dir)
        
        self.config_dir = config_dir.resolve()
        self.config_file = self.config_dir / "config.yaml"
        self.cache_dir = self.config_dir / "cache"
        self._config: Dict[str, Any] = {}
        
        # Ensure configuration directory exists
        self._ensure_config_dir()
        
        # Load configuration
        self._load_config()
    
    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Configuration directory: {self.config_dir}")
        except OSError as e:
            logger.error(f"Failed to create configuration directory: {e}")
            raise
    
    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                
                # Merge with defaults
                self._config = self.DEFAULT_CONFIG.copy()
                self._config.update(file_config)
                
                logger.debug("Configuration loaded from file")
            except (yaml.YAMLError, OSError) as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
                self._config = self.DEFAULT_CONFIG.copy()
        else:
            # Create default configuration
            self._config = self.DEFAULT_CONFIG.copy()
            self.save()
            logger.info("Created default configuration file")
    
    def save(self) -> None:
        """
        Save current configuration to file.
        
        Raises:
            OSError: If configuration file cannot be written
            yaml.YAMLError: If configuration cannot be serialized
        """
        try:
            # Add metadata
            config_to_save = self._config.copy()
            config_to_save["_metadata"] = {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0",
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=True)
            
            logger.debug("Configuration saved to file")
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            config.get('default_author')
            config.get('package_managers.python')
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
            
        Examples:
            config.set('default_author', 'John Doe')
            config.set('package_managers.python', 'poetry')
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def unset(self, key: str) -> bool:
        """
        Remove configuration key.
        
        Args:
            key: Configuration key to remove
            
        Returns:
            True if key was removed, False if key didn't exist
        """
        keys = key.split('.')
        config = self._config
        
        try:
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                config = config[k]
            
            # Remove the key
            del config[keys[-1]]
            logger.debug(f"Configuration key removed: {key}")
            return True
        except (KeyError, TypeError):
            return False
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = self.DEFAULT_CONFIG.copy()
        self.save()
        logger.info("Configuration reset to defaults")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return self._config.copy()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary values.
        
        Args:
            config_dict: Dictionary of configuration updates
        """
        self._config.update(config_dict)
        logger.debug(f"Configuration updated with {len(config_dict)} values")
    
    def validate(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Validate cache directory
        cache_dir = Path(self.get('cache_dir', '')).expanduser()
        if not cache_dir.parent.exists():
            errors.append(f"Cache directory parent does not exist: {cache_dir.parent}")
        
        # Validate log level
        log_level = self.get('log_level', 'INFO')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if log_level not in valid_levels:
            errors.append(f"Invalid log level: {log_level}. Must be one of {valid_levels}")
        
        # Validate package managers
        package_managers = self.get('package_managers', {})
        valid_managers = {
            'python': ['pip', 'poetry', 'pipenv'],
            'node': ['npm', 'yarn', 'pnpm'],
            'rust': ['cargo'],
            'go': ['go'],
        }
        
        for lang, manager in package_managers.items():
            if lang in valid_managers and manager not in valid_managers[lang]:
                errors.append(f"Invalid {lang} package manager: {manager}")
        
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return False
        
        logger.debug("Configuration validation passed")
        return True
    
    def get_cache_dir(self) -> Path:
        """
        Get the cache directory path.
        
        Returns:
            Path object for the cache directory
        """
        cache_dir = Path(self.get('cache_dir', '~/.templateheaven/cache')).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_template_cache_dir(self) -> Path:
        """
        Get the template cache directory path.
        
        Returns:
            Path object for the template cache directory
        """
        template_cache = self.get_cache_dir() / "templates"
        template_cache.mkdir(parents=True, exist_ok=True)
        return template_cache
    
    def get_metadata_cache_dir(self) -> Path:
        """
        Get the metadata cache directory path.
        
        Returns:
            Path object for the metadata cache directory
        """
        metadata_cache = self.get_cache_dir() / "metadata"
        metadata_cache.mkdir(parents=True, exist_ok=True)
        return metadata_cache
    
    def is_first_run(self) -> bool:
        """
        Check if this is the first run (no custom configuration).
        
        Returns:
            True if this is the first run
        """
        return not self.config_file.exists() or self._config == self.DEFAULT_CONFIG
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get configuration information for display.
        
        Returns:
            Dictionary with configuration metadata
        """
        return {
            "config_file": str(self.config_file),
            "cache_dir": str(self.get_cache_dir()),
            "is_first_run": self.is_first_run(),
            "last_updated": self.get("_metadata.last_updated"),
            "version": self.get("_metadata.version"),
        }
