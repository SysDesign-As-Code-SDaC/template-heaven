"""
Web MCP Server.

This server provides web scraping and API interaction capabilities
through the MCP protocol, allowing AI assistants to fetch and process web content.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import requests
from urllib.parse import urlparse, urljoin

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class WebServer(BaseMCPServer):
    """MCP server for web operations."""

    def __init__(self, name: str, config: Dict[str, Any], auth: Dict[str, Any]):
        super().__init__(name, config, auth)
        self.allowed_domains = config.get("allowed_domains", [])
        self.max_content_size = config.get("max_content_size", "1MB")
        self.request_timeout = config.get("request_timeout", 30)
        self.user_agent = config.get("user_agent", "MCP-Web-Server/1.0")

        # Convert max content size to bytes
        self.max_content_size_bytes = self._parse_size(self.max_content_size)

    @classmethod
    def validate_config(cls, config: Dict[str, Any]):
        """Validate web server configuration."""
        allowed_domains = config.get("allowed_domains", [])
        if allowed_domains:
            for domain in allowed_domains:
                if not isinstance(domain, str) or not domain.strip():
                    raise ValueError(f"Invalid domain in allowed_domains: {domain}")

    async def initialize(self):
        """Initialize the web server."""
        self.initialized = True
        logger.info(f"Web server {self.name} initialized")

    async def shutdown(self):
        """Shutdown the web server."""
        self.initialized = False
        logger.info(f"Web server {self.name} shutdown")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": True,
            "timestamp": datetime.utcnow().isoformat(),
            "server": self.name,
            "allowed_domains": len(self.allowed_domains)
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available web tools."""
        return [
            {
                "name": "fetch_url",
                "description": "Fetch content from a URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST"],
                            "default": "GET"
                        },
                        "headers": {
                            "type": "object",
                            "description": "Additional headers to send"
                        },
                        "data": {
                            "type": "object",
                            "description": "Data for POST requests"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "extract_text",
                "description": "Extract readable text from HTML content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "html_content": {
                            "type": "string",
                            "description": "HTML content to extract text from"
                        }
                    },
                    "required": ["html_content"]
                }
            }
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a web tool."""
        try:
            if tool_name == "fetch_url":
                return await self._fetch_url(arguments)
            elif tool_name == "extract_text":
                return await self._extract_text(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            return self._handle_error(f"call_tool_{tool_name}", e)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List web resources."""
        return []

    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Read a web resource."""
        raise NotImplementedError("Web resources not implemented")

    def _validate_url(self, url: str) -> bool:
        """Validate URL against allowed domains."""
        if not self.allowed_domains:
            return True  # Allow all if no restrictions

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            for allowed in self.allowed_domains:
                if domain.endswith(allowed.lower()) or domain == allowed.lower():
                    return True

            return False
        except Exception:
            return False

    async def _fetch_url(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch content from URL."""
        url = arguments["url"]
        method = arguments.get("method", "GET").upper()
        headers = arguments.get("headers", {})
        data = arguments.get("data", {})

        # Validate URL
        if not self._validate_url(url):
            raise ValueError(f"URL not allowed: {url}")

        # Set user agent
        headers.setdefault("User-Agent", self.user_agent)

        try:
            if method == "GET":
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.request_timeout
                )
            elif method == "POST":
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.request_timeout
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()

            # Check content size
            content_length = len(response.content)
            if content_length > self.max_content_size_bytes:
                raise ValueError(f"Content too large: {content_length} bytes")

            return {
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_length": content_length,
                "content_type": response.headers.get("content-type"),
                "text": response.text[:10000] if len(response.text) > 10000 else response.text
            }

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {str(e)}")

    async def _extract_text(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract readable text from HTML."""
        html_content = arguments["html_content"]

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)

            return {
                "text": clean_text,
                "length": len(clean_text),
                "lines": len(lines)
            }

        except ImportError:
            raise ValueError("BeautifulSoup not available for text extraction")
        except Exception as e:
            raise ValueError(f"Text extraction failed: {str(e)}")

    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '1MB') to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
