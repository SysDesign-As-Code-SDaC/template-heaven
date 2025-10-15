"""
API Server for MCP Middleware.

This server provides AI assistants with the ability to make HTTP requests,
interact with REST APIs, GraphQL endpoints, and handle authentication.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
import aiohttp
import requests
from urllib.parse import urlparse, urljoin
from datetime import datetime

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class APIServer(BaseMCPServer):
    """
    MCP server for API interactions.

    Provides tools for making HTTP requests, handling authentication,
    and interacting with REST APIs and GraphQL endpoints.
    """

    def __init__(self, name: str, config: Dict[str, Any], auth: Optional[Dict[str, Any]] = None):
        super().__init__(name, config, auth)
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_tokens: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> bool:
        """Initialize the API server."""
        try:
            # Validate configuration
            self._validate_config()

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Initialize authentication if configured
            if self.auth:
                await self._initialize_auth()

            logger.info(f"API server '{self.name}' initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize API server '{self.name}': {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the API server."""
        try:
            if self.session:
                await self.session.close()
                self.session = None

            logger.info(f"API server '{self.name}' shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error shutting down API server '{self.name}': {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            health_status = {
                "status": "healthy" if self.session else "unhealthy",
                "session_active": self.session is not None,
                "auth_configured": self.auth is not None,
                "timestamp": datetime.utcnow().isoformat()
            }
            return health_status

        except Exception as e:
            logger.error(f"Health check failed for API server '{self.name}': {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available API tools."""
        return [
            {
                "name": "http_request",
                "description": "Make an HTTP request to any URL with custom method, headers, and body",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                            "description": "HTTP method to use"
                        },
                        "url": {
                            "type": "string",
                            "description": "URL to request"
                        },
                        "headers": {
                            "type": "object",
                            "description": "HTTP headers as key-value pairs",
                            "additionalProperties": {"type": "string"}
                        },
                        "body": {
                            "type": "string",
                            "description": "Request body (for POST/PUT/PATCH)"
                        },
                        "params": {
                            "type": "object",
                            "description": "Query parameters as key-value pairs",
                            "additionalProperties": {"type": "string"}
                        },
                        "auth_type": {
                            "type": "string",
                            "enum": ["none", "basic", "bearer", "api_key"],
                            "description": "Authentication type"
                        },
                        "auth_credentials": {
                            "type": "object",
                            "description": "Authentication credentials"
                        }
                    },
                    "required": ["method", "url"]
                }
            },
            {
                "name": "graphql_query",
                "description": "Execute a GraphQL query against a GraphQL endpoint",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "description": "GraphQL endpoint URL"
                        },
                        "query": {
                            "type": "string",
                            "description": "GraphQL query string"
                        },
                        "variables": {
                            "type": "object",
                            "description": "GraphQL variables"
                        },
                        "headers": {
                            "type": "object",
                            "description": "HTTP headers",
                            "additionalProperties": {"type": "string"}
                        }
                    },
                    "required": ["endpoint", "query"]
                }
            },
            {
                "name": "rest_api_discover",
                "description": "Discover REST API endpoints and capabilities",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "base_url": {
                            "type": "string",
                            "description": "Base URL of the REST API"
                        },
                        "include_options": {
                            "type": "boolean",
                            "description": "Include OPTIONS requests to discover allowed methods",
                            "default": false
                        }
                    },
                    "required": ["base_url"]
                }
            },
            {
                "name": "oauth2_token",
                "description": "Obtain OAuth2 access token",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "token_url": {
                            "type": "string",
                            "description": "OAuth2 token endpoint URL"
                        },
                        "client_id": {
                            "type": "string",
                            "description": "OAuth2 client ID"
                        },
                        "client_secret": {
                            "type": "string",
                            "description": "OAuth2 client secret"
                        },
                        "grant_type": {
                            "type": "string",
                            "enum": ["client_credentials", "password", "authorization_code"],
                            "description": "OAuth2 grant type"
                        },
                        "scope": {
                            "type": "string",
                            "description": "OAuth2 scope"
                        }
                    },
                    "required": ["token_url", "client_id", "client_secret", "grant_type"]
                }
            }
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute an API tool."""
        try:
            if tool_name == "http_request":
                return await self._execute_http_request(arguments)
            elif tool_name == "graphql_query":
                return await self._execute_graphql_query(arguments)
            elif tool_name == "rest_api_discover":
                return await self._discover_rest_api(arguments)
            elif tool_name == "oauth2_token":
                return await self._obtain_oauth2_token(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing API tool '{tool_name}': {e}")
            return [{
                "type": "text",
                "text": f"Error executing {tool_name}: {str(e)}"
            }]

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available API resources."""
        resources = []

        # Add configured API endpoints as resources
        if "endpoints" in self.config:
            for endpoint_name, endpoint_config in self.config["endpoints"].items():
                resources.append({
                    "uri": f"api://{self.name}/{endpoint_name}",
                    "mimeType": "application/json",
                    "description": endpoint_config.get("description", f"API endpoint: {endpoint_name}")
                })

        return resources

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read an API resource."""
        try:
            if not uri.startswith("api://"):
                raise ValueError(f"Invalid API resource URI: {uri}")

            # Parse URI: api://server_name/endpoint_name
            parts = uri[6:].split("/", 1)  # Remove "api://" prefix
            if len(parts) != 2:
                raise ValueError(f"Invalid API resource URI format: {uri}")

            server_name, endpoint_name = parts

            if server_name != self.name:
                raise ValueError(f"Resource belongs to different server: {server_name}")

            # Get endpoint configuration
            endpoints = self.config.get("endpoints", {})
            if endpoint_name not in endpoints:
                raise ValueError(f"Unknown endpoint: {endpoint_name}")

            endpoint_config = endpoints[endpoint_name]

            # Make request to endpoint
            url = endpoint_config["url"]
            method = endpoint_config.get("method", "GET")
            headers = endpoint_config.get("headers", {})

            # Add authentication if configured
            if self.auth:
                headers.update(await self._get_auth_headers())

            async with self.session.request(method, url, headers=headers) as response:
                content = await response.text()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": response.headers.get("content-type", "application/json"),
                        "text": content
                    }]
                }

        except Exception as e:
            logger.error(f"Error reading API resource '{uri}': {e}")
            raise

    def _validate_config(self):
        """Validate API server configuration."""
        required_keys = []
        if not all(key in self.config for key in required_keys):
            missing = [key for key in required_keys if key not in self.config]
            raise ValueError(f"Missing required configuration keys: {missing}")

        # Validate timeout
        if "timeout" in self.config:
            if not isinstance(self.config["timeout"], (int, float)) or self.config["timeout"] <= 0:
                raise ValueError("Timeout must be a positive number")

        # Validate endpoints if provided
        if "endpoints" in self.config:
            if not isinstance(self.config["endpoints"], dict):
                raise ValueError("Endpoints must be a dictionary")

            for endpoint_name, endpoint_config in self.config["endpoints"].items():
                if not isinstance(endpoint_config, dict):
                    raise ValueError(f"Endpoint '{endpoint_name}' configuration must be a dictionary")

                if "url" not in endpoint_config:
                    raise ValueError(f"Endpoint '{endpoint_name}' missing required 'url' key")

    async def _initialize_auth(self):
        """Initialize authentication configuration."""
        auth_type = self.auth.get("type")

        if auth_type == "oauth2":
            await self._initialize_oauth2()
        elif auth_type == "api_key":
            self._initialize_api_key()

    async def _initialize_oauth2(self):
        """Initialize OAuth2 authentication."""
        oauth_config = self.auth.get("oauth2", {})
        token_url = oauth_config.get("token_url")
        client_id = oauth_config.get("client_id")
        client_secret = oauth_config.get("client_secret")

        if not all([token_url, client_id, client_secret]):
            raise ValueError("OAuth2 configuration missing required fields")

        # Obtain initial token
        token_data = await self._obtain_oauth2_token({
            "token_url": token_url,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": oauth_config.get("grant_type", "client_credentials"),
            "scope": oauth_config.get("scope")
        })

        if token_data:
            self.auth_tokens["oauth2"] = token_data[0] if token_data else {}

    def _initialize_api_key(self):
        """Initialize API key authentication."""
        api_key_config = self.auth.get("api_key", {})
        key = api_key_config.get("key")
        header_name = api_key_config.get("header_name", "X-API-Key")

        if not key:
            raise ValueError("API key authentication missing 'key' field")

        self.auth_tokens["api_key"] = {
            "key": key,
            "header_name": header_name
        }

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        headers = {}

        if not self.auth:
            return headers

        auth_type = self.auth.get("type")

        if auth_type == "basic":
            import base64
            username = self.auth.get("username", "")
            password = self.auth.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"

        elif auth_type == "bearer":
            token = self.auth.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "api_key":
            api_key_config = self.auth_tokens.get("api_key", {})
            header_name = api_key_config.get("header_name", "X-API-Key")
            key = api_key_config.get("key")
            if key:
                headers[header_name] = key

        elif auth_type == "oauth2":
            token_data = self.auth_tokens.get("oauth2", {})
            access_token = token_data.get("access_token")
            if access_token:
                headers["Authorization"] = f"Bearer {access_token}"

        return headers

    async def _execute_http_request(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute an HTTP request."""
        method = args["method"]
        url = args["url"]
        headers = args.get("headers", {})
        body = args.get("body")
        params = args.get("params", {})
        auth_type = args.get("auth_type", "none")
        auth_credentials = args.get("auth_credentials", {})

        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {url}")

        # Check if URL is allowed
        allowed_domains = self.config.get("allowed_domains", [])
        if allowed_domains and parsed_url.netloc not in allowed_domains:
            raise ValueError(f"Domain not allowed: {parsed_url.netloc}")

        # Add authentication headers
        if auth_type != "none":
            auth_headers = await self._get_auth_headers_for_request(auth_type, auth_credentials)
            headers.update(auth_headers)
        else:
            # Use default authentication
            default_headers = await self._get_auth_headers()
            headers.update(default_headers)

        # Prepare request data
        request_kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "params": params
        }

        if body and method in ["POST", "PUT", "PATCH"]:
            try:
                # Try to parse as JSON first
                json_body = json.loads(body)
                request_kwargs["json"] = json_body
            except json.JSONDecodeError:
                # Treat as plain text
                request_kwargs["data"] = body

        # Make request
        async with self.session.request(**request_kwargs) as response:
            response_text = await response.text()
            response_headers = dict(response.headers)

            return [{
                "type": "text",
                "text": f"HTTP {response.status} {response.reason}\n\nHeaders:\n{json.dumps(response_headers, indent=2)}\n\nBody:\n{response_text}"
            }]

    async def _execute_graphql_query(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a GraphQL query."""
        endpoint = args["endpoint"]
        query = args["query"]
        variables = args.get("variables", {})
        headers = args.get("headers", {})

        # Validate endpoint
        parsed_url = urlparse(endpoint)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid GraphQL endpoint: {endpoint}")

        # Add default authentication
        default_headers = await self._get_auth_headers()
        headers.update(default_headers)

        # Prepare GraphQL request
        graphql_data = {
            "query": query,
            "variables": variables
        }

        request_kwargs = {
            "method": "POST",
            "url": endpoint,
            "json": graphql_data,
            "headers": headers
        }

        # Make request
        async with self.session.request(**request_kwargs) as response:
            response_data = await response.json()
            response_text = json.dumps(response_data, indent=2)

            return [{
                "type": "text",
                "text": f"GraphQL Response (HTTP {response.status}):\n\n{response_text}"
            }]

    async def _discover_rest_api(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover REST API endpoints and capabilities."""
        base_url = args["base_url"]
        include_options = args.get("include_options", False)

        # Validate base URL
        parsed_url = urlparse(base_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid base URL: {base_url}")

        discovery_results = []

        # Common endpoints to check
        endpoints_to_check = [
            "/",
            "/api",
            "/api/v1",
            "/docs",
            "/swagger",
            "/openapi.json"
        ]

        headers = await self._get_auth_headers()

        for endpoint in endpoints_to_check:
            try:
                url = urljoin(base_url, endpoint)
                result = f"Endpoint: {url}\n"

                # HEAD request first
                async with self.session.head(url, headers=headers) as head_response:
                    result += f"HEAD: {head_response.status} {head_response.reason}\n"
                    result += f"Allowed: {head_response.headers.get('Allow', 'N/A')}\n"

                # OPTIONS request if requested
                if include_options:
                    async with self.session.options(url, headers=headers) as options_response:
                        result += f"OPTIONS: {options_response.status} {options_response.reason}\n"
                        result += f"Allowed: {options_response.headers.get('Allow', 'N/A')}\n"

                discovery_results.append(result)

            except Exception as e:
                discovery_results.append(f"Endpoint: {urljoin(base_url, endpoint)}\nError: {str(e)}\n")

        return [{
            "type": "text",
            "text": "REST API Discovery Results:\n\n" + "\n---\n".join(discovery_results)
        }]

    async def _obtain_oauth2_token(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Obtain OAuth2 access token."""
        token_url = args["token_url"]
        client_id = args["client_id"]
        client_secret = args["client_secret"]
        grant_type = args["grant_type"]
        scope = args.get("scope")

        # Prepare token request data
        data = {
            "grant_type": grant_type,
            "client_id": client_id,
            "client_secret": client_secret
        }

        if scope:
            data["scope"] = scope

        # Make token request
        async with self.session.post(token_url, data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                return [{
                    "type": "text",
                    "text": f"OAuth2 Token Obtained:\n\n{json.dumps(token_data, indent=2)}"
                }]
            else:
                error_text = await response.text()
                return [{
                    "type": "text",
                    "text": f"OAuth2 Token Request Failed (HTTP {response.status}):\n\n{error_text}"
                }]

    async def _get_auth_headers_for_request(self, auth_type: str, credentials: Dict[str, Any]) -> Dict[str, str]:
        """Get authentication headers for a specific request."""
        headers = {}

        if auth_type == "basic":
            import base64
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            credentials_b64 = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials_b64}"

        elif auth_type == "bearer":
            token = credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif auth_type == "api_key":
            header_name = credentials.get("header_name", "X-API-Key")
            key = credentials.get("key")
            if key:
                headers[header_name] = key

        return headers
