"""
Service Registry

Manages communication with microservices in the Template Heaven ecosystem.
Routes MCP requests to appropriate services and handles service discovery.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
import json

from .config import MCPConfig
from .protocol import MCPTool, MCPResource, MCPPrompt, TemplateInfo, StackInfo, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    name: str
    url: str
    health_check_url: Optional[str] = None
    capabilities: List[str] = None
    timeout: float = 30.0
    retries: int = 3

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.health_check_url is None:
            self.health_check_url = f"{self.url}/health"


class ServiceRegistry:
    """
    Registry for managing microservice endpoints and routing MCP requests.

    This class maintains a catalog of all Template Heaven microservices and
    handles communication with them via HTTP APIs.
    """

    def __init__(self, config: MCPConfig):
        self.config = config
        self.services: Dict[str, ServiceEndpoint] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.service_health: Dict[str, bool] = {}

        # Register core services
        self._register_services()

    def _register_services(self):
        """Register all Template Heaven microservices."""
        # Template Management Service
        self.services["template-service"] = ServiceEndpoint(
            name="template-service",
            url=self.config.template_service_url,
            capabilities=["templates", "stacks", "projects"]
        )

        # Template Validation Service
        self.services["validation-service"] = ServiceEndpoint(
            name="validation-service",
            url=self.config.validation_service_url,
            capabilities=["validation", "linting", "testing"]
        )

        # Template Generation Service
        self.services["generation-service"] = ServiceEndpoint(
            name="generation-service",
            url=self.config.generation_service_url,
            capabilities=["generation", "scaffolding"]
        )

        # Code Analysis Service
        self.services["analysis-service"] = ServiceEndpoint(
            name="analysis-service",
            url=self.config.analysis_service_url,
            capabilities=["analysis", "metrics", "insights"]
        )

        # User Management Service
        self.services["user-service"] = ServiceEndpoint(
            name="user-service",
            url=self.config.user_service_url,
            capabilities=["authentication", "authorization", "profiles"]
        )

        # Template Sync Service
        self.services["sync-service"] = ServiceEndpoint(
            name="sync-service",
            url=self.config.sync_service_url,
            capabilities=["sync", "upstream", "updates"]
        )

        # API Gateway Service
        self.services["api-gateway"] = ServiceEndpoint(
            name="api-gateway",
            url=self.config.api_gateway_url,
            capabilities=["routing", "rate-limiting", "caching"]
        )

    async def initialize(self):
        """Initialize the service registry."""
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.http_session = aiohttp.ClientSession(timeout=timeout)

        # Perform initial health checks
        await self.check_all_services_health()

        logger.info("Service registry initialized")

    async def shutdown(self):
        """Shutdown the service registry."""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

        logger.info("Service registry shutdown")

    async def check_all_services_health(self):
        """Check health of all registered services."""
        health_tasks = []
        for service_name, service in self.services.items():
            task = self.check_service_health(service_name)
            health_tasks.append(task)

        await asyncio.gather(*health_tasks, return_exceptions=True)

        healthy_count = sum(1 for healthy in self.service_health.values() if healthy)
        total_count = len(self.service_health)

        logger.info(f"Service health check: {healthy_count}/{total_count} services healthy")

    async def check_service_health(self, service_name: str) -> bool:
        """Check health of a specific service."""
        if service_name not in self.services:
            self.service_health[service_name] = False
            return False

        service = self.services[service_name]

        try:
            async with self.http_session.get(service.health_check_url) as response:
                healthy = response.status == 200
                self.service_health[service_name] = healthy

                if not healthy:
                    logger.warning(f"Service {service_name} health check failed: {response.status}")
                else:
                    logger.debug(f"Service {service_name} is healthy")

                return healthy

        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            self.service_health[service_name] = False
            return False

    async def call_service(self, service_name: str, endpoint: str,
                          method: str = "GET", data: Optional[Dict[str, Any]] = None,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a service endpoint.

        Args:
            service_name: Name of the service
            endpoint: API endpoint path
            method: HTTP method
            data: Request body data
            params: Query parameters

        Returns:
            Response data

        Raises:
            Exception: If service call fails
        """
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")

        service = self.services[service_name]

        if not self.service_health.get(service_name, False):
            # Try health check first
            if not await self.check_service_health(service_name):
                raise Exception(f"Service {service_name} is not healthy")

        url = f"{service.url}{endpoint}"

        # Retry logic
        last_exception = None
        for attempt in range(service.retries):
            try:
                async with self.http_session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"Service error: {response.status} - {error_text}")

                    return await response.json()

            except Exception as e:
                last_exception = e
                if attempt < service.retries - 1:
                    logger.warning(f"Service call attempt {attempt + 1} failed for {service_name}: {e}")
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All service call attempts failed for {service_name}: {e}")

        raise last_exception

    # Template Heaven specific service methods

    async def list_templates(self, stack: Optional[str] = None,
                           category: Optional[str] = None) -> List[TemplateInfo]:
        """List available templates."""
        try:
            params = {}
            if stack:
                params["stack"] = stack
            if category:
                params["category"] = category

            response = await self.call_service(
                "template-service",
                "/api/v1/templates",
                params=params
            )

            return [TemplateInfo(**template) for template in response.get("templates", [])]

        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return []

    async def generate_template(self, template_name: str, destination: str,
                              options: Dict[str, Any], connection_id: str) -> Dict[str, Any]:
        """Generate a project from a template."""
        try:
            data = {
                "template": template_name,
                "destination": destination,
                "options": options,
                "connection_id": connection_id
            }

            return await self.call_service(
                "generation-service",
                "/api/v1/generate",
                method="POST",
                data=data
            )

        except Exception as e:
            logger.error(f"Failed to generate template {template_name}: {e}")
            raise

    async def validate_template(self, template_path: str, rules: List[str],
                              connection_id: str) -> ValidationResult:
        """Validate a template."""
        try:
            data = {
                "path": template_path,
                "rules": rules,
                "connection_id": connection_id
            }

            response = await self.call_service(
                "validation-service",
                "/api/v1/validate",
                method="POST",
                data=data
            )

            return ValidationResult(**response)

        except Exception as e:
            logger.error(f"Failed to validate template {template_path}: {e}")
            raise

    async def list_stacks(self) -> List[StackInfo]:
        """List available stacks."""
        try:
            response = await self.call_service(
                "template-service",
                "/api/v1/stacks"
            )

            return [StackInfo(**stack) for stack in response.get("stacks", [])]

        except Exception as e:
            logger.error(f"Failed to list stacks: {e}")
            return []

    async def create_project(self, name: str, template: str,
                           options: Dict[str, Any], connection_id: str) -> Dict[str, Any]:
        """Create a new project."""
        try:
            data = {
                "name": name,
                "template": template,
                "options": options,
                "connection_id": connection_id
            }

            return await self.call_service(
                "api-gateway",
                "/api/v1/projects",
                method="POST",
                data=data
            )

        except Exception as e:
            logger.error(f"Failed to create project {name}: {e}")
            raise

    # MCP protocol methods

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available MCP tools from all services."""
        tools = []

        try:
            # Template tools
            template_tools = await self.call_service("template-service", "/api/v1/tools")
            tools.extend(template_tools.get("tools", []))

        except Exception as e:
            logger.warning(f"Failed to get template tools: {e}")

        try:
            # Validation tools
            validation_tools = await self.call_service("validation-service", "/api/v1/tools")
            tools.extend(validation_tools.get("tools", []))

        except Exception as e:
            logger.warning(f"Failed to get validation tools: {e}")

        try:
            # Analysis tools
            analysis_tools = await self.call_service("analysis-service", "/api/v1/tools")
            tools.extend(analysis_tools.get("tools", []))

        except Exception as e:
            logger.warning(f"Failed to get analysis tools: {e}")

        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any],
                       connection_id: str) -> Dict[str, Any]:
        """Call an MCP tool."""
        # Route tool calls to appropriate services
        if tool_name.startswith("template-"):
            service_name = "template-service"
        elif tool_name.startswith("validate-") or tool_name.startswith("lint-"):
            service_name = "validation-service"
        elif tool_name.startswith("analyze-") or tool_name.startswith("metrics-"):
            service_name = "analysis-service"
        elif tool_name.startswith("user-") or tool_name.startswith("auth-"):
            service_name = "user-service"
        elif tool_name.startswith("sync-"):
            service_name = "sync-service"
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        data = {
            "tool": tool_name,
            "arguments": arguments,
            "connection_id": connection_id
        }

        return await self.call_service(
            service_name,
            "/api/v1/tools/call",
            method="POST",
            data=data
        )

    async def get_all_resources(self) -> List[Dict[str, Any]]:
        """Get all available MCP resources."""
        resources = []

        try:
            # Template resources
            template_resources = await self.call_service("template-service", "/api/v1/resources")
            resources.extend(template_resources.get("resources", []))

        except Exception as e:
            logger.warning(f"Failed to get template resources: {e}")

        return resources

    async def read_resource(self, uri: str, connection_id: str) -> Dict[str, Any]:
        """Read an MCP resource."""
        # Route based on URI pattern
        if uri.startswith("template://"):
            service_name = "template-service"
        elif uri.startswith("validation://"):
            service_name = "validation-service"
        elif uri.startswith("analysis://"):
            service_name = "analysis-service"
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

        data = {
            "uri": uri,
            "connection_id": connection_id
        }

        return await self.call_service(
            service_name,
            "/api/v1/resources/read",
            method="POST",
            data=data
        )

    async def unsubscribe_resource(self, uri: str, connection_id: str):
        """Unsubscribe from a resource."""
        # This is mainly for cleanup - services handle their own subscriptions
        logger.debug(f"Unsubscribed {connection_id} from {uri}")

    async def get_all_prompts(self) -> List[Dict[str, Any]]:
        """Get all available MCP prompts."""
        prompts = []

        try:
            # Template prompts
            template_prompts = await self.call_service("template-service", "/api/v1/prompts")
            prompts.extend(template_prompts.get("prompts", []))

        except Exception as e:
            logger.warning(f"Failed to get template prompts: {e}")

        return prompts

    async def get_prompt(self, prompt_name: str, arguments: Dict[str, Any],
                        connection_id: str) -> Dict[str, Any]:
        """Get an MCP prompt."""
        data = {
            "name": prompt_name,
            "arguments": arguments,
            "connection_id": connection_id
        }

        return await self.call_service(
            "template-service",
            "/api/v1/prompts/get",
            method="POST",
            data=data
        )

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            "services": {
                name: {
                    "url": service.url,
                    "healthy": self.service_health.get(name, False),
                    "capabilities": service.capabilities
                }
                for name, service in self.services.items()
            },
            "overall_health": all(self.service_health.values())
        }
