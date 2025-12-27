"""
Diagram Generator for Template Heaven.

Generates C4 model diagrams using Mermaid syntax from architecture answers.
"""

from typing import Dict, List, Optional
from pathlib import Path

from .architecture_questionnaire import ArchitectureAnswers, ArchitecturePattern
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DiagramGenerator:
    """Generates C4 model diagrams using Mermaid syntax."""

    def __init__(self):
        """Initialize the diagram generator."""
        self.logger = logger

    def generate_system_context(
        self, project_name: str, answers: ArchitectureAnswers
    ) -> str:
        """
        Generate System Context diagram (C4 Level 1).

        Shows the system and its relationships with users and external systems.

        Args:
            project_name: Name of the project/system
            answers: Architecture questionnaire answers

        Returns:
            Mermaid diagram code as string
        """
        system_name = project_name.replace(" ", "-").lower()
        system_label = project_name

        # Extract users from target_users
        users = self._extract_users(answers.target_users)

        # Extract external systems from integration requirements
        external_systems = self._extract_external_systems(
            answers.integration_requirements
        )

        # Build Mermaid diagram
        lines = ["graph TB"]
        lines.append(f'    System["{system_label}<br/>{self._get_system_description(answers)}"]')

        # Add user relationships
        for i, user in enumerate(users):
            user_id = f"User{i+1}"
            lines.append(f'    {user_id}["{user}"]')
            lines.append(f"    {user_id} -->|Uses| System")

        # Add external system relationships
        for i, ext_system in enumerate(external_systems):
            ext_id = f"Ext{i+1}"
            lines.append(f'    {ext_id}["{ext_system}"]')
            lines.append(f"    System -->|Integrates with| {ext_id}")

        # Add database if specified
        if answers.database_requirements:
            db_name = answers.database_requirements[0] if answers.database_requirements else "Database"
            lines.append(f'    Database["{db_name}"]')
            lines.append(f"    System -->|Stores data in| Database")

        return "\n".join(lines)

    def generate_container_diagram(
        self, project_name: str, answers: ArchitectureAnswers
    ) -> str:
        """
        Generate Container diagram (C4 Level 2).

        Shows the high-level technical building blocks and how they interact.

        Args:
            project_name: Name of the project/system
            answers: Architecture questionnaire answers

        Returns:
            Mermaid diagram code as string
        """
        system_name = project_name.replace(" ", "-").lower()
        pattern = answers.architecture_pattern

        lines = ["graph TB"]

        # Determine containers based on architecture pattern
        if pattern == ArchitecturePattern.MICROSERVICES:
            containers = self._generate_microservices_containers(answers)
        elif pattern == ArchitecturePattern.SERVERLESS:
            containers = self._generate_serverless_containers(answers)
        elif pattern == ArchitecturePattern.EVENT_DRIVEN:
            containers = self._generate_event_driven_containers(answers)
        else:
            containers = self._generate_monolith_containers(answers)

        # Add containers to diagram
        for container in containers:
            lines.append(f'    {container["id"]}["{container["label"]}"]')

        # Add relationships
        for container in containers:
            if "relationships" in container:
                for rel in container["relationships"]:
                    lines.append(f'    {container["id"]} -->|{rel["label"]}| {rel["target"]}')

        return "\n".join(lines)

    def generate_component_diagram(
        self, project_name: str, answers: ArchitectureAnswers
    ) -> str:
        """
        Generate Component diagram (C4 Level 3).

        Shows the components within a container (simplified based on architecture pattern).

        Args:
            project_name: Name of the project/system
            answers: Architecture questionnaire answers

        Returns:
            Mermaid diagram code as string
        """
        pattern = answers.architecture_pattern

        lines = ["graph TB"]

        # Generate components based on architecture pattern
        if pattern == ArchitecturePattern.MICROSERVICES:
            components = [
                {"id": "API", "label": "API Gateway"},
                {"id": "Auth", "label": "Auth Service"},
                {"id": "Business", "label": "Business Logic"},
                {"id": "Data", "label": "Data Access"},
            ]
        elif pattern == ArchitecturePattern.SERVERLESS:
            components = [
                {"id": "API", "label": "API Functions"},
                {"id": "Process", "label": "Processing Functions"},
                {"id": "Storage", "label": "Storage Layer"},
            ]
        elif pattern == ArchitecturePattern.EVENT_DRIVEN:
            components = [
                {"id": "Producer", "label": "Event Producer"},
                {"id": "Broker", "label": "Message Broker"},
                {"id": "Consumer", "label": "Event Consumer"},
            ]
        else:
            components = [
                {"id": "UI", "label": "User Interface"},
                {"id": "Business", "label": "Business Logic"},
                {"id": "Data", "label": "Data Layer"},
            ]

        # Add components
        for component in components:
            lines.append(f'    {component["id"]}["{component["label"]}"]')

        # Add relationships (simple flow)
        for i in range(len(components) - 1):
            lines.append(f'    {components[i]["id"]} --> {components[i+1]["id"]}')

        return "\n".join(lines)

    def generate_all_diagrams(
        self, project_name: str, answers: ArchitectureAnswers
    ) -> Dict[str, str]:
        """
        Generate all C4 model diagrams.

        Args:
            project_name: Name of the project/system
            answers: Architecture questionnaire answers

        Returns:
            Dictionary mapping diagram names to Mermaid code
        """
        diagrams = {
            "system_context": self.generate_system_context(project_name, answers),
            "container": self.generate_container_diagram(project_name, answers),
            "component": self.generate_component_diagram(project_name, answers),
        }

        self.logger.info(f"Generated {len(diagrams)} diagrams for {project_name}")
        return diagrams

    def _extract_users(self, target_users: str) -> List[str]:
        """Extract user types from target_users string."""
        import re
        if not target_users:
            return ["End Users"]

        # Split by comma, 'and', or '&', with optional surrounding whitespace
        users = [u.strip() for u in re.split(r'[,&]|\s+and\s+', target_users) if u.strip()]

        if not users:
            users = [target_users.strip()]

        # Limit to 3 users for diagram clarity
        return users[:3]

    def _extract_external_systems(self, integration_requirements: List[str]) -> List[str]:
        """Extract external systems from integration requirements."""
        if not integration_requirements:
            return []

        external_systems = []
        for req in integration_requirements[:3]:  # Limit to 3 for clarity
            # Extract system names (simple heuristic)
            if "API" in req:
                external_systems.append("External API")
            elif "Database" in req or "DB" in req:
                external_systems.append("External Database")
            elif "Service" in req:
                external_systems.append("External Service")
            else:
                external_systems.append(req)

        return external_systems[:3]

    def _get_system_description(self, answers: ArchitectureAnswers) -> str:
        """Get a brief system description."""
        if answers.project_vision:
            # Use first sentence or first 50 chars
            desc = answers.project_vision.split(".")[0]
            if len(desc) > 50:
                desc = desc[:47] + "..."
            return desc
        return "System"

    def _generate_microservices_containers(
        self, answers: ArchitectureAnswers
    ) -> List[Dict]:
        """Generate containers for microservices architecture."""
        containers = [
            {
                "id": "Gateway",
                "label": "API Gateway",
                "relationships": [{"target": "Auth", "label": "Routes to"}],
            },
            {
                "id": "Auth",
                "label": "Auth Service",
                "relationships": [{"target": "Service1", "label": "Authenticates"}],
            },
            {
                "id": "Service1",
                "label": "Microservice 1",
                "relationships": [{"target": "DB1", "label": "Uses"}],
            },
            {
                "id": "DB1",
                "label": answers.database_requirements[0] if answers.database_requirements else "Database",
            },
        ]

        if answers.caching_strategy:
            containers.append(
                {
                    "id": "Cache",
                    "label": "Cache Layer",
                    "relationships": [{"target": "Service1", "label": "Caches for"}],
                }
            )

        return containers

    def _generate_serverless_containers(
        self, answers: ArchitectureAnswers
    ) -> List[Dict]:
        """Generate containers for serverless architecture."""
        containers = [
            {
                "id": "Functions",
                "label": "Serverless Functions",
                "relationships": [{"target": "Storage", "label": "Uses"}],
            },
            {
                "id": "Storage",
                "label": answers.database_requirements[0] if answers.database_requirements else "Storage",
            },
        ]

        if answers.api_style:
            containers.insert(
                0,
                {
                    "id": "API",
                    "label": f"{answers.api_style} API",
                    "relationships": [{"target": "Functions", "label": "Triggers"}],
                },
            )

        return containers

    def _generate_event_driven_containers(
        self, answers: ArchitectureAnswers
    ) -> List[Dict]:
        """Generate containers for event-driven architecture."""
        containers = [
            {
                "id": "Producer",
                "label": "Event Producer",
                "relationships": [{"target": "Broker", "label": "Publishes to"}],
            },
            {
                "id": "Broker",
                "label": "Message Broker",
                "relationships": [{"target": "Consumer", "label": "Delivers to"}],
            },
            {
                "id": "Consumer",
                "label": "Event Consumer",
                "relationships": [{"target": "Storage", "label": "Stores in"}],
            },
            {
                "id": "Storage",
                "label": answers.database_requirements[0] if answers.database_requirements else "Database",
            },
        ]

        return containers

    def _generate_monolith_containers(
        self, answers: ArchitectureAnswers
    ) -> List[Dict]:
        """Generate containers for monolithic architecture."""
        containers = [
            {
                "id": "App",
                "label": "Application",
                "relationships": [{"target": "DB", "label": "Uses"}],
            },
            {
                "id": "DB",
                "label": answers.database_requirements[0] if answers.database_requirements else "Database",
            },
        ]

        if answers.api_style:
            containers.insert(
                0,
                {
                    "id": "API",
                    "label": f"{answers.api_style} API",
                    "relationships": [{"target": "App", "label": "Calls"}],
                },
            )

        if answers.caching_strategy:
            containers.append(
                {
                    "id": "Cache",
                    "label": "Cache",
                    "relationships": [{"target": "App", "label": "Caches for"}],
                }
            )

        return containers

