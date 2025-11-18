"""
Tests for Diagram Generator.

This module contains tests for generating C4 model diagrams using Mermaid syntax.
"""

import pytest
from pathlib import Path

from templateheaven.core.diagram_generator import DiagramGenerator
from templateheaven.core.architecture_questionnaire import (
    ArchitectureAnswers,
    ArchitecturePattern,
    DeploymentModel,
    ScalabilityRequirement,
)


class TestDiagramGenerator:
    """Test DiagramGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a DiagramGenerator instance."""
        return DiagramGenerator()

    @pytest.fixture
    def sample_answers(self):
        """Create sample architecture answers."""
        answers = ArchitectureAnswers()
        answers.project_vision = "Build a scalable microservices platform"
        answers.target_users = "Enterprise customers, Developers, Administrators"
        answers.business_objectives = ["Scalability", "Reliability"]
        answers.architecture_pattern = ArchitecturePattern.MICROSERVICES
        answers.deployment_model = DeploymentModel.CLOUD_NATIVE
        answers.scalability_requirement = ScalabilityRequirement.HIGH
        answers.database_requirements = ["PostgreSQL", "Redis"]
        answers.integration_requirements = ["REST APIs", "GraphQL", "External Payment Service"]
        answers.api_style = "REST"
        answers.caching_strategy = "Redis"
        return answers

    @pytest.fixture
    def minimal_answers(self):
        """Create minimal architecture answers."""
        answers = ArchitectureAnswers()
        answers.project_vision = "Simple project"
        answers.target_users = "Users"
        answers.architecture_pattern = ArchitecturePattern.MONOLITH
        return answers

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator is not None
        assert generator.logger is not None

    def test_generate_system_context(self, generator, sample_answers):
        """Test generating system context diagram."""
        diagram = generator.generate_system_context("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert len(diagram) > 0
        assert "graph TB" in diagram
        assert "test-project" in diagram.lower() or "test" in diagram.lower()

    def test_generate_system_context_with_minimal_data(self, generator, minimal_answers):
        """Test generating system context with minimal data."""
        diagram = generator.generate_system_context("simple-project", minimal_answers)

        assert isinstance(diagram, str)
        assert "graph TB" in diagram
        assert "simple-project" in diagram.lower() or "simple" in diagram.lower()

    def test_generate_container_diagram_microservices(self, generator, sample_answers):
        """Test generating container diagram for microservices."""
        sample_answers.architecture_pattern = ArchitecturePattern.MICROSERVICES
        diagram = generator.generate_container_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert len(diagram) > 0
        assert "graph TB" in diagram
        assert "Gateway" in diagram or "gateway" in diagram.lower()

    def test_generate_container_diagram_serverless(self, generator, sample_answers):
        """Test generating container diagram for serverless."""
        sample_answers.architecture_pattern = ArchitecturePattern.SERVERLESS
        diagram = generator.generate_container_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert "graph TB" in diagram
        assert "Functions" in diagram or "functions" in diagram.lower()

    def test_generate_container_diagram_event_driven(self, generator, sample_answers):
        """Test generating container diagram for event-driven."""
        sample_answers.architecture_pattern = ArchitecturePattern.EVENT_DRIVEN
        diagram = generator.generate_container_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert "graph TB" in diagram
        assert "Broker" in diagram or "broker" in diagram.lower()

    def test_generate_container_diagram_monolith(self, generator, sample_answers):
        """Test generating container diagram for monolith."""
        sample_answers.architecture_pattern = ArchitecturePattern.MONOLITH
        diagram = generator.generate_container_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert "graph TB" in diagram
        assert "App" in diagram or "app" in diagram.lower()

    def test_generate_component_diagram_microservices(self, generator, sample_answers):
        """Test generating component diagram for microservices."""
        sample_answers.architecture_pattern = ArchitecturePattern.MICROSERVICES
        diagram = generator.generate_component_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert len(diagram) > 0
        assert "graph TB" in diagram
        assert "API" in diagram or "api" in diagram.lower()

    def test_generate_component_diagram_serverless(self, generator, sample_answers):
        """Test generating component diagram for serverless."""
        sample_answers.architecture_pattern = ArchitecturePattern.SERVERLESS
        diagram = generator.generate_component_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert "graph TB" in diagram

    def test_generate_component_diagram_event_driven(self, generator, sample_answers):
        """Test generating component diagram for event-driven."""
        sample_answers.architecture_pattern = ArchitecturePattern.EVENT_DRIVEN
        diagram = generator.generate_component_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert "graph TB" in diagram
        assert "Broker" in diagram or "broker" in diagram.lower()

    def test_generate_component_diagram_monolith(self, generator, sample_answers):
        """Test generating component diagram for monolith."""
        sample_answers.architecture_pattern = ArchitecturePattern.MONOLITH
        diagram = generator.generate_component_diagram("test-project", sample_answers)

        assert isinstance(diagram, str)
        assert "graph TB" in diagram

    def test_generate_all_diagrams(self, generator, sample_answers):
        """Test generating all diagrams."""
        diagrams = generator.generate_all_diagrams("test-project", sample_answers)

        assert isinstance(diagrams, dict)
        assert len(diagrams) == 3
        assert "system_context" in diagrams
        assert "container" in diagrams
        assert "component" in diagrams

        for diagram_name, diagram_content in diagrams.items():
            assert isinstance(diagram_content, str)
            assert len(diagram_content) > 0
            assert "graph TB" in diagram_content

    def test_generate_all_diagrams_with_minimal_data(self, generator, minimal_answers):
        """Test generating all diagrams with minimal data."""
        diagrams = generator.generate_all_diagrams("simple-project", minimal_answers)

        assert isinstance(diagrams, dict)
        assert len(diagrams) == 3

        for diagram_name, diagram_content in diagrams.items():
            assert isinstance(diagram_content, str)
            assert len(diagram_content) > 0

    def test_extract_users(self, generator):
        """Test user extraction from target_users string."""
        # Test with comma separator
        users = generator._extract_users("Enterprise customers, Developers, Admins")
        assert len(users) <= 3
        assert len(users) > 0

        # Test with "and" separator
        users = generator._extract_users("Users and Administrators")
        assert len(users) <= 3
        assert len(users) > 0

        # Test with single user
        users = generator._extract_users("End Users")
        assert len(users) == 1

        # Test with empty string
        users = generator._extract_users("")
        assert len(users) == 1  # Should default to "End Users"

    def test_extract_external_systems(self, generator):
        """Test external system extraction."""
        # Test with integration requirements
        integrations = ["REST APIs", "GraphQL", "External Payment Service"]
        systems = generator._extract_external_systems(integrations)
        assert len(systems) <= 3
        assert len(systems) > 0

        # Test with empty list
        systems = generator._extract_external_systems([])
        assert len(systems) == 0

    def test_get_system_description(self, generator, sample_answers):
        """Test system description extraction."""
        desc = generator._get_system_description(sample_answers)
        assert isinstance(desc, str)
        assert len(desc) > 0

        # Test with empty vision
        sample_answers.project_vision = ""
        desc = generator._get_system_description(sample_answers)
        assert isinstance(desc, str)

    def test_microservices_containers(self, generator, sample_answers):
        """Test microservices container generation."""
        sample_answers.architecture_pattern = ArchitecturePattern.MICROSERVICES
        containers = generator._generate_microservices_containers(sample_answers)
        assert isinstance(containers, list)
        assert len(containers) > 0

    def test_serverless_containers(self, generator, sample_answers):
        """Test serverless container generation."""
        sample_answers.architecture_pattern = ArchitecturePattern.SERVERLESS
        containers = generator._generate_serverless_containers(sample_answers)
        assert isinstance(containers, list)
        assert len(containers) > 0

    def test_event_driven_containers(self, generator, sample_answers):
        """Test event-driven container generation."""
        sample_answers.architecture_pattern = ArchitecturePattern.EVENT_DRIVEN
        containers = generator._generate_event_driven_containers(sample_answers)
        assert isinstance(containers, list)
        assert len(containers) > 0

    def test_monolith_containers(self, generator, sample_answers):
        """Test monolith container generation."""
        sample_answers.architecture_pattern = ArchitecturePattern.MONOLITH
        containers = generator._generate_monolith_containers(sample_answers)
        assert isinstance(containers, list)
        assert len(containers) > 0

