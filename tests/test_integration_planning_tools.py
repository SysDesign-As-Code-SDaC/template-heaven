"""
Integration tests for planning tools (Diagrams, ADRs, RFCs).

This module tests the integration of all planning tools together.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from templateheaven.core.architecture_doc_generator import ArchitectureDocGenerator
from templateheaven.core.diagram_generator import DiagramGenerator
from templateheaven.core.adr_manager import ADRManager
from templateheaven.core.rfc_manager import RFCManager
from templateheaven.core.architecture_questionnaire import (
    ArchitectureAnswers,
    ArchitecturePattern,
    DeploymentModel,
    ScalabilityRequirement,
)


class TestPlanningToolsIntegration:
    """Integration tests for planning tools."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_answers(self):
        """Create comprehensive sample architecture answers."""
        answers = ArchitectureAnswers()
        answers.project_vision = "Build a modern microservices platform for enterprise customers"
        answers.target_users = "Enterprise customers, Developers, System Administrators"
        answers.business_objectives = ["Scalability", "Reliability", "Security", "Performance"]
        answers.success_metrics = ["uptime: 99.9%", "response_time: <200ms"]
        answers.architecture_pattern = ArchitecturePattern.MICROSERVICES
        answers.deployment_model = DeploymentModel.CLOUD_NATIVE
        answers.scalability_requirement = ScalabilityRequirement.HIGH
        answers.database_requirements = ["PostgreSQL", "Redis"]
        answers.integration_requirements = ["REST APIs", "GraphQL", "External Payment Service"]
        answers.api_style = "REST"
        answers.caching_strategy = "Redis"
        answers.security_requirements = ["authentication", "authorization", "encryption"]
        return answers

    def test_full_documentation_generation(self, temp_project_dir, sample_answers):
        """Test complete documentation generation with diagrams."""
        generator = ArchitectureDocGenerator()
        
        # Generate all docs
        docs = generator.generate_all_docs(
            project_name="Test Project",
            answers=sample_answers,
            output_dir=temp_project_dir
        )
        
        # Verify all expected docs were generated
        assert "architecture" in docs
        assert "system_design" in docs
        assert "roadmap" in docs
        assert "infrastructure" in docs
        assert "security" in docs
        
        # Verify architecture doc exists and contains diagrams
        arch_doc = docs["architecture"]
        assert arch_doc.exists()
        content = arch_doc.read_text()
        assert "```mermaid" in content
        assert "System Context" in content
        
        # Verify system design doc exists and contains diagrams
        system_doc = docs["system_design"]
        assert system_doc.exists()
        system_content = system_doc.read_text()
        assert "```mermaid" in system_content
        assert "Container Diagram" in system_content
        assert "Component Diagram" in system_content
        
        # Verify diagram files were created
        diagrams_dir = temp_project_dir / "docs" / "architecture" / "diagrams"
        assert diagrams_dir.exists()
        assert (diagrams_dir / "system_context.mmd").exists()
        assert (diagrams_dir / "container.mmd").exists()
        assert (diagrams_dir / "component.mmd").exists()
        
        # Verify diagram content
        system_context_content = (diagrams_dir / "system_context.mmd").read_text()
        assert "graph TB" in system_context_content
        assert "System" in system_context_content

    def test_adr_workflow(self, temp_project_dir):
        """Test complete ADR creation and management workflow."""
        adr_manager = ADRManager(temp_project_dir)
        
        # Create first ADR
        adr1_path = adr_manager.create_adr(
            title="Use Microservices Architecture",
            context="We need to scale independently",
            decision="We will use microservices",
            consequences="Increased complexity but better scalability",
            alternatives=["Monolith", "Serverless"],
            status="Accepted"
        )
        
        assert adr1_path.exists()
        assert "0001" in adr1_path.name
        
        # Create second ADR
        adr2_path = adr_manager.create_adr(
            title="Use PostgreSQL as Primary Database",
            context="Need reliable relational database",
            decision="PostgreSQL will be our primary database",
            consequences="Reliable but requires management",
            status="Proposed"
        )
        
        assert adr2_path.exists()
        assert "0002" in adr2_path.name
        
        # List ADRs
        adrs = adr_manager.list_adrs()
        assert len(adrs) == 2
        assert adrs[0]["number"] == 1
        assert adrs[1]["number"] == 2
        
        # Get specific ADR
        found_adr = adr_manager.get_adr(1)
        assert found_adr == adr1_path
        
        # Update status
        success = adr_manager.update_adr_status(2, "Accepted")
        assert success
        
        # Verify status update
        updated_adrs = adr_manager.list_adrs()
        assert updated_adrs[1]["status"] == "Accepted"

    def test_rfc_workflow(self, temp_project_dir):
        """Test complete RFC creation and management workflow."""
        rfc_manager = RFCManager(temp_project_dir)
        
        # Create first RFC
        rfc1_path = rfc_manager.create_rfc(
            title="Implement Feature Flags System",
            summary="Add feature flags for gradual rollouts",
            motivation="We need to deploy features safely",
            design="Use LaunchDarkly for feature flag management",
            alternatives=["Custom solution", "Unleash"],
            open_questions=["Which provider?", "Cost?"],
            status="Draft"
        )
        
        assert rfc1_path.exists()
        assert "RFC-0001" in rfc1_path.name
        
        # Create second RFC
        rfc2_path = rfc_manager.create_rfc(
            title="Migrate to GraphQL",
            summary="Replace REST API with GraphQL",
            motivation="Better client flexibility",
            design="Use Apollo Server",
            status="Review"
        )
        
        assert rfc2_path.exists()
        assert "RFC-0002" in rfc2_path.name
        
        # List RFCs
        rfcs = rfc_manager.list_rfcs()
        assert len(rfcs) == 2
        assert rfcs[0]["number"] == 1
        assert rfcs[1]["number"] == 2
        
        # Get specific RFC
        found_rfc = rfc_manager.get_rfc(1)
        assert found_rfc == rfc1_path
        
        # Update status
        success = rfc_manager.update_rfc_status(1, "Accepted")
        assert success
        
        # Verify status update
        updated_rfcs = rfc_manager.list_rfcs()
        assert updated_rfcs[0]["status"] == "Accepted"

    def test_diagram_generation_for_all_patterns(self, temp_project_dir):
        """Test diagram generation for all architecture patterns."""
        generator = DiagramGenerator()
        
        patterns = [
            ArchitecturePattern.MICROSERVICES,
            ArchitecturePattern.SERVERLESS,
            ArchitecturePattern.EVENT_DRIVEN,
            ArchitecturePattern.MONOLITH,
        ]
        
        for pattern in patterns:
            answers = ArchitectureAnswers()
            answers.architecture_pattern = pattern
            answers.project_vision = f"Test {pattern.value} project"
            answers.target_users = "Users"
            answers.database_requirements = ["PostgreSQL"]
            
            diagrams = generator.generate_all_diagrams("Test Project", answers)
            
            assert "system_context" in diagrams
            assert "container" in diagrams
            assert "component" in diagrams
            
            # Verify diagrams are valid Mermaid syntax
            for diagram_name, diagram_content in diagrams.items():
                assert "graph TB" in diagram_content
                assert len(diagram_content) > 0

    def test_documentation_with_adrs_and_rfcs(self, temp_project_dir, sample_answers):
        """Test that documentation generation works alongside ADRs and RFCs."""
        # Generate documentation
        doc_generator = ArchitectureDocGenerator()
        docs = doc_generator.generate_all_docs(
            project_name="Test Project",
            answers=sample_answers,
            output_dir=temp_project_dir
        )
        
        # Create ADRs
        adr_manager = ADRManager(temp_project_dir)
        adr_path = adr_manager.create_adr(
            title="Architecture Decision",
            context="Test context",
            decision="Test decision",
            consequences="Test consequences"
        )
        
        # Create RFCs
        rfc_manager = RFCManager(temp_project_dir)
        rfc_path = rfc_manager.create_rfc(
            title="Test RFC",
            summary="Test summary",
            motivation="Test motivation",
            design="Test design"
        )
        
        # Verify all files exist
        assert docs["architecture"].exists()
        assert adr_path.exists()
        assert rfc_path.exists()
        
        # Verify directory structure
        assert (temp_project_dir / "docs" / "architecture").exists()
        assert (temp_project_dir / "docs" / "adr").exists()
        assert (temp_project_dir / "docs" / "rfc").exists()

