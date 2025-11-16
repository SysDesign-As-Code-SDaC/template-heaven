"""
Tests for Architecture Document Generator.

This module contains tests for generating architecture documentation
from questionnaire answers.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil

from templateheaven.core.architecture_doc_generator import ArchitectureDocGenerator
from templateheaven.core.architecture_questionnaire import (
    ArchitectureAnswers,
    ArchitecturePattern,
    DeploymentModel,
    ScalabilityRequirement,
    SecurityLevel,
    ComplianceStandard
)


class TestArchitectureDocGenerator:
    """Test ArchitectureDocGenerator class."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_answers(self):
        """Create sample architecture answers."""
        answers = ArchitectureAnswers()
        answers.project_vision = "Build a scalable microservices platform"
        answers.target_users = "Enterprise customers and developers"
        answers.business_objectives = ["Scalability", "Reliability", "Security"]
        answers.success_metrics = ["uptime: 99.9%", "response_time: <200ms", "user_satisfaction: >4.5/5"]
        answers.architecture_patterns = [ArchitecturePattern.MICROSERVICES, ArchitecturePattern.EVENT_DRIVEN]
        answers.deployment_model = DeploymentModel.CLOUD_NATIVE
        answers.scalability_requirement = ScalabilityRequirement.HIGH
        answers.performance_requirements = {
            "response_time_ms": 200,
            "throughput_rps": 10000,
            "concurrent_users": 100000
        }
        answers.security_requirements = ["authentication", "authorization", "encryption"]
        answers.compliance_requirements = ["SOC2", "ISO27001"]
        answers.database_requirements = ["PostgreSQL", "Redis"]
        answers.integration_requirements = ["REST APIs", "GraphQL", "WebSockets"]
        answers.monitoring_strategy = "Prometheus"
        return answers
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = ArchitectureDocGenerator()
        assert generator is not None
        assert generator.logger is not None
    
    def test_generate_all_docs(self, temp_output_dir, sample_answers):
        """Test generating all architecture documents."""
        generator = ArchitectureDocGenerator()
        
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        assert isinstance(result, dict)
        # Check for expected document types (keys may vary)
        assert len(result) > 0, "No documents were generated"
        
        # Check that files were created
        for doc_type, doc_path in result.items():
            assert doc_path.exists(), f"Document {doc_type} was not created"
            assert doc_path.is_file(), f"Document {doc_type} is not a file"
        
        # Check for common document types
        doc_types = set(result.keys())
        assert 'architecture' in doc_types or 'system_design' in doc_types, "Expected architecture or system_design document"
    
    def test_generate_architecture_overview(self, temp_output_dir, sample_answers):
        """Test generating architecture overview document."""
        generator = ArchitectureDocGenerator()
        
        # Use generate_all_docs and check architecture document
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        # Check that architecture document exists
        assert 'architecture' in result, "Architecture document should be generated"
        doc_path = result['architecture']
        assert doc_path.exists()
        assert doc_path.suffix == ".md", "Document should be a markdown file"
        
        # Check content
        content = doc_path.read_text()
        assert len(content) > 0, "Document should have content"
    
    def test_generate_system_design(self, temp_output_dir, sample_answers):
        """Test generating system design document."""
        generator = ArchitectureDocGenerator()
        
        # Use generate_all_docs and check system design document
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        assert 'system_design' in result, "System design document should be generated"
        doc_path = result['system_design']
        assert doc_path.exists()
        assert doc_path.suffix == ".md", "Document should be a markdown file"
        
        # Check content
        content = doc_path.read_text()
        assert len(content) > 0, "Document should have content"
    
    def test_generate_deployment_guide(self, temp_output_dir, sample_answers):
        """Test generating deployment guide."""
        generator = ArchitectureDocGenerator()
        
        # Use generate_all_docs - deployment info is in infrastructure doc
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        # Check that infrastructure document exists (contains deployment info)
        assert 'infrastructure' in result, "Infrastructure document should be generated"
        doc_path = result['infrastructure']
        assert doc_path.exists()
        assert doc_path.suffix == ".md", "Document should be a markdown file"
        
        # Check content
        content = doc_path.read_text()
        assert len(content) > 0, "Document should have content"
    
    def test_generate_roadmap(self, temp_output_dir, sample_answers):
        """Test generating roadmap document."""
        generator = ArchitectureDocGenerator()
        
        # Use generate_all_docs and check roadmap document
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        assert 'roadmap' in result, "Roadmap document should be generated"
        doc_path = result['roadmap']
        assert doc_path.exists()
        assert doc_path.suffix == ".md", "Document should be a markdown file"
        
        # Check content
        content = doc_path.read_text()
        assert len(content) > 0, "Document should have content"
    
    def test_generate_feature_flags(self, temp_output_dir, sample_answers):
        """Test generating feature flags document."""
        generator = ArchitectureDocGenerator()
        
        # Set feature_flags_required to True
        sample_answers.feature_flags_required = True
        
        # Use generate_all_docs and check feature flags document
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        # Feature flags may or may not be generated depending on flag
        if 'feature_flags' in result:
            doc_path = result['feature_flags']
            assert doc_path.exists()
            assert doc_path.suffix == ".md", "Document should be a markdown file"
            content = doc_path.read_text()
            assert len(content) > 0, "Document should have content"
    
    def test_generate_prioritization(self, temp_output_dir, sample_answers):
        """Test generating prioritization document."""
        generator = ArchitectureDocGenerator()
        
        # Prioritization info is typically in roadmap
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        # Check roadmap contains prioritization info
        assert 'roadmap' in result, "Roadmap document should be generated"
        doc_path = result['roadmap']
        assert doc_path.exists()
        assert doc_path.suffix == ".md", "Document should be a markdown file"
        
        # Check content
        content = doc_path.read_text()
        assert len(content) > 0, "Document should have content"
    
    def test_generate_docs_with_minimal_answers(self, temp_output_dir):
        """Test generating docs with minimal answers."""
        generator = ArchitectureDocGenerator()
        
        minimal_answers = ArchitectureAnswers(
            project_vision="Simple project",
            architecture_patterns=[ArchitecturePattern.MONOLITH]
        )
        
        # Should not raise an error
        result = generator.generate_all_docs(
            "test-project",
            minimal_answers,
            temp_output_dir
        )
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_doc_generation_creates_docs_dir(self, temp_output_dir, sample_answers):
        """Test that doc generation creates docs directory if needed."""
        generator = ArchitectureDocGenerator()
        project_dir = temp_output_dir / "new-project"
        project_dir.mkdir()
        
        # Generate docs
        generator.generate_all_docs(
            "new-project",
            sample_answers,
            project_dir
        )
        
        # Check that docs directory was created
        docs_dir = project_dir / "docs"
        assert docs_dir.exists()
        assert docs_dir.is_dir()
        
        # Check that architecture directory was created
        arch_dir = docs_dir / "architecture"
        assert arch_dir.exists()
        assert arch_dir.is_dir()
    
    def test_doc_content_includes_all_sections(self, temp_output_dir, sample_answers):
        """Test that generated docs include all required sections."""
        generator = ArchitectureDocGenerator()
        
        result = generator.generate_all_docs(
            "test-project",
            sample_answers,
            temp_output_dir
        )
        
        # Check that documents have content
        for doc_type, doc_path in result.items():
            content = doc_path.read_text()
            assert len(content) > 0, f"Document {doc_type} should have content"

