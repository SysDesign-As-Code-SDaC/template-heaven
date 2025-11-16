"""
Tests for Architecture Questionnaire API endpoints.

This module contains tests for the API endpoints that allow AI/LLM
integration to auto-fill architecture questionnaires.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status

from templateheaven.api.main import create_app
from templateheaven.core.architecture_questionnaire import (
    ArchitectureAnswers,
    ArchitecturePattern,
    DeploymentModel,
    ScalabilityRequirement,
    SecurityLevel
)


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_questionnaire_fill_request():
    """Sample request for filling questionnaire."""
    return {
        "project_name": "test-project",
        "project_description": "A test project for microservices",
        "project_type": "backend",
        "target_audience": "enterprise",
        "expected_scale": "high",
        "additional_context": "Need high availability and security"
    }


class TestArchitectureAPI:
    """Test Architecture API endpoints."""
    
    def test_get_questionnaire_structure(self, client):
        """Test getting questionnaire structure."""
        response = client.get("/api/v1/architecture/questionnaire/structure")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "questions" in data["data"]
        assert isinstance(data["data"]["questions"], list)
        assert len(data["data"]["questions"]) > 0
    
    def test_get_questionnaire_by_category(self, client):
        """Test getting questions by category."""
        response = client.get("/api/v1/architecture/questionnaire/category/project_overview")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "questions" in data["data"]
        
        # Verify all questions are in the requested category
        questions = data["data"]["questions"]
        for question in questions:
            assert question["category"] == "project_overview"
    
    def test_get_questionnaire_by_category_invalid(self, client):
        """Test getting questions with invalid category."""
        response = client.get("/api/v1/architecture/questionnaire/category/invalid_category")
        
        # Should return empty list or 404
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
    
    @patch('templateheaven.api.routes.architecture.ArchitectureQuestionnaire')
    def test_fill_questionnaire_with_ai(self, mock_questionnaire_class, client, sample_questionnaire_fill_request):
        """Test filling questionnaire using AI/LLM."""
        # Mock the questionnaire instance
        mock_questionnaire = Mock()
        mock_questionnaire_class.return_value = mock_questionnaire
        
        # Mock AI fill method
        mock_answers = ArchitectureAnswers(
            project_vision="Build scalable microservices",
            architecture_patterns=[ArchitecturePattern.MICROSERVICES],
            deployment_model=DeploymentModel.CLOUD_NATIVE
        )
        mock_questionnaire.fill_with_ai = Mock(return_value=mock_answers)
        
        response = client.post(
            "/api/v1/architecture/questionnaire/fill",
            json=sample_questionnaire_fill_request
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "answers" in data["data"]
        
        # Verify AI fill was called
        mock_questionnaire.fill_with_ai.assert_called_once()
    
    def test_fill_questionnaire_validation_error(self, client):
        """Test filling questionnaire with invalid data."""
        invalid_request = {
            "project_name": ""  # Empty name should fail validation
        }
        
        response = client.post(
            "/api/v1/architecture/questionnaire/fill",
            json=invalid_request
        )
        
        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_validate_answers(self, client):
        """Test validating questionnaire answers."""
        answers_data = {
            "project_vision": "Test vision",
            "target_users": "Test users",
            "architecture_patterns": ["microservices"],
            "deployment_model": "cloud-native"
        }
        
        response = client.post(
            "/api/v1/architecture/questionnaire/validate",
            json={"answers": answers_data}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "success" in data
        assert "data" in data
        assert "valid" in data["data"]
        assert isinstance(data["data"]["valid"], bool)
    
    def test_validate_answers_invalid(self, client):
        """Test validating invalid answers."""
        invalid_answers = {
            "architecture_patterns": ["invalid_pattern"]  # Invalid pattern
        }
        
        response = client.post(
            "/api/v1/architecture/questionnaire/validate",
            json={"answers": invalid_answers}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should indicate validation failed
        assert "success" in data
        assert "data" in data
        assert "valid" in data["data"]
        # May be False or may have errors list
        if "errors" in data["data"]:
            assert len(data["data"]["errors"]) > 0
    
    def test_get_question_by_key(self, client):
        """Test getting a specific question by key."""
        response = client.get("/api/v1/architecture/questionnaire/question/project_vision")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "question" in data["data"]
        assert data["data"]["question"]["key"] == "project_vision"
    
    def test_get_question_by_key_not_found(self, client):
        """Test getting non-existent question."""
        response = client.get("/api/v1/architecture/questionnaire/question/non_existent_key")
        
        # Should return 404 or empty result
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_200_OK]
    
    def test_api_response_format(self, client):
        """Test that API responses follow consistent format."""
        response = client.get("/api/v1/architecture/questionnaire/structure")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert "success" in data
        assert "data" in data
        assert isinstance(data["success"], bool)
        
        # Check that data contains expected fields
        if "questions" in data["data"]:
            assert isinstance(data["data"]["questions"], list)

