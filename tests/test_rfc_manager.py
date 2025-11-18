"""
Tests for RFC Manager.

This module contains tests for managing Request for Comments (RFCs).
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from templateheaven.core.rfc_manager import RFCManager


class TestRFCManager:
    """Test RFCManager class."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    @pytest.fixture
    def rfc_manager(self, temp_project_dir):
        """Create an RFCManager instance."""
        return RFCManager(temp_project_dir)

    def test_rfc_manager_initialization(self, rfc_manager, temp_project_dir):
        """Test RFC manager initialization."""
        assert rfc_manager is not None
        assert rfc_manager.project_dir == temp_project_dir
        assert rfc_manager.rfc_dir.exists()
        assert rfc_manager.rfc_dir.name == "rfc"

    def test_create_rfc(self, rfc_manager):
        """Test creating an RFC."""
        rfc_path = rfc_manager.create_rfc(
            title="Multi-branch Architecture",
            summary="Propose a multi-branch architecture for organizing templates.",
            motivation="Current single-branch approach doesn't scale well.",
            design="Use separate branches for each technology stack.",
            alternatives=["Monorepo", "Separate repositories"],
            open_questions=["How to handle cross-stack dependencies?"],
        )

        assert rfc_path.exists()
        assert rfc_path.suffix == ".md"
        assert "RFC-0001" in rfc_path.name

        # Check content
        content = rfc_path.read_text()
        assert "RFC-0001" in content
        assert "Multi-branch Architecture" in content
        assert "Multi-branch" in content

    def test_create_multiple_rfcs(self, rfc_manager):
        """Test creating multiple RFCs with sequential numbering."""
        rfc1 = rfc_manager.create_rfc(
            title="First RFC",
            summary="First summary",
            motivation="First motivation",
            design="First design",
        )

        rfc2 = rfc_manager.create_rfc(
            title="Second RFC",
            summary="Second summary",
            motivation="Second motivation",
            design="Second design",
        )

        assert "RFC-0001" in rfc1.name
        assert "RFC-0002" in rfc2.name

    def test_list_rfcs(self, rfc_manager):
        """Test listing RFCs."""
        # Create some RFCs
        rfc_manager.create_rfc(
            title="Test RFC 1",
            summary="Summary 1",
            motivation="Motivation 1",
            design="Design 1",
        )

        rfc_manager.create_rfc(
            title="Test RFC 2",
            summary="Summary 2",
            motivation="Motivation 2",
            design="Design 2",
        )

        rfcs = rfc_manager.list_rfcs()
        assert len(rfcs) == 2
        assert all("number" in rfc for rfc in rfcs)
        assert all("title" in rfc for rfc in rfcs)

    def test_get_rfc(self, rfc_manager):
        """Test getting RFC by number."""
        rfc_manager.create_rfc(
            title="Test RFC",
            summary="Summary",
            motivation="Motivation",
            design="Design",
        )

        rfc_path = rfc_manager.get_rfc(1)
        assert rfc_path is not None
        assert rfc_path.exists()

        # Test non-existent RFC
        assert rfc_manager.get_rfc(999) is None

    def test_update_rfc_status(self, rfc_manager):
        """Test updating RFC status."""
        rfc_manager.create_rfc(
            title="Test RFC",
            summary="Summary",
            motivation="Motivation",
            design="Design",
            status="Draft",
        )

        # Update status
        success = rfc_manager.update_rfc_status(1, "Accepted")
        assert success is True

        # Verify status was updated
        rfc_path = rfc_manager.get_rfc(1)
        content = rfc_path.read_text()
        assert "## Status" in content
        assert "Accepted" in content

    def test_update_rfc_status_invalid(self, rfc_manager):
        """Test updating RFC status with invalid status."""
        rfc_manager.create_rfc(
            title="Test RFC",
            summary="Summary",
            motivation="Motivation",
            design="Design",
        )

        with pytest.raises(ValueError):
            rfc_manager.update_rfc_status(1, "InvalidStatus")

    def test_rfc_with_alternatives(self, rfc_manager):
        """Test creating RFC with alternatives."""
        alternatives = ["Option A", "Option B", "Option C"]
        rfc_path = rfc_manager.create_rfc(
            title="Test RFC",
            summary="Summary",
            motivation="Motivation",
            design="Design",
            alternatives=alternatives,
        )

        content = rfc_path.read_text()
        assert "Alternatives" in content
        for alt in alternatives:
            assert alt in content

    def test_rfc_with_open_questions(self, rfc_manager):
        """Test creating RFC with open questions."""
        questions = ["Question 1", "Question 2"]
        rfc_path = rfc_manager.create_rfc(
            title="Test RFC",
            summary="Summary",
            motivation="Motivation",
            design="Design",
            open_questions=questions,
        )

        content = rfc_path.read_text()
        assert "Open Questions" in content
        for question in questions:
            assert question in content

    def test_rfc_without_optional_fields(self, rfc_manager):
        """Test creating RFC without optional fields."""
        rfc_path = rfc_manager.create_rfc(
            title="Test RFC",
            summary="Summary",
            motivation="Motivation",
            design="Design",
        )

        content = rfc_path.read_text()
        # Should still be valid even without alternatives or questions
        assert "RFC-0001" in content
        assert "Test RFC" in content

    def test_slugify_title(self, rfc_manager):
        """Test title slugification."""
        rfc_path = rfc_manager.create_rfc(
            title="Complex Title with Special Chars!",
            summary="Summary",
            motivation="Motivation",
            design="Design",
        )

        # Check filename is properly slugified
        assert "complex-title-with-special-chars" in rfc_path.name.lower()

    def test_get_next_number_empty(self, rfc_manager):
        """Test getting next number when no RFCs exist."""
        number = rfc_manager._get_next_number()
        assert number == 1

    def test_get_next_number_with_existing(self, rfc_manager):
        """Test getting next number with existing RFCs."""
        rfc_manager.create_rfc(
            title="First",
            summary="Summary",
            motivation="Motivation",
            design="Design",
        )
        rfc_manager.create_rfc(
            title="Second",
            summary="Summary",
            motivation="Motivation",
            design="Design",
        )

        number = rfc_manager._get_next_number()
        assert number == 3

    def test_parse_rfc_metadata(self, rfc_manager):
        """Test parsing RFC metadata."""
        rfc_path = rfc_manager.create_rfc(
            title="Test RFC",
            summary="Summary",
            motivation="Motivation",
            design="Design",
            status="Review",
        )

        metadata = rfc_manager._parse_rfc_metadata(rfc_path)
        assert metadata["number"] == 1
        assert metadata["title"] == "Test RFC"
        assert metadata["status"] == "Review"
        assert metadata["path"] == rfc_path

