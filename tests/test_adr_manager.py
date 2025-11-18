"""
Tests for ADR Manager.

This module contains tests for managing Architecture Decision Records.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from templateheaven.core.adr_manager import ADRManager


class TestADRManager:
    """Test ADRManager class."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path)

    @pytest.fixture
    def adr_manager(self, temp_project_dir):
        """Create an ADRManager instance."""
        return ADRManager(temp_project_dir)

    def test_adr_manager_initialization(self, adr_manager, temp_project_dir):
        """Test ADR manager initialization."""
        assert adr_manager is not None
        assert adr_manager.project_dir == temp_project_dir
        assert adr_manager.adr_dir.exists()
        assert adr_manager.adr_dir.name == "adr"

    def test_create_adr(self, adr_manager):
        """Test creating an ADR."""
        adr_path = adr_manager.create_adr(
            title="Use FastAPI for API",
            context="We need to choose a web framework for our API.",
            decision="We will use FastAPI for its async support and automatic documentation.",
            consequences="FastAPI provides excellent performance and developer experience.",
            alternatives=["Flask", "Django REST Framework"],
        )

        assert adr_path.exists()
        assert adr_path.suffix == ".md"
        assert "0001" in adr_path.name

        # Check content
        content = adr_path.read_text()
        assert "ADR-0001" in content
        assert "Use FastAPI for API" in content
        assert "FastAPI" in content

    def test_create_multiple_adrs(self, adr_manager):
        """Test creating multiple ADRs with sequential numbering."""
        adr1 = adr_manager.create_adr(
            title="First Decision",
            context="First context",
            decision="First decision",
            consequences="First consequences",
        )

        adr2 = adr_manager.create_adr(
            title="Second Decision",
            context="Second context",
            decision="Second decision",
            consequences="Second consequences",
        )

        assert "0001" in adr1.name
        assert "0002" in adr2.name

    def test_list_adrs(self, adr_manager):
        """Test listing ADRs."""
        # Create some ADRs
        adr_manager.create_adr(
            title="Test ADR 1",
            context="Context 1",
            decision="Decision 1",
            consequences="Consequences 1",
        )

        adr_manager.create_adr(
            title="Test ADR 2",
            context="Context 2",
            decision="Decision 2",
            consequences="Consequences 2",
        )

        adrs = adr_manager.list_adrs()
        assert len(adrs) == 2
        assert all("number" in adr for adr in adrs)
        assert all("title" in adr for adr in adrs)

    def test_get_adr(self, adr_manager):
        """Test getting ADR by number."""
        adr_manager.create_adr(
            title="Test ADR",
            context="Context",
            decision="Decision",
            consequences="Consequences",
        )

        adr_path = adr_manager.get_adr(1)
        assert adr_path is not None
        assert adr_path.exists()

        # Test non-existent ADR
        assert adr_manager.get_adr(999) is None

    def test_update_adr_status(self, adr_manager):
        """Test updating ADR status."""
        adr_manager.create_adr(
            title="Test ADR",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            status="Proposed",
        )

        # Update status
        success = adr_manager.update_adr_status(1, "Accepted")
        assert success is True

        # Verify status was updated
        adr_path = adr_manager.get_adr(1)
        content = adr_path.read_text()
        assert "## Status" in content
        assert "Accepted" in content

    def test_update_adr_status_invalid(self, adr_manager):
        """Test updating ADR status with invalid status."""
        adr_manager.create_adr(
            title="Test ADR",
            context="Context",
            decision="Decision",
            consequences="Consequences",
        )

        with pytest.raises(ValueError):
            adr_manager.update_adr_status(1, "InvalidStatus")

    def test_adr_with_alternatives(self, adr_manager):
        """Test creating ADR with alternatives."""
        alternatives = ["Option A", "Option B", "Option C"]
        adr_path = adr_manager.create_adr(
            title="Test ADR",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            alternatives=alternatives,
        )

        content = adr_path.read_text()
        assert "Alternatives Considered" in content
        for alt in alternatives:
            assert alt in content

    def test_adr_without_alternatives(self, adr_manager):
        """Test creating ADR without alternatives."""
        adr_path = adr_manager.create_adr(
            title="Test ADR",
            context="Context",
            decision="Decision",
            consequences="Consequences",
        )

        content = adr_path.read_text()
        # Should not have alternatives section if none provided
        assert "Alternatives Considered" not in content

    def test_slugify_title(self, adr_manager):
        """Test title slugification."""
        adr_path = adr_manager.create_adr(
            title="Complex Title with Special Chars!",
            context="Context",
            decision="Decision",
            consequences="Consequences",
        )

        # Check filename is properly slugified
        assert "complex-title-with-special-chars" in adr_path.name.lower()

    def test_get_next_number_empty(self, adr_manager):
        """Test getting next number when no ADRs exist."""
        number = adr_manager._get_next_number()
        assert number == 1

    def test_get_next_number_with_existing(self, adr_manager):
        """Test getting next number with existing ADRs."""
        adr_manager.create_adr(
            title="First",
            context="Context",
            decision="Decision",
            consequences="Consequences",
        )
        adr_manager.create_adr(
            title="Second",
            context="Context",
            decision="Decision",
            consequences="Consequences",
        )

        number = adr_manager._get_next_number()
        assert number == 3

    def test_parse_adr_metadata(self, adr_manager):
        """Test parsing ADR metadata."""
        adr_path = adr_manager.create_adr(
            title="Test ADR",
            context="Context",
            decision="Decision",
            consequences="Consequences",
            status="Accepted",
        )

        metadata = adr_manager._parse_adr_metadata(adr_path)
        assert metadata["number"] == 1
        assert metadata["title"] == "Test ADR"
        assert metadata["status"] == "Accepted"
        assert metadata["path"] == adr_path

