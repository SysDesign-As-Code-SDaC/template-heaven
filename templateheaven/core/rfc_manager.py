"""
RFC (Request for Comments) Manager for Template Heaven.

Manages RFCs for proposing and documenting major changes.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from jinja2 import Template, FileSystemLoader, Environment

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RFCManager:
    """Manages RFCs (Request for Comments)."""

    RFC_STATUSES = ["Draft", "Review", "Accepted", "Rejected"]

    def __init__(self, project_dir: Path):
        """
        Initialize RFC Manager.

        Args:
            project_dir: Root directory of the project
        """
        self.project_dir = Path(project_dir)
        self.rfc_dir = self.project_dir / "docs" / "rfc"
        self.rfc_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def create_rfc(
        self,
        title: str,
        summary: str,
        motivation: str,
        design: str,
        alternatives: Optional[List[str]] = None,
        open_questions: Optional[List[str]] = None,
        status: str = "Draft",
    ) -> Path:
        """
        Create a new RFC.

        Args:
            title: Title of the RFC
            summary: Brief summary of the RFC
            motivation: Why this RFC is needed
            design: Detailed design description
            alternatives: List of alternatives considered
            open_questions: List of open questions
            status: Status of the RFC (default: Draft)

        Returns:
            Path to the created RFC file
        """
        if status not in self.RFC_STATUSES:
            raise ValueError(f"Invalid status. Must be one of: {self.RFC_STATUSES}")

        # Get next RFC number
        rfc_number = self._get_next_number()

        # Load template
        template = self._load_template()

        # Render RFC
        content = template.render(
            number=rfc_number,
            number_formatted=f"{rfc_number:04d}",
            title=title,
            status=status,
            summary=summary,
            motivation=motivation,
            design=design,
            alternatives=alternatives or [],
            open_questions=open_questions or [],
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Write RFC file
        filename = f"RFC-{rfc_number:04d}-{self._slugify(title)}.md"
        rfc_path = self.rfc_dir / filename
        rfc_path.write_text(content, encoding="utf-8")

        self.logger.info(f"Created RFC {rfc_number:04d}: {title}")
        return rfc_path

    def list_rfcs(self) -> List[Dict[str, any]]:
        """
        List all RFCs.

        Returns:
            List of RFC metadata dictionaries
        """
        rfcs = []
        if not self.rfc_dir.exists():
            return rfcs

        for rfc_file in sorted(self.rfc_dir.glob("*.md")):
            try:
                metadata = self._parse_rfc_metadata(rfc_file)
                rfcs.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to parse RFC {rfc_file}: {e}")

        return rfcs

    def get_rfc(self, number: int) -> Optional[Path]:
        """
        Get RFC by number.

        Args:
            number: RFC number

        Returns:
            Path to RFC file, or None if not found
        """
        pattern = f"RFC-{number:04d}-*.md"
        matches = list(self.rfc_dir.glob(pattern))
        return matches[0] if matches else None

    def update_rfc_status(self, number: int, status: str) -> bool:
        """
        Update RFC status.

        Args:
            number: RFC number
            status: New status

        Returns:
            True if updated successfully, False otherwise
        """
        if status not in self.RFC_STATUSES:
            raise ValueError(f"Invalid status. Must be one of: {self.RFC_STATUSES}")

        rfc_path = self.get_rfc(number)
        if not rfc_path:
            return False

        # Read current content
        content = rfc_path.read_text(encoding="utf-8")

        # Update status line
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("## Status"):
                # Update next non-empty line
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        lines[j] = status
                        break
                break

        # Write updated content
        rfc_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Updated RFC {number:04d} status to {status}")
        return True

    def _get_next_number(self) -> int:
        """Get the next RFC number."""
        existing_numbers = []
        if self.rfc_dir.exists():
            for rfc_file in self.rfc_dir.glob("*.md"):
                try:
                    # Extract number from filename (e.g., "RFC-0001-title.md")
                    parts = rfc_file.stem.split("-")
                    if len(parts) >= 2 and parts[0] == "RFC":
                        existing_numbers.append(int(parts[1]))
                except (ValueError, IndexError):
                    continue

        return max(existing_numbers, default=0) + 1

    def _load_template(self) -> Template:
        """Load RFC template."""
        # Try to load from file first
        template_path = Path(__file__).parent / "templates" / "rfc_template.j2"
        if template_path.exists():
            env = Environment(loader=FileSystemLoader(str(template_path.parent)))
            return env.get_template("rfc_template.j2")

        # Fallback to inline template
        template_content = """# RFC-{{ number_formatted }}: {{ title }}

## Status

{{ status }}

## Summary

{{ summary }}

## Motivation

{{ motivation }}

## Detailed Design

{{ design }}

{% if alternatives %}
## Alternatives

{% for alternative in alternatives %}
- {{ alternative }}
{% endfor %}
{% endif %}

{% if open_questions %}
## Open Questions

{% for question in open_questions %}
- {{ question }}
{% endfor %}
{% endif %}

---

**Date:** {{ date }}
"""
        return Template(template_content)

    def _slugify(self, title: str) -> str:
        """Convert title to URL-friendly slug."""
        import re
        slug = title.lower()
        # Replace non-alphanumeric characters with a hyphen
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        return slug[:50]  # Limit length

    def _parse_rfc_metadata(self, rfc_path: Path) -> Dict[str, any]:
        """
        Parse RFC metadata from file.

        Args:
            rfc_path: Path to RFC file

        Returns:
            Dictionary with RFC metadata
        """
        content = rfc_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        metadata = {
            "path": rfc_path,
            "number": None,
            "title": None,
            "status": None,
            "date": None,
        }

        # Extract number from filename
        try:
            parts = rfc_path.stem.split("-")
            if len(parts) >= 2 and parts[0] == "RFC":
                metadata["number"] = int(parts[1])
        except (ValueError, IndexError):
            pass

        # Parse content
        for i, line in enumerate(lines):
            if line.startswith("# RFC-"):
                # Extract title
                parts = line.split(":", 1)
                if len(parts) == 2:
                    metadata["title"] = parts[1].strip()
            elif line.startswith("## Status"):
                # Get status from next non-empty line
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        metadata["status"] = lines[j].strip()
                        break
            elif line.startswith("**Date:**"):
                metadata["date"] = line.split(":", 1)[1].strip() if ":" in line else None

        return metadata

