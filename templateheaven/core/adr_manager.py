"""
Architecture Decision Records (ADR) Manager for Template Heaven.

Manages ADRs following the MADR (Markdown ADR) format.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from jinja2 import Template, FileSystemLoader, Environment

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ADRManager:
    """Manages Architecture Decision Records (ADRs)."""

    ADR_STATUSES = ["Proposed", "Accepted", "Deprecated", "Superseded"]

    def __init__(self, project_dir: Path):
        """
        Initialize ADR Manager.

        Args:
            project_dir: Root directory of the project
        """
        self.project_dir = Path(project_dir)
        self.adr_dir = self.project_dir / "docs" / "adr"
        self.adr_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def create_adr(
        self,
        title: str,
        context: str,
        decision: str,
        consequences: str,
        alternatives: Optional[List[str]] = None,
        status: str = "Proposed",
    ) -> Path:
        """
        Create a new ADR.

        Args:
            title: Title of the ADR
            context: Context explaining the issue
            decision: The decision that was made
            consequences: Positive and negative consequences
            alternatives: List of alternatives considered
            status: Status of the ADR (default: Proposed)

        Returns:
            Path to the created ADR file
        """
        if status not in self.ADR_STATUSES:
            raise ValueError(f"Invalid status. Must be one of: {self.ADR_STATUSES}")

        # Get next ADR number
        adr_number = self._get_next_number()

        # Load template
        template = self._load_template()

        # Render ADR
        content = template.render(
            number=adr_number,
            number_formatted=f"{adr_number:04d}",
            title=title,
            status=status,
            context=context,
            decision=decision,
            consequences=consequences,
            alternatives=alternatives or [],
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Write ADR file
        filename = f"{adr_number:04d}-{self._slugify(title)}.md"
        adr_path = self.adr_dir / filename
        adr_path.write_text(content, encoding="utf-8")

        self.logger.info(f"Created ADR {adr_number:04d}: {title}")
        return adr_path

    def list_adrs(self) -> List[Dict[str, any]]:
        """
        List all ADRs.

        Returns:
            List of ADR metadata dictionaries
        """
        adrs = []
        if not self.adr_dir.exists():
            return adrs

        for adr_file in sorted(self.adr_dir.glob("*.md")):
            try:
                metadata = self._parse_adr_metadata(adr_file)
                adrs.append(metadata)
            except Exception as e:
                self.logger.warning(f"Failed to parse ADR {adr_file}: {e}")

        return adrs

    def get_adr(self, number: int) -> Optional[Path]:
        """
        Get ADR by number.

        Args:
            number: ADR number

        Returns:
            Path to ADR file, or None if not found
        """
        pattern = f"{number:04d}-*.md"
        matches = list(self.adr_dir.glob(pattern))
        return matches[0] if matches else None

    def update_adr_status(self, number: int, status: str) -> bool:
        """
        Update ADR status.

        Args:
            number: ADR number
            status: New status

        Returns:
            True if updated successfully, False otherwise
        """
        if status not in self.ADR_STATUSES:
            raise ValueError(f"Invalid status. Must be one of: {self.ADR_STATUSES}")

        adr_path = self.get_adr(number)
        if not adr_path:
            return False

        # Read current content
        content = adr_path.read_text(encoding="utf-8")

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
        adr_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Updated ADR {number:04d} status to {status}")
        return True

    def _get_next_number(self) -> int:
        """Get the next ADR number."""
        existing_numbers = []
        if self.adr_dir.exists():
            for adr_file in self.adr_dir.glob("*.md"):
                try:
                    # Extract number from filename (e.g., "0001-title.md")
                    number_str = adr_file.stem.split("-")[0]
                    existing_numbers.append(int(number_str))
                except (ValueError, IndexError):
                    continue

        return max(existing_numbers, default=0) + 1

    def _load_template(self) -> Template:
        """Load ADR template."""
        # Try to load from file first
        template_path = Path(__file__).parent / "templates" / "adr_template.j2"
        if template_path.exists():
            env = Environment(loader=FileSystemLoader(str(template_path.parent)))
            return env.get_template("adr_template.j2")
        
        # Fallback to inline template
        template_content = """# ADR-{{ number_formatted }}: {{ title }}

## Status

{{ status }}

## Context

{{ context }}

## Decision

{{ decision }}

## Consequences

{{ consequences }}

{% if alternatives %}
## Alternatives Considered

{% for alternative in alternatives %}
- {{ alternative }}
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

    def _parse_adr_metadata(self, adr_path: Path) -> Dict[str, any]:
        """
        Parse ADR metadata from file.

        Args:
            adr_path: Path to ADR file

        Returns:
            Dictionary with ADR metadata
        """
        content = adr_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        metadata = {
            "path": adr_path,
            "number": None,
            "title": None,
            "status": None,
            "date": None,
        }

        # Extract number from filename
        try:
            number_str = adr_path.stem.split("-")[0]
            metadata["number"] = int(number_str)
        except (ValueError, IndexError):
            pass

        # Parse content
        for i, line in enumerate(lines):
            if line.startswith("# ADR-"):
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

