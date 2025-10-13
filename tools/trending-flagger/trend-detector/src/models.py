from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import List

class TrendLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TemplateType(Enum):
    FRAMEWORK = "framework"
    LIBRARY = "library"
    TOOL = "tool"
    TEMPLATE = "template"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"

@dataclass
class RepositoryMetrics:
    """Repository metrics for trend analysis."""
    stars: int
    forks: int
    watchers: int
    issues: int
    pull_requests: int
    commits: int
    contributors: int
    created_at: datetime
    updated_at: datetime
    language: str
    topics: List[str]
    size: int
    license: str

@dataclass
class TrendAlert:
    """Trend alert for human review."""
    repository_url: str
    repository_name: str
    trend_level: TrendLevel
    trend_score: float
    metrics: RepositoryMetrics
    trend_reasons: List[str]
    priority_score: float
    created_at: datetime
    template_type: TemplateType
    human_review_required: bool