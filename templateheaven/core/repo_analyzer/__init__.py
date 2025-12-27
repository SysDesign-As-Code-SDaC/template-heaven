"""
Repository analysis module for Template Heaven.

Provides deep code analysis, architecture pattern detection, and integration
recommendations for open-source repositories.
"""

from .code_analyzer import RepositoryAnalyzer, RepositoryAnalysis
from .architecture_detector import ArchitectureDetector, ArchitecturePattern
from .integration_recommender import IntegrationRecommender, IntegrationRecommendation

__all__ = [
    "RepositoryAnalyzer",
    "RepositoryAnalysis",
    "ArchitectureDetector",
    "ArchitecturePattern",
    "IntegrationRecommender",
    "IntegrationRecommendation",
]

