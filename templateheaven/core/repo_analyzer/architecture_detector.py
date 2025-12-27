"""
Architecture pattern detection for repositories.

Detects common architecture patterns (MVC, microservices, event-driven, etc.)
in open-source repositories by analyzing code structure and organization.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum

from .code_analyzer import RepositoryAnalysis
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ArchitecturePattern(Enum):
    """Common architecture patterns."""
    MVC = "mvc"
    MICROSERVICES = "microservices"
    MONOLITH = "monolith"
    EVENT_DRIVEN = "event-driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    CQRS = "cqrs"
    SERVERLESS = "serverless"
    API_FIRST = "api-first"
    SERVICE_MESH = "service-mesh"
    CLEAN_ARCHITECTURE = "clean-architecture"


@dataclass
class PatternDetection:
    """Result of pattern detection."""
    pattern: ArchitecturePattern
    confidence: float  # 0.0 to 1.0
    indicators: List[str]
    evidence: Dict[str, Any]


class ArchitectureDetector:
    """Detects architecture patterns in repositories."""
    
    def __init__(self):
        """Initialize architecture detector."""
        self.pattern_indicators = {
            ArchitecturePattern.MVC: {
                "directories": ["models", "views", "controllers", "controllers", "views"],
                "files": ["controller", "view", "model"],
                "keywords": ["mvc", "model-view-controller", "controller", "view", "model"]
            },
            ArchitecturePattern.MICROSERVICES: {
                "directories": ["services", "microservices", "api", "gateway"],
                "files": ["service", "microservice", "gateway"],
                "keywords": ["microservice", "service mesh", "api gateway", "service discovery"],
                "structure": ["multiple_services", "independent_deployment"]
            },
            ArchitecturePattern.EVENT_DRIVEN: {
                "directories": ["events", "handlers", "listeners", "publishers", "subscribers"],
                "files": ["event", "handler", "listener", "publisher", "subscriber"],
                "keywords": ["event", "message", "queue", "pub/sub", "event bus", "event sourcing"]
            },
            ArchitecturePattern.CQRS: {
                "directories": ["commands", "queries", "read", "write"],
                "files": ["command", "query", "read", "write"],
                "keywords": ["cqrs", "command query", "read model", "write model"]
            },
            ArchitecturePattern.HEXAGONAL: {
                "directories": ["domain", "ports", "adapters", "application"],
                "files": ["port", "adapter", "domain"],
                "keywords": ["hexagonal", "ports and adapters", "domain", "adapter"]
            },
            ArchitecturePattern.CLEAN_ARCHITECTURE: {
                "directories": ["entities", "use_cases", "interfaces", "frameworks"],
                "files": ["entity", "use_case", "interface"],
                "keywords": ["clean architecture", "use case", "entity", "boundary"]
            },
            ArchitecturePattern.SERVERLESS: {
                "directories": ["functions", "lambda", "handlers"],
                "files": ["function", "lambda", "handler"],
                "keywords": ["serverless", "lambda", "function", "faas", "cloud function"],
                "config": ["serverless.yml", "serverless.yaml", "template.yaml"]
            },
            ArchitecturePattern.API_FIRST: {
                "directories": ["api", "openapi", "swagger", "spec"],
                "files": ["openapi", "swagger", "api.yaml", "api.yml"],
                "keywords": ["openapi", "swagger", "api first", "api specification"]
            }
        }
        logger.info("ArchitectureDetector initialized")
    
    def detect_patterns(
        self,
        analysis: RepositoryAnalysis
    ) -> List[PatternDetection]:
        """
        Detect architecture patterns in repository.
        
        Args:
            analysis: RepositoryAnalysis object
            
        Returns:
            List of detected patterns with confidence scores
        """
        logger.info(f"Detecting architecture patterns for {analysis.repository_name}")
        
        detections = []
        structure = analysis.structure
        files = structure.get("files", [])
        directories = structure.get("directories", [])
        
        # Check each pattern
        for pattern, indicators in self.pattern_indicators.items():
            confidence, pattern_indicators, evidence = self._check_pattern(
                pattern,
                indicators,
                files,
                directories,
                analysis
            )
            
            if confidence > 0.3:  # Only include patterns with reasonable confidence
                detections.append(PatternDetection(
                    pattern=pattern,
                    confidence=confidence,
                    indicators=pattern_indicators,
                    evidence=evidence
                ))
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Detected {len(detections)} architecture patterns")
        return detections
    
    def _check_pattern(
        self,
        pattern: ArchitecturePattern,
        indicators: Dict[str, List[str]],
        files: List[str],
        directories: List[str],
        analysis: RepositoryAnalysis
    ) -> tuple[float, List[str], Dict[str, Any]]:
        """
        Check if a pattern matches the repository.
        
        Returns:
            Tuple of (confidence, indicators_found, evidence)
        """
        confidence = 0.0
        found_indicators = []
        evidence = {
            "directory_matches": [],
            "file_matches": [],
            "keyword_matches": [],
            "structure_evidence": {}
        }
        
        # Check directory structure
        dir_indicators = indicators.get("directories", [])
        dir_matches = 0
        for dir_name in dir_indicators:
            for directory in directories:
                if dir_name.lower() in directory.lower():
                    dir_matches += 1
                    found_indicators.append(f"Directory: {directory}")
                    evidence["directory_matches"].append(directory)
        
        if dir_indicators:
            confidence += (dir_matches / len(dir_indicators)) * 0.4
        
        # Check file names
        file_indicators = indicators.get("files", [])
        file_matches = 0
        for file_indicator in file_indicators:
            for file_path in files:
                if file_indicator.lower() in file_path.lower():
                    file_matches += 1
                    found_indicators.append(f"File: {file_path}")
                    evidence["file_matches"].append(file_path)
                    break  # Count each indicator once
        
        if file_indicators:
            confidence += (file_matches / len(file_indicators)) * 0.3
        
        # Check keywords in dependencies and tech stack
        keyword_indicators = indicators.get("keywords", [])
        keyword_matches = 0
        all_text = " ".join([
            str(analysis.technology_stack),
            str(analysis.dependencies),
            analysis.metadata.get("description", "")
        ]).lower()
        
        for keyword in keyword_indicators:
            if keyword.lower() in all_text:
                keyword_matches += 1
                found_indicators.append(f"Keyword: {keyword}")
                evidence["keyword_matches"].append(keyword)
        
        if keyword_indicators:
            confidence += (keyword_matches / len(keyword_indicators)) * 0.2
        
        # Check structure-specific indicators
        structure_indicators = indicators.get("structure", [])
        for struct_indicator in structure_indicators:
            if struct_indicator == "multiple_services":
                # Check for multiple service directories or independent modules
                service_dirs = [d for d in directories if "service" in d.lower()]
                if len(service_dirs) >= 2:
                    confidence += 0.1
                    found_indicators.append("Multiple service directories")
                    evidence["structure_evidence"]["multiple_services"] = True
            
            elif struct_indicator == "independent_deployment":
                # Check for Dockerfiles or deployment configs
                dockerfiles = [f for f in files if "dockerfile" in f.lower() or "docker-compose" in f.lower()]
                if len(dockerfiles) >= 2:
                    confidence += 0.1
                    found_indicators.append("Multiple deployment configs")
                    evidence["structure_evidence"]["independent_deployment"] = True
        
        # Check config files
        config_indicators = indicators.get("config", [])
        config_matches = 0
        for config_file in config_indicators:
            if any(config_file.lower() in f.lower() for f in files):
                config_matches += 1
                found_indicators.append(f"Config: {config_file}")
        
        if config_indicators:
            confidence += (config_matches / len(config_indicators)) * 0.1
        
        # Normalize confidence to 0.0-1.0
        confidence = min(1.0, confidence)
        
        return confidence, found_indicators, evidence
    
    def get_primary_pattern(
        self,
        detections: List[PatternDetection]
    ) -> Optional[PatternDetection]:
        """
        Get the primary architecture pattern.
        
        Args:
            detections: List of pattern detections
            
        Returns:
            Primary pattern or None
        """
        if not detections:
            return None
        
        # Return highest confidence pattern
        return detections[0]
    
    def get_pattern_compatibility(
        self,
        pattern: ArchitecturePattern,
        target_requirements: Dict[str, Any]
    ) -> float:
        """
        Assess compatibility of a pattern with target requirements.
        
        Args:
            pattern: Architecture pattern
            target_requirements: Target project requirements
            
        Returns:
            Compatibility score (0.0-1.0)
        """
        # Pattern-specific compatibility rules
        compatibility_rules = {
            ArchitecturePattern.MICROSERVICES: {
                "high_scale": 0.9,
                "large_team": 0.8,
                "independent_deployment": 0.9,
                "low_complexity": 0.3,
                "small_team": 0.2
            },
            ArchitecturePattern.MONOLITH: {
                "low_scale": 0.9,
                "small_team": 0.9,
                "rapid_development": 0.9,
                "high_scale": 0.3,
                "large_team": 0.2
            },
            ArchitecturePattern.EVENT_DRIVEN: {
                "async_processing": 0.9,
                "loose_coupling": 0.9,
                "real_time": 0.8,
                "simple_crud": 0.3
            },
            ArchitecturePattern.SERVERLESS: {
                "variable_traffic": 0.9,
                "cost_optimization": 0.9,
                "low_latency": 0.7,
                "long_running": 0.2
            }
        }
        
        rules = compatibility_rules.get(pattern, {})
        if not rules:
            return 0.5  # Neutral compatibility
        
        compatibility = 0.0
        matched_factors = 0
        
        for requirement, value in target_requirements.items():
            if requirement in rules:
                weight = rules[requirement]
                if isinstance(value, bool) and value:
                    compatibility += weight
                    matched_factors += 1
                elif isinstance(value, (int, float)) and value > 0:
                    # Normalize and weight
                    compatibility += weight * min(1.0, value / 10.0)
                    matched_factors += 1
        
        if matched_factors == 0:
            return 0.5
        
        return min(1.0, compatibility / matched_factors)

