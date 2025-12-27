"""
Integration recommendation engine.

Matches repository capabilities with project requirements and suggests
integration approaches (library, service, microservice, etc.).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from .code_analyzer import RepositoryAnalysis
from .architecture_detector import ArchitectureDetector, ArchitecturePattern, PatternDetection
from ...utils.logger import get_logger

logger = get_logger(__name__)


class IntegrationApproach(Enum):
    """Types of integration approaches."""
    NPM_PACKAGE = "npm_package"
    PYTHON_PACKAGE = "python_package"
    RUST_CRATE = "rust_crate"
    GO_MODULE = "go_module"
    DOCKER_SERVICE = "docker_service"
    MICROSERVICE = "microservice"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    STANDALONE_SERVICE = "standalone_service"


class IntegrationEffort(Enum):
    """Integration effort levels."""
    LOW = "low"  # < 1 hour
    MEDIUM = "medium"  # 1-4 hours
    HIGH = "high"  # 4-16 hours
    VERY_HIGH = "very_high"  # > 16 hours


@dataclass
class IntegrationRecommendation:
    """Integration recommendation for a repository."""
    repository: Dict[str, Any]
    repository_analysis: RepositoryAnalysis
    relevance_score: float  # 0.0 to 1.0
    use_case: str
    integration_approach: IntegrationApproach
    effort: IntegrationEffort
    compatibility: float  # 0.0 to 1.0
    code_example: Optional[str] = None
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    integration_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repository": self.repository,
            "relevance_score": self.relevance_score,
            "use_case": self.use_case,
            "integration_approach": self.integration_approach.value,
            "effort": self.effort.value,
            "compatibility": self.compatibility,
            "code_example": self.code_example,
            "alternatives": self.alternatives,
            "pros": self.pros,
            "cons": self.cons,
            "integration_steps": self.integration_steps
        }


class IntegrationRecommender:
    """Recommends integration strategies for repositories."""
    
    def __init__(self, architecture_detector: Optional[ArchitectureDetector] = None):
        """
        Initialize integration recommender.
        
        Args:
            architecture_detector: Optional architecture detector instance
        """
        self.architecture_detector = architecture_detector or ArchitectureDetector()
        logger.info("IntegrationRecommender initialized")
    
    def recommend_integration(
        self,
        repository_analysis: RepositoryAnalysis,
        requirements: Dict[str, Any],
        repository_metadata: Optional[Dict[str, Any]] = None
    ) -> IntegrationRecommendation:
        """
        Recommend integration approach for a repository.
        
        Args:
            repository_analysis: Analysis of the repository
            requirements: Project requirements
            repository_metadata: Optional repository metadata
            
        Returns:
            IntegrationRecommendation object
        """
        logger.info(f"Generating integration recommendation for {repository_analysis.repository_name}")
        
        # Determine integration approach
        approach = self._determine_approach(repository_analysis, requirements)
        
        # Calculate relevance score
        relevance = self._calculate_relevance(repository_analysis, requirements)
        
        # Assess compatibility
        compatibility = self._assess_compatibility(repository_analysis, requirements)
        
        # Estimate effort
        effort = self._estimate_effort(repository_analysis, approach)
        
        # Identify use case
        use_case = self._identify_use_case(repository_analysis, requirements)
        
        # Generate code example
        code_example = self._generate_code_example(repository_analysis, approach)
        
        # Generate integration steps
        integration_steps = self._generate_integration_steps(repository_analysis, approach)
        
        # Identify pros and cons
        pros, cons = self._identify_pros_cons(repository_analysis, approach, requirements)
        
        # Build repository info
        repo_info = {
            "name": repository_analysis.repository_name,
            "url": repository_analysis.repository_url,
            "stars": repository_analysis.metadata.get("stars", 0),
            "license": repository_analysis.metadata.get("license"),
            "technology_stack": repository_analysis.technology_stack
        }
        
        if repository_metadata:
            repo_info.update(repository_metadata)
        
        recommendation = IntegrationRecommendation(
            repository=repo_info,
            repository_analysis=repository_analysis,
            relevance_score=relevance,
            use_case=use_case,
            integration_approach=approach,
            effort=effort,
            compatibility=compatibility,
            code_example=code_example,
            integration_steps=integration_steps,
            pros=pros,
            cons=cons
        )
        
        logger.info(f"Generated recommendation with relevance {relevance:.2f} and compatibility {compatibility:.2f}")
        return recommendation
    
    def _determine_approach(
        self,
        analysis: RepositoryAnalysis,
        requirements: Dict[str, Any]
    ) -> IntegrationApproach:
        """Determine the best integration approach."""
        tech_stack = analysis.technology_stack
        dependencies = analysis.dependencies
        structure = analysis.structure
        
        # Check for package managers
        if "package.json" in dependencies:
            return IntegrationApproach.NPM_PACKAGE
        elif "requirements.txt" in dependencies or "pyproject.toml" in dependencies:
            return IntegrationApproach.PYTHON_PACKAGE
        elif "Cargo.toml" in dependencies:
            return IntegrationApproach.RUST_CRATE
        elif "go.mod" in dependencies:
            return IntegrationApproach.GO_MODULE
        
        # Check for Docker
        if any("dockerfile" in f.lower() for f in structure.get("files", [])):
            return IntegrationApproach.DOCKER_SERVICE
        
        # Check architecture patterns
        patterns = self.architecture_detector.detect_patterns(analysis)
        if patterns:
            primary = self.architecture_detector.get_primary_pattern(patterns)
            if primary and primary.pattern == ArchitecturePattern.MICROSERVICES:
                return IntegrationApproach.MICROSERVICE
        
        # Default to library
        return IntegrationApproach.LIBRARY
    
    def _calculate_relevance(
        self,
        analysis: RepositoryAnalysis,
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate relevance score (0.0-1.0)."""
        score = 0.0
        
        # Match technology stack
        required_stack = requirements.get("technology_stack", [])
        if required_stack:
            matches = sum(1 for tech in required_stack if tech in analysis.technology_stack)
            score += (matches / len(required_stack)) * 0.4
        
        # Match use case keywords
        use_case = requirements.get("use_case", "").lower()
        description = analysis.metadata.get("description", "").lower()
        if use_case and use_case in description:
            score += 0.3
        
        # Check quality metrics
        quality = analysis.quality_metrics.get("completeness_score", 0.0)
        score += quality * 0.2
        
        # Check stars (popularity indicator)
        stars = analysis.metadata.get("stars", 0)
        if stars > 1000:
            score += 0.1
        elif stars > 100:
            score += 0.05
        
        return min(1.0, score)
    
    def _assess_compatibility(
        self,
        analysis: RepositoryAnalysis,
        requirements: Dict[str, Any]
    ) -> float:
        """Assess compatibility with project requirements."""
        compatibility = 0.5  # Start neutral
        
        # Technology stack compatibility
        required_stack = requirements.get("technology_stack", [])
        if required_stack:
            stack_matches = sum(1 for tech in required_stack if tech in analysis.technology_stack)
            if stack_matches > 0:
                compatibility += 0.2
        
        # License compatibility
        license_type = analysis.metadata.get("license", "").lower()
        if "mit" in license_type or "apache" in license_type or "bsd" in license_type:
            compatibility += 0.1
        
        # Architecture pattern compatibility
        patterns = self.architecture_detector.detect_patterns(analysis)
        if patterns:
            primary = self.architecture_detector.get_primary_pattern(patterns)
            if primary:
                pattern_compat = self.architecture_detector.get_pattern_compatibility(
                    primary.pattern,
                    requirements
                )
                compatibility += pattern_compat * 0.2
        
        return min(1.0, compatibility)
    
    def _estimate_effort(
        self,
        analysis: RepositoryAnalysis,
        approach: IntegrationApproach
    ) -> IntegrationEffort:
        """Estimate integration effort."""
        # Simple heuristics based on approach and complexity
        if approach in [IntegrationApproach.NPM_PACKAGE, IntegrationApproach.PYTHON_PACKAGE]:
            return IntegrationEffort.LOW
        elif approach == IntegrationApproach.DOCKER_SERVICE:
            return IntegrationEffort.MEDIUM
        elif approach == IntegrationApproach.MICROSERVICE:
            return IntegrationEffort.HIGH
        else:
            # Estimate based on complexity
            file_count = len(analysis.structure.get("files", []))
            if file_count < 10:
                return IntegrationEffort.LOW
            elif file_count < 50:
                return IntegrationEffort.MEDIUM
            else:
                return IntegrationEffort.HIGH
    
    def _identify_use_case(
        self,
        analysis: RepositoryAnalysis,
        requirements: Dict[str, Any]
    ) -> str:
        """Identify the primary use case."""
        description = analysis.metadata.get("description", "")
        if description:
            # Extract first sentence or key phrase
            return description.split(".")[0].strip()
        
        # Fallback to requirement use case
        return requirements.get("use_case", "General purpose library")
    
    def _generate_code_example(
        self,
        analysis: RepositoryAnalysis,
        approach: IntegrationApproach
    ) -> Optional[str]:
        """Generate code example for integration."""
        repo_name = analysis.repository_name.split("/")[-1]
        
        if approach == IntegrationApproach.NPM_PACKAGE:
            return f"""// Install
npm install {repo_name}

// Import and use
import {{ ... }} from '{repo_name}';"""
        
        elif approach == IntegrationApproach.PYTHON_PACKAGE:
            return f"""# Install
pip install {repo_name}

# Import and use
from {repo_name} import ..."""
        
        elif approach == IntegrationApproach.DOCKER_SERVICE:
            return f"""# docker-compose.yml
services:
  {repo_name}:
    image: {repo_name}:latest
    ports:
      - "8080:8080"
"""
        
        return None
    
    def _generate_integration_steps(
        self,
        analysis: RepositoryAnalysis,
        approach: IntegrationApproach
    ) -> List[str]:
        """Generate step-by-step integration instructions."""
        steps = []
        repo_name = analysis.repository_name.split("/")[-1]
        
        if approach == IntegrationApproach.NPM_PACKAGE:
            steps = [
                f"Install package: npm install {repo_name}",
                "Import the module in your code",
                "Configure according to documentation",
                "Test integration"
            ]
        elif approach == IntegrationApproach.PYTHON_PACKAGE:
            steps = [
                f"Install package: pip install {repo_name}",
                "Add to requirements.txt",
                "Import in your Python code",
                "Configure and test"
            ]
        elif approach == IntegrationApproach.DOCKER_SERVICE:
            steps = [
                "Add service to docker-compose.yml",
                "Configure environment variables",
                "Set up networking between services",
                "Test service connectivity"
            ]
        else:
            steps = [
                "Review repository documentation",
                "Clone or download repository",
                "Follow integration guide",
                "Test integration"
            ]
        
        return steps
    
    def _identify_pros_cons(
        self,
        analysis: RepositoryAnalysis,
        approach: IntegrationApproach,
        requirements: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Identify pros and cons of integration."""
        pros = []
        cons = []
        
        # Pros
        stars = analysis.metadata.get("stars", 0)
        if stars > 1000:
            pros.append("Highly popular and well-maintained")
        elif stars > 100:
            pros.append("Active community")
        
        if analysis.metadata.get("license") in ["MIT", "Apache-2.0", "BSD-3-Clause"]:
            pros.append("Permissive license")
        
        if analysis.quality_metrics.get("completeness_score", 0) > 0.7:
            pros.append("Well-documented and structured")
        
        # Cons
        if stars < 50:
            cons.append("Low popularity, may lack community support")
        
        if approach == IntegrationApproach.MICROSERVICE:
            cons.append("Requires infrastructure for service deployment")
        
        if not analysis.metadata.get("license"):
            cons.append("No license specified - verify usage rights")
        
        return pros, cons

