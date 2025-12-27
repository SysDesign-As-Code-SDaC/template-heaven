"""
Deep code analysis for repositories.

Analyzes repository structure, dependencies, technology stack, and extracts
integration points for open-source repositories.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from enum import Enum

from ...utils.logger import get_logger

logger = get_logger(__name__)


class DependencyType(Enum):
    """Types of dependency files."""
    NPM = "package.json"
    PYTHON = "requirements.txt"
    PYTHON_POETRY = "pyproject.toml"
    RUST = "Cargo.toml"
    GO = "go.mod"
    JAVA_MAVEN = "pom.xml"
    JAVA_GRADLE = "build.gradle"
    DOCKER = "Dockerfile"
    DOCKER_COMPOSE = "docker-compose.yml"


@dataclass
class RepositoryAnalysis:
    """Results of repository analysis."""
    repository_url: str
    repository_name: str
    structure: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    technology_stack: List[str] = field(default_factory=list)
    architecture_patterns: List[str] = field(default_factory=list)
    integration_points: List[Dict[str, Any]] = field(default_factory=list)
    api_contracts: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repository_url": self.repository_url,
            "repository_name": self.repository_name,
            "structure": self.structure,
            "dependencies": self.dependencies,
            "technology_stack": self.technology_stack,
            "architecture_patterns": self.architecture_patterns,
            "integration_points": self.integration_points,
            "api_contracts": self.api_contracts,
            "quality_metrics": self.quality_metrics,
            "metadata": self.metadata
        }


class RepositoryAnalyzer:
    """Analyzes repository code structure and dependencies."""
    
    def __init__(self):
        """Initialize repository analyzer."""
        self.dependency_patterns = {
            DependencyType.NPM: self._parse_package_json,
            DependencyType.PYTHON: self._parse_requirements_txt,
            DependencyType.PYTHON_POETRY: self._parse_pyproject_toml,
            DependencyType.RUST: self._parse_cargo_toml,
            DependencyType.GO: self._parse_go_mod,
            DependencyType.JAVA_MAVEN: self._parse_pom_xml,
            DependencyType.JAVA_GRADLE: self._parse_gradle,
        }
        logger.info("RepositoryAnalyzer initialized")
    
    async def analyze_repository(
        self,
        repo_url: str,
        repo_data: Optional[Dict[str, Any]] = None,
        code_contents: Optional[Dict[str, str]] = None
    ) -> RepositoryAnalysis:
        """
        Analyze a repository for structure, dependencies, and integration points.
        
        Args:
            repo_url: Repository URL
            repo_data: Optional repository metadata from GitHub API
            code_contents: Optional dictionary of file paths to file contents
            
        Returns:
            RepositoryAnalysis object
        """
        logger.info(f"Analyzing repository: {repo_url}")
        
        analysis = RepositoryAnalysis(
            repository_url=repo_url,
            repository_name=self._extract_repo_name(repo_url)
        )
        
        # Store metadata
        if repo_data:
            analysis.metadata = {
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "language": repo_data.get("language"),
                "description": repo_data.get("description"),
                "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else None,
                "topics": repo_data.get("topics", []),
                "created_at": repo_data.get("created_at"),
                "updated_at": repo_data.get("updated_at"),
            }
        
        # Analyze structure
        if code_contents:
            analysis.structure = self._analyze_structure(code_contents)
            analysis.dependencies = self._analyze_dependencies(code_contents)
            analysis.technology_stack = self._detect_technology_stack(code_contents, analysis.dependencies)
            analysis.integration_points = self._find_integration_points(code_contents)
            analysis.api_contracts = self._extract_api_contracts(code_contents)
        
        # Calculate quality metrics
        analysis.quality_metrics = self._calculate_quality_metrics(analysis, repo_data)
        
        logger.info(f"Completed analysis for {repo_url}")
        return analysis
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        # Handle various URL formats
        patterns = [
            r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, repo_url)
            if match:
                return f"{match.group(1)}/{match.group(2)}"
        
        return repo_url.split("/")[-1].replace(".git", "")
    
    def _analyze_structure(self, code_contents: Dict[str, str]) -> Dict[str, Any]:
        """Analyze repository directory structure."""
        structure = {
            "files": list(code_contents.keys()),
            "directories": set(),
            "file_types": {},
            "entry_points": [],
            "config_files": [],
        }
        
        for file_path in code_contents.keys():
            path = Path(file_path)
            
            # Track directories
            for parent in path.parents:
                if str(parent) != ".":
                    structure["directories"].add(str(parent))
            
            # Track file types
            ext = path.suffix.lower()
            structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
            
            # Identify entry points
            if path.name in ["main.py", "index.js", "main.rs", "main.go", "app.py", "server.py"]:
                structure["entry_points"].append(file_path)
            
            # Identify config files
            config_names = [
                "package.json", "requirements.txt", "pyproject.toml", "Cargo.toml",
                "go.mod", "pom.xml", "build.gradle", "Dockerfile", "docker-compose.yml",
                "README.md", ".gitignore", ".env.example"
            ]
            if path.name in config_names:
                structure["config_files"].append(file_path)
        
        structure["directories"] = sorted(list(structure["directories"]))
        return structure
    
    def _analyze_dependencies(self, code_contents: Dict[str, str]) -> Dict[str, List[str]]:
        """Analyze dependencies from various dependency files."""
        dependencies = {}
        
        for file_path, content in code_contents.items():
            path = Path(file_path)
            file_name = path.name.lower()
            
            # Try to match dependency file type
            for dep_type, parser in self.dependency_patterns.items():
                if file_name == dep_type.value.lower():
                    try:
                        deps = parser(content)
                        if deps:
                            dependencies[dep_type.value] = deps
                    except Exception as e:
                        logger.warning(f"Failed to parse {file_path}: {e}")
        
        return dependencies
    
    def _parse_package_json(self, content: str) -> List[str]:
        """Parse package.json for dependencies."""
        try:
            data = json.loads(content)
            deps = []
            
            # Combine all dependency types
            for dep_type in ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]:
                if dep_type in data:
                    deps.extend([f"{name}@{version}" for name, version in data[dep_type].items()])
            
            return deps
        except json.JSONDecodeError:
            return []
    
    def _parse_requirements_txt(self, content: str) -> List[str]:
        """Parse requirements.txt for dependencies."""
        deps = []
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove version specifiers for simplicity
                dep = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
                deps.append(dep)
        return deps
    
    def _parse_pyproject_toml(self, content: str) -> List[str]:
        """Parse pyproject.toml for dependencies."""
        try:
            import tomli
            data = tomli.loads(content)
            deps = []
            
            # Check for poetry or standard dependencies
            if "tool" in data and "poetry" in data["tool"]:
                poetry_deps = data["tool"]["poetry"].get("dependencies", {})
                deps.extend([name for name in poetry_deps.keys() if name != "python"])
            elif "project" in data and "dependencies" in data["project"]:
                deps.extend(data["project"]["dependencies"])
            
            return deps
        except Exception:
            # Fallback to regex parsing if tomli not available
            deps = []
            for line in content.split("\n"):
                match = re.search(r'^(\w+)\s*=', line)
                if match:
                    deps.append(match.group(1))
            return deps
    
    def _parse_cargo_toml(self, content: str) -> List[str]:
        """Parse Cargo.toml for dependencies."""
        deps = []
        in_dependencies = False
        
        for line in content.split("\n"):
            line = line.strip()
            if line == "[dependencies]" or line == "[dev-dependencies]":
                in_dependencies = True
                continue
            elif line.startswith("[") and in_dependencies:
                in_dependencies = False
                continue
            
            if in_dependencies:
                match = re.match(r'^(\w+)\s*=', line)
                if match:
                    deps.append(match.group(1))
        
        return deps
    
    def _parse_go_mod(self, content: str) -> List[str]:
        """Parse go.mod for dependencies."""
        deps = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("require"):
                # Extract module names
                matches = re.findall(r'(\S+)\s+v[\d.]+', line)
                deps.extend(matches)
        return deps
    
    def _parse_pom_xml(self, content: str) -> List[str]:
        """Parse pom.xml for dependencies."""
        deps = []
        # Simple regex-based parsing
        matches = re.findall(r'<artifactId>([^<]+)</artifactId>', content)
        deps.extend(matches)
        return deps
    
    def _parse_gradle(self, content: str) -> List[str]:
        """Parse build.gradle for dependencies."""
        deps = []
        # Simple regex-based parsing
        matches = re.findall(r"implementation\s+['\"]([^'\"]+)['\"]", content)
        deps.extend(matches)
        return deps
    
    def _detect_technology_stack(self, code_contents: Dict[str, str], dependencies: Dict[str, List[str]]) -> List[str]:
        """Detect technology stack from files and dependencies."""
        stack = set()
        
        # Detect from file extensions and names
        for file_path in code_contents.keys():
            path = Path(file_path)
            ext = path.suffix.lower()
            
            if ext == ".py":
                stack.add("Python")
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                stack.add("JavaScript/TypeScript")
            elif ext == ".rs":
                stack.add("Rust")
            elif ext == ".go":
                stack.add("Go")
            elif ext in [".java", ".kt"]:
                stack.add("Java/Kotlin")
            elif ext == ".rs":
                stack.add("Rust")
        
        # Detect frameworks from dependencies
        for deps in dependencies.values():
            for dep in deps:
                dep_lower = dep.lower()
                if "react" in dep_lower:
                    stack.add("React")
                elif "vue" in dep_lower:
                    stack.add("Vue")
                elif "angular" in dep_lower:
                    stack.add("Angular")
                elif "express" in dep_lower:
                    stack.add("Express.js")
                elif "fastapi" in dep_lower or "django" in dep_lower or "flask" in dep_lower:
                    stack.add("Python Web Framework")
                elif "spring" in dep_lower:
                    stack.add("Spring Framework")
                elif "gin" in dep_lower or "echo" in dep_lower:
                    stack.add("Go Web Framework")
        
        return sorted(list(stack))
    
    def _find_integration_points(self, code_contents: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find potential integration points in the code."""
        integration_points = []
        
        # Look for common integration patterns
        for file_path, content in code_contents.items():
            # API endpoints
            api_patterns = [
                (r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', "REST API"),
                (r'router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', "REST API"),
                (r'@route\(["\']([^"\']+)["\']', "Flask Route"),
            ]
            
            for pattern, point_type in api_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    integration_points.append({
                        "type": point_type,
                        "file": file_path,
                        "pattern": match.group(0),
                        "endpoint": match.group(2) if len(match.groups()) > 1 else match.group(1)
                    })
            
            # Export statements (module exports)
            export_patterns = [
                (r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)', "Module Export"),
                (r'module\.exports\s*=\s*(\w+)', "Module Export"),
            ]
            
            for pattern, point_type in export_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    integration_points.append({
                        "type": point_type,
                        "file": file_path,
                        "export_name": match.group(1)
                    })
        
        return integration_points
    
    def _extract_api_contracts(self, code_contents: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract API contracts and interfaces."""
        contracts = []
        
        # Look for interface definitions, type definitions, schemas
        for file_path, content in code_contents.items():
            # TypeScript interfaces
            interface_pattern = r'interface\s+(\w+)\s*\{([^}]+)\}'
            for match in re.finditer(interface_pattern, content, re.DOTALL):
                contracts.append({
                    "type": "TypeScript Interface",
                    "name": match.group(1),
                    "file": file_path,
                    "definition": match.group(2).strip()
                })
            
            # Python dataclasses/Pydantic models
            class_patterns = [
                (r'class\s+(\w+).*?:\s*(?:BaseModel|dataclass)', "Pydantic Model"),
                (r'@dataclass\s+class\s+(\w+)', "Dataclass"),
            ]
            
            for pattern, contract_type in class_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    contracts.append({
                        "type": contract_type,
                        "name": match.group(1),
                        "file": file_path
                    })
        
        return contracts
    
    def _calculate_quality_metrics(
        self,
        analysis: RepositoryAnalysis,
        repo_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for the repository."""
        metrics = {
            "completeness_score": 0.0,
            "documentation_score": 0.0,
            "structure_score": 0.0,
        }
        
        # Check for documentation
        has_readme = any("readme" in f.lower() for f in analysis.structure.get("files", []))
        has_license = analysis.metadata.get("license") is not None
        has_config = len(analysis.structure.get("config_files", [])) > 0
        
        metrics["documentation_score"] = (
            (0.4 if has_readme else 0.0) +
            (0.3 if has_license else 0.0) +
            (0.3 if has_config else 0.0)
        )
        
        # Structure score based on organization
        entry_points = len(analysis.structure.get("entry_points", []))
        directories = len(analysis.structure.get("directories", []))
        metrics["structure_score"] = min(1.0, (entry_points * 0.3 + directories * 0.1))
        
        # Overall completeness
        metrics["completeness_score"] = (
            metrics["documentation_score"] * 0.4 +
            metrics["structure_score"] * 0.3 +
            (0.3 if analysis.dependencies else 0.0)
        )
        
        return metrics

