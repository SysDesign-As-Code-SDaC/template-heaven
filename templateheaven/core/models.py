"""
Data models for Template Heaven.

This module defines the core data structures used throughout the application,
including templates, project configurations, and stack categories.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path


class StackCategory(Enum):
    """
    Technology stack categories supported by Template Heaven.
    
    These categories align with the multi-branch architecture of the
    Template Heaven repository, where each category has its own branch.
    """
    
    # Core Development Stacks
    FULLSTACK = "fullstack"
    FRONTEND = "frontend"
    BACKEND = "backend"
    MOBILE = "mobile"
    
    # AI/ML Stacks
    AI_ML = "ai-ml"
    ADVANCED_AI = "advanced-ai"
    AGENTIC_AI = "agentic-ai"
    GENERATIVE_AI = "generative-ai"
    
    # Infrastructure Stacks
    DEVOPS = "devops"
    MICROSERVICES = "microservices"
    MONOREPO = "monorepo"
    SERVERLESS = "serverless"
    
    # Specialized Stacks
    WEB3 = "web3"
    QUANTUM_COMPUTING = "quantum-computing"
    COMPUTATIONAL_BIOLOGY = "computational-biology"
    SCIENTIFIC_COMPUTING = "scientific-computing"
    
    # Emerging Technology Stacks
    SPACE_TECHNOLOGIES = "space-technologies"
    WIRELESS_6G = "6g-wireless"
    STRUCTURAL_BATTERIES = "structural-batteries"
    POLYFUNCTIONAL_ROBOTS = "polyfunctional-robots"
    
    # Gold Standard Templates
    GOLD_STANDARD = "gold-standard"
    
    # Development Tools
    MODERN_LANGUAGES = "modern-languages"
    VSCODE_EXTENSIONS = "vscode-extensions"
    DOCS = "docs"
    WORKFLOWS = "workflows"
    
    @classmethod
    def get_display_name(cls, category: 'StackCategory') -> str:
        """
        Get a human-readable display name for a stack category.
        
        Args:
            category: The stack category enum value
            
        Returns:
            Human-readable display name
        """
        display_names = {
            cls.FULLSTACK: "Fullstack Applications",
            cls.FRONTEND: "Frontend Frameworks",
            cls.BACKEND: "Backend Services",
            cls.MOBILE: "Mobile Development",
            cls.AI_ML: "AI/ML & Data Science",
            cls.ADVANCED_AI: "Advanced AI & LLMs",
            cls.AGENTIC_AI: "Agentic AI Systems",
            cls.GENERATIVE_AI: "Generative AI",
            cls.DEVOPS: "DevOps & Infrastructure",
            cls.MICROSERVICES: "Microservices",
            cls.MONOREPO: "Monorepo Systems",
            cls.SERVERLESS: "Serverless Computing",
            cls.WEB3: "Web3 & Blockchain",
            cls.QUANTUM_COMPUTING: "Quantum Computing",
            cls.COMPUTATIONAL_BIOLOGY: "Computational Biology",
            cls.SCIENTIFIC_COMPUTING: "Scientific Computing",
            cls.SPACE_TECHNOLOGIES: "Space Technologies",
            cls.WIRELESS_6G: "6G Wireless",
            cls.STRUCTURAL_BATTERIES: "Structural Batteries",
            cls.POLYFUNCTIONAL_ROBOTS: "Polyfunctional Robots",
            cls.MODERN_LANGUAGES: "Modern Languages",
            cls.VSCODE_EXTENSIONS: "VSCode Extensions",
            cls.DOCS: "Documentation",
            cls.WORKFLOWS: "Workflows & Best Practices",
        }
        return display_names.get(category, category.value.title())
    
    @classmethod
    def get_description(cls, category: 'StackCategory') -> str:
        """
        Get a description for a stack category.
        
        Args:
            category: The stack category enum value
            
        Returns:
            Description of the stack category
        """
        descriptions = {
            cls.FULLSTACK: "Full-stack applications with frontend and backend",
            cls.FRONTEND: "Frontend frameworks and UI libraries",
            cls.BACKEND: "Backend services, APIs, and server applications",
            cls.MOBILE: "Mobile and desktop applications",
            cls.AI_ML: "Traditional machine learning and data science",
            cls.ADVANCED_AI: "Large language models, RAG, and vector databases",
            cls.AGENTIC_AI: "Autonomous systems and AI agents",
            cls.GENERATIVE_AI: "Content creation and generation systems",
            cls.DEVOPS: "CI/CD, infrastructure, Docker, and Kubernetes",
            cls.MICROSERVICES: "Microservices architecture and patterns",
            cls.MONOREPO: "Monorepo build systems and workspaces",
            cls.SERVERLESS: "Serverless and edge computing platforms",
            cls.WEB3: "Blockchain and smart contract development",
            cls.QUANTUM_COMPUTING: "Quantum computing frameworks and algorithms",
            cls.COMPUTATIONAL_BIOLOGY: "Bioinformatics and genomics pipelines",
            cls.SCIENTIFIC_COMPUTING: "HPC, CUDA, and molecular dynamics",
            cls.SPACE_TECHNOLOGIES: "Satellite systems and orbital computing",
            cls.WIRELESS_6G: "Next-generation communication systems",
            cls.STRUCTURAL_BATTERIES: "Energy storage integration systems",
            cls.POLYFUNCTIONAL_ROBOTS: "Multi-task robotic systems",
            cls.MODERN_LANGUAGES: "Rust, Zig, Mojo, Julia, and modern languages",
            cls.VSCODE_EXTENSIONS: "VSCode extension development",
            cls.DOCS: "Documentation templates and tools",
            cls.WORKFLOWS: "Software engineering best practices and workflows",
        }
        return descriptions.get(category, f"Templates for {category.value}")


@dataclass
class Template:
    """
    Template metadata and configuration.
    
    Represents a project template with all necessary information for
    discovery, selection, and initialization.
    
    Attributes:
        name: Unique template name (e.g., 'react-vite')
        stack: Technology stack category
        description: Human-readable description
        path: Path to template files (relative to package data)
        tags: List of tags for searching and filtering
        dependencies: Dictionary of dependencies and versions
        upstream_url: Original upstream repository URL (optional)
        version: Template version (optional)
        author: Template author (optional)
        license: Template license (optional)
        features: List of template features (optional)
        min_python_version: Minimum Python version required (optional)
        min_node_version: Minimum Node.js version required (optional)
        stars: GitHub stars count (for stack validation)
        forks: GitHub forks count (for stack validation)
        growth_rate: Weekly growth rate (for stack validation)
        technologies: List of technologies used (for stack validation)
    """
    
    name: str
    stack: StackCategory
    description: str
    path: str
    tags: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    upstream_url: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    features: List[str] = field(default_factory=list)
    min_python_version: Optional[str] = None
    min_node_version: Optional[str] = None

    # Stack-specific validation fields
    stars: int = 0
    forks: int = 0
    growth_rate: float = 0.0
    technologies: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate template data after initialization."""
        if not self.name:
            raise ValueError("Template name cannot be empty")
        if not self.description:
            raise ValueError("Template description cannot be empty")
        if not self.path:
            raise ValueError("Template path cannot be empty")
    
    def get_display_name(self) -> str:
        """
        Get a human-readable display name for the template.
        
        Returns:
            Formatted display name
        """
        return self.name.replace('-', ' ').replace('_', ' ').title()
    
    def has_tag(self, tag: str) -> bool:
        """
        Check if template has a specific tag.
        
        Args:
            tag: Tag to search for (case-insensitive)
            
        Returns:
            True if template has the tag
        """
        return tag.lower() in [t.lower() for t in self.tags]
    
    def matches_search(self, query: str) -> bool:
        """
        Check if template matches a search query.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            True if template matches the query
        """
        query_lower = query.lower()
        searchable_text = f"{self.name} {self.description} {' '.join(self.tags)}"
        return query_lower in searchable_text.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert template to dictionary representation.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            "name": self.name,
            "stack": self.stack.value,
            "description": self.description,
            "path": self.path,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "upstream_url": self.upstream_url,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "features": self.features,
            "min_python_version": self.min_python_version,
            "min_node_version": self.min_node_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Template':
        """
        Create template from dictionary representation.
        
        Args:
            data: Dictionary containing template data
            
        Returns:
            Template instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            stack = StackCategory(data["stack"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid stack category: {e}")
        
        return cls(
            name=data["name"],
            stack=stack,
            description=data["description"],
            path=data["path"],
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", {}),
            upstream_url=data.get("upstream_url"),
            version=data.get("version"),
            author=data.get("author"),
            license=data.get("license"),
            features=data.get("features", []),
            min_python_version=data.get("min_python_version"),
            min_node_version=data.get("min_node_version"),
        )


@dataclass
class ProjectConfig:
    """
    Project configuration for template initialization.
    
    Contains all the information needed to initialize a new project
    from a template, including customization options.
    
    Attributes:
        name: Project name
        directory: Target directory path
        template: Template to use
        author: Project author (optional)
        license: Project license (optional)
        package_manager: Package manager to use (npm, yarn, pnpm, pip, poetry)
        description: Project description (optional)
        version: Initial project version (optional)
        features: List of features to enable (optional)
        custom_variables: Custom template variables (optional)
    """
    
    name: str
    directory: str
    template: Template
    author: Optional[str] = None
    license: Optional[str] = None
    package_manager: str = "npm"
    description: Optional[str] = None
    version: Optional[str] = None
    features: List[str] = field(default_factory=list)
    custom_variables: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate project configuration after initialization."""
        if not self.name:
            raise ValueError("Project name cannot be empty")
        if not self.directory:
            raise ValueError("Project directory cannot be empty")
        
        # Validate package manager
        valid_managers = ["npm", "yarn", "pnpm", "pip", "poetry", "cargo", "go"]
        if self.package_manager not in valid_managers:
            raise ValueError(f"Invalid package manager: {self.package_manager}")
        
        # Set default version if not provided
        if not self.version:
            self.version = "0.1.0"
    
    def get_project_path(self) -> Path:
        """
        Get the full project path.
        
        Returns:
            Path object for the project directory
        """
        return Path(self.directory) / self.name
    
    def get_template_variables(self) -> Dict[str, Any]:
        """
        Get all template variables for substitution.
        
        Returns:
            Dictionary of template variables
        """
        variables = {
            "project_name": self.name,
            "project_description": self.description or f"A {self.template.stack.value} project",
            "author": self.author or "Unknown",
            "license": self.license or "MIT",
            "version": self.version or "0.1.0",
            "package_manager": self.package_manager,
            "template_name": self.template.name,
            "template_stack": self.template.stack.value,
        }
        
        # Add custom variables
        variables.update(self.custom_variables)
        
        return variables
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert project configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "name": self.name,
            "directory": self.directory,
            "template": self.template.to_dict(),
            "author": self.author,
            "license": self.license,
            "package_manager": self.package_manager,
            "description": self.description,
            "version": self.version,
            "features": self.features,
            "custom_variables": self.custom_variables,
        }


@dataclass
class TemplateSearchResult:
    """
    Result of a template search operation.
    
    Contains the template and relevance score for search results.
    
    Attributes:
        template: The matching template
        score: Relevance score (0.0 to 1.0)
        match_reason: Reason for the match (optional)
    """
    
    template: Template
    score: float
    match_reason: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate search result after initialization."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")


@dataclass
class TemplateValidationResult:
    """
    Result of template validation.
    
    Contains validation status and any issues found.
    
    Attributes:
        is_valid: Whether the template is valid
        issues: List of validation issues
        warnings: List of validation warnings
    """
    
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str) -> None:
        """
        Add a validation issue.
        
        Args:
            issue: Description of the issue
        """
        self.issues.append(issue)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """
        Add a validation warning.
        
        Args:
            warning: Description of the warning
        """
        self.warnings.append(warning)
