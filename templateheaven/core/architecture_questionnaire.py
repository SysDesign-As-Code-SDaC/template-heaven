"""
Architecture Questionnaire for Template Heaven.

This module provides comprehensive system design and architecture questions
that are mandatory during project scaffolding to prevent architectural drift.
Based on solution architecture patterns and best practices.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ArchitecturePattern(Enum):
    """Common architecture patterns."""
    MONOLITH = "monolith"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event-driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    CQRS = "cqrs"
    API_FIRST = "api-first"
    SERVICE_MESH = "service-mesh"
    MULTI_TENANT = "multi-tenant"


class DeploymentModel(Enum):
    """Deployment models."""
    SINGLE_REGION = "single-region"
    MULTI_REGION = "multi-region"
    EDGE = "edge"
    HYBRID_CLOUD = "hybrid-cloud"
    MULTI_CLOUD = "multi-cloud"
    ON_PREMISE = "on-premise"
    CLOUD_NATIVE = "cloud-native"  # Alias for cloud-native deployments
    HYBRID = "hybrid"  # Alias for hybrid-cloud


class ScalabilityRequirement(Enum):
    """Scalability requirements."""
    LOW = "low"  # < 1K users
    MEDIUM = "medium"  # 1K-100K users
    HIGH = "high"  # 100K-1M users
    VERY_HIGH = "very-high"  # > 1M users
    AUTO_SCALE = "auto-scale"  # Auto-scaling required


class SecurityLevel(Enum):
    """Security level requirements."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    ENTERPRISE = "enterprise"


class ComplianceStandard(Enum):
    """Compliance standards."""
    NONE = "none"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci-dss"


@dataclass
class ArchitectureQuestion:
    """Represents a single architecture question."""
    id: str
    category: str
    question: str
    question_type: str  # 'text', 'select', 'multiselect', 'number', 'boolean'
    required: bool = True
    options: Optional[List[str]] = None
    help_text: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    default: Optional[Any] = None
    
    @property
    def key(self) -> str:
        """Alias for id to match test expectations."""
        return self.id
    
    @property
    def choices(self) -> Optional[List[str]]:
        """Alias for options to match test expectations."""
        return self.options


@dataclass
class ArchitectureAnswers:
    """Container for architecture questionnaire answers."""
    # Project Overview
    project_vision: str = ""
    target_users: str = ""
    business_objectives: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    
    # Architecture Patterns
    architecture_pattern: Optional[ArchitecturePattern] = None
    architecture_patterns: List[ArchitecturePattern] = field(default_factory=list)
    deployment_model: Optional[DeploymentModel] = None
    scalability_requirement: Optional[ScalabilityRequirement] = None
    
    # Technical Requirements
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    security_requirements: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    integration_requirements: List[str] = field(default_factory=list)
    
    # Infrastructure
    cloud_provider: Optional[str] = None
    containerization: bool = False
    orchestration_platform: Optional[str] = None
    database_requirements: List[str] = field(default_factory=list)
    caching_strategy: Optional[str] = None
    cdn_required: bool = False
    
    # Data Architecture
    data_volume: str = ""
    data_velocity: str = ""
    data_variety: str = ""
    data_retention_policy: str = ""
    backup_strategy: str = ""
    
    # API Design
    api_style: Optional[str] = None  # REST, GraphQL, gRPC, WebSocket
    api_versioning_strategy: Optional[str] = None
    api_security_model: Optional[str] = None
    api_rate_limiting: bool = False
    
    # Observability
    logging_strategy: Optional[str] = None
    monitoring_strategy: Optional[str] = None
    tracing_strategy: Optional[str] = None
    alerting_strategy: Optional[str] = None
    
    # Development & Operations
    ci_cd_strategy: Optional[str] = None
    testing_strategy: List[str] = field(default_factory=list)
    code_review_process: Optional[str] = None
    deployment_frequency: Optional[str] = None
    
    # Features & Roadmap
    must_have_features: List[str] = field(default_factory=list)
    nice_to_have_features: List[str] = field(default_factory=list)
    future_features: List[str] = field(default_factory=list)
    feature_flags_required: bool = False
    
    # Risk & Constraints
    technical_constraints: List[str] = field(default_factory=list)
    business_constraints: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Team & Timeline
    team_size: Optional[int] = None
    timeline: Optional[str] = None
    budget_constraints: Optional[str] = None
    
    # Additional Context
    additional_notes: str = ""
    reference_architectures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert answers to dictionary."""
        return {
            "project_vision": self.project_vision,
            "target_users": self.target_users,
            "business_objectives": self.business_objectives,
            "success_metrics": self.success_metrics,
            "architecture_pattern": self.architecture_pattern.value if self.architecture_pattern else None,
            "architecture_patterns": [p.value for p in self.architecture_patterns] if self.architecture_patterns else [],
            "deployment_model": self.deployment_model.value if self.deployment_model else None,
            "scalability_requirement": self.scalability_requirement.value if self.scalability_requirement else None,
            "performance_requirements": self.performance_requirements,
            "security_requirements": self.security_requirements,
            "compliance_requirements": self.compliance_requirements,
            "integration_requirements": self.integration_requirements,
            "cloud_provider": self.cloud_provider,
            "containerization": self.containerization,
            "orchestration_platform": self.orchestration_platform,
            "database_requirements": self.database_requirements,
            "caching_strategy": self.caching_strategy,
            "cdn_required": self.cdn_required,
            "data_volume": self.data_volume,
            "data_velocity": self.data_velocity,
            "data_variety": self.data_variety,
            "data_retention_policy": self.data_retention_policy,
            "backup_strategy": self.backup_strategy,
            "api_style": self.api_style,
            "api_versioning_strategy": self.api_versioning_strategy,
            "api_security_model": self.api_security_model,
            "api_rate_limiting": self.api_rate_limiting,
            "logging_strategy": self.logging_strategy,
            "monitoring_strategy": self.monitoring_strategy,
            "tracing_strategy": self.tracing_strategy,
            "alerting_strategy": self.alerting_strategy,
            "ci_cd_strategy": self.ci_cd_strategy,
            "testing_strategy": self.testing_strategy,
            "code_review_process": self.code_review_process,
            "deployment_frequency": self.deployment_frequency,
            "must_have_features": self.must_have_features,
            "nice_to_have_features": self.nice_to_have_features,
            "future_features": self.future_features,
            "feature_flags_required": self.feature_flags_required,
            "technical_constraints": self.technical_constraints,
            "business_constraints": self.business_constraints,
            "risk_factors": self.risk_factors,
            "mitigation_strategies": self.mitigation_strategies,
            "team_size": self.team_size,
            "timeline": self.timeline,
            "budget_constraints": self.budget_constraints,
            "additional_notes": self.additional_notes,
            "reference_architectures": self.reference_architectures,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureAnswers':
        """Create ArchitectureAnswers from dictionary."""
        answers = cls()
        
        # Set basic fields
        answers.project_vision = data.get('project_vision', '')
        answers.target_users = data.get('target_users', '')
        answers.business_objectives = data.get('business_objectives', [])
        answers.success_metrics = data.get('success_metrics', [])
        
        # Set architecture patterns
        if 'architecture_pattern' in data and data['architecture_pattern']:
            try:
                answers.architecture_pattern = ArchitecturePattern(data['architecture_pattern'])
            except ValueError:
                pass
        
        if 'architecture_patterns' in data and data['architecture_patterns']:
            answers.architecture_patterns = [
                ArchitecturePattern(p) for p in data['architecture_patterns']
                if isinstance(p, str)
            ]
        
        # Set deployment model
        if 'deployment_model' in data and data['deployment_model']:
            try:
                answers.deployment_model = DeploymentModel(data['deployment_model'])
            except ValueError:
                pass
        
        # Set scalability requirement
        if 'scalability_requirement' in data and data['scalability_requirement']:
            try:
                answers.scalability_requirement = ScalabilityRequirement(data['scalability_requirement'])
            except ValueError:
                pass
        
        # Set other fields
        answers.performance_requirements = data.get('performance_requirements', {})
        answers.security_requirements = data.get('security_requirements', [])
        answers.compliance_requirements = data.get('compliance_requirements', [])
        answers.integration_requirements = data.get('integration_requirements', [])
        answers.cloud_provider = data.get('cloud_provider')
        answers.containerization = data.get('containerization', False)
        answers.orchestration_platform = data.get('orchestration_platform')
        answers.database_requirements = data.get('database_requirements', [])
        answers.caching_strategy = data.get('caching_strategy')
        answers.cdn_required = data.get('cdn_required', False)
        answers.data_volume = data.get('data_volume', '')
        answers.data_velocity = data.get('data_velocity', '')
        answers.data_variety = data.get('data_variety', '')
        answers.data_retention_policy = data.get('data_retention_policy', '')
        answers.backup_strategy = data.get('backup_strategy', '')
        answers.api_style = data.get('api_style')
        answers.api_versioning_strategy = data.get('api_versioning_strategy')
        answers.api_security_model = data.get('api_security_model')
        answers.api_rate_limiting = data.get('api_rate_limiting', False)
        answers.logging_strategy = data.get('logging_strategy')
        answers.monitoring_strategy = data.get('monitoring_strategy')
        answers.tracing_strategy = data.get('tracing_strategy')
        answers.alerting_strategy = data.get('alerting_strategy')
        answers.ci_cd_strategy = data.get('ci_cd_strategy')
        answers.testing_strategy = data.get('testing_strategy', [])
        answers.code_review_process = data.get('code_review_process')
        answers.deployment_frequency = data.get('deployment_frequency')
        answers.must_have_features = data.get('must_have_features', [])
        answers.nice_to_have_features = data.get('nice_to_have_features', [])
        answers.future_features = data.get('future_features', [])
        answers.feature_flags_required = data.get('feature_flags_required', False)
        answers.technical_constraints = data.get('technical_constraints', [])
        answers.business_constraints = data.get('business_constraints', [])
        answers.risk_factors = data.get('risk_factors', [])
        answers.mitigation_strategies = data.get('mitigation_strategies', [])
        answers.team_size = data.get('team_size')
        answers.timeline = data.get('timeline')
        answers.budget_constraints = data.get('budget_constraints')
        answers.additional_notes = data.get('additional_notes', '')
        answers.reference_architectures = data.get('reference_architectures', [])
        
        return answers
    
    def validate(self) -> bool:
        """Validate that required fields are filled."""
        # Basic validation - at minimum, vision and target users should be set
        if not self.project_vision or not self.target_users:
            return False
        
        # Should have at least one architecture pattern
        if not self.architecture_pattern and not self.architecture_patterns:
            return False
        
        return True


class ArchitectureQuestionnaire:
    """Comprehensive architecture questionnaire system."""
    
    def __init__(self):
        """Initialize the questionnaire."""
        self.questions = self._build_questionnaire()
        logger.debug(f"Initialized architecture questionnaire with {len(self.questions)} questions")
    
    def _build_questionnaire(self) -> List[ArchitectureQuestion]:
        """Build the comprehensive questionnaire."""
        questions = []
        
        # 1. Project Overview
        questions.extend([
            ArchitectureQuestion(
                id="project_vision",
                category="project_overview",
                question="What is the vision and purpose of this project? Describe the problem it solves.",
                question_type="text",
                required=True,
                help_text="A clear vision helps guide architectural decisions and prevents scope creep."
            ),
            ArchitectureQuestion(
                id="target_users",
                category="project_overview",
                question="Who are the target users? (e.g., end-users, developers, admins, API consumers)",
                question_type="text",
                required=True,
                help_text="Understanding users helps determine scalability, security, and UX requirements."
            ),
            ArchitectureQuestion(
                id="business_objectives",
                category="project_overview",
                question="What are the key business objectives? (comma-separated)",
                question_type="text",
                required=True,
                help_text="Examples: Increase revenue, reduce costs, improve user experience, enter new market"
            ),
            ArchitectureQuestion(
                id="success_metrics",
                category="project_overview",
                question="How will you measure success? (comma-separated KPIs)",
                question_type="text",
                required=True,
                help_text="Examples: User signups, API response time, uptime, conversion rate"
            ),
        ])
        
        # 2. Architecture Patterns
        questions.extend([
            ArchitectureQuestion(
                id="architecture_pattern",
                category="Architecture Patterns",
                question="Which architecture pattern best fits your project?",
                question_type="select",
                required=True,
                options=[p.value for p in ArchitecturePattern],
                help_text="Choose based on team size, complexity, and scalability needs. Reference: solution-architecture-patterns"
            ),
            ArchitectureQuestion(
                id="deployment_model",
                category="Architecture Patterns",
                question="What deployment model will you use?",
                question_type="select",
                required=True,
                options=[d.value for d in DeploymentModel],
                help_text="Consider data residency, latency, and compliance requirements"
            ),
            ArchitectureQuestion(
                id="scalability_requirement",
                category="Architecture Patterns",
                question="What is the expected scale?",
                question_type="select",
                required=True,
                options=[s.value for s in ScalabilityRequirement],
                help_text="Helps determine infrastructure and architecture choices"
            ),
        ])
        
        # 3. Performance Requirements
        questions.extend([
            ArchitectureQuestion(
                id="performance_requirements",
                category="Performance",
                question="What are the performance requirements? (JSON format: {'response_time_ms': 200, 'throughput_rps': 1000, 'concurrent_users': 10000})",
                question_type="text",
                required=True,
                help_text="Define acceptable response times, throughput, and concurrent user capacity"
            ),
        ])
        
        # 4. Security Requirements
        questions.extend([
            ArchitectureQuestion(
                id="security_requirements",
                category="Security",
                question="What security requirements apply? (comma-separated: authentication, authorization, encryption, audit-logging, etc.)",
                question_type="text",
                required=True,
                help_text="Consider OAuth2, JWT, RBAC, data encryption, security headers, etc."
            ),
            ArchitectureQuestion(
                id="compliance_requirements",
                category="Security",
                question="What compliance requirements apply? (comma-separated: GDPR, HIPAA, SOC2, PCI-DSS, etc.)",
                question_type="text",
                required=False,
                help_text="Compliance affects data handling, storage, and access controls"
            ),
        ])
        
        # 5. Integration Requirements
        questions.extend([
            ArchitectureQuestion(
                id="integration_requirements",
                category="Integration",
                question="What external systems/services need integration? (comma-separated)",
                question_type="text",
                required=False,
                help_text="Examples: Payment gateways, email services, third-party APIs, legacy systems"
            ),
        ])
        
        # 6. Infrastructure
        questions.extend([
            ArchitectureQuestion(
                id="cloud_provider",
                category="Infrastructure",
                question="Which cloud provider? (aws, azure, gcp, multi-cloud, on-premise, other)",
                question_type="select",
                required=False,
                options=["aws", "azure", "gcp", "multi-cloud", "on-premise", "other", "none"],
                help_text="Affects available services and architecture patterns"
            ),
            ArchitectureQuestion(
                id="containerization",
                category="Infrastructure",
                question="Will you use containerization? (Docker)",
                question_type="boolean",
                required=True,
                default=True
            ),
            ArchitectureQuestion(
                id="orchestration_platform",
                category="Infrastructure",
                question="Which orchestration platform? (kubernetes, docker-swarm, nomad, none)",
                question_type="select",
                required=False,
                options=["kubernetes", "docker-swarm", "nomad", "none"],
                help_text="Required for microservices and production deployments"
            ),
            ArchitectureQuestion(
                id="database_requirements",
                category="Infrastructure",
                question="What database requirements? (comma-separated: relational, document, graph, time-series, etc.)",
                question_type="text",
                required=True,
                help_text="Examples: PostgreSQL, MongoDB, Redis, InfluxDB, Neo4j"
            ),
            ArchitectureQuestion(
                id="caching_strategy",
                category="Infrastructure",
                question="What caching strategy? (redis, memcached, cdn, application-cache, none)",
                question_type="select",
                required=False,
                options=["redis", "memcached", "cdn", "application-cache", "none"],
                help_text="Improves performance and reduces database load"
            ),
            ArchitectureQuestion(
                id="cdn_required",
                category="Infrastructure",
                question="Is a CDN required?",
                question_type="boolean",
                required=False,
                default=False,
                help_text="Needed for global content delivery and static assets"
            ),
        ])
        
        # 7. Data Architecture
        questions.extend([
            ArchitectureQuestion(
                id="data_volume",
                category="Data Architecture",
                question="Expected data volume? (e.g., '10GB/day', '1TB/month', 'petabytes')",
                question_type="text",
                required=True,
                help_text="Affects storage and database choices"
            ),
            ArchitectureQuestion(
                id="data_velocity",
                category="Data Architecture",
                question="Data velocity? (e.g., 'real-time', 'batch-hourly', 'streaming')",
                question_type="text",
                required=True,
                help_text="Determines if streaming, batch, or real-time processing is needed"
            ),
            ArchitectureQuestion(
                id="data_variety",
                category="Data Architecture",
                question="Data variety? (structured, semi-structured, unstructured, mixed)",
                question_type="select",
                required=True,
                options=["structured", "semi-structured", "unstructured", "mixed"],
                help_text="Affects database and processing choices"
            ),
            ArchitectureQuestion(
                id="data_retention_policy",
                category="Data Architecture",
                question="Data retention policy? (e.g., '7 years', '90 days', 'indefinite')",
                question_type="text",
                required=True,
                help_text="Affects storage costs and archival strategies"
            ),
            ArchitectureQuestion(
                id="backup_strategy",
                category="Data Architecture",
                question="Backup strategy? (e.g., 'daily-snapshots', 'continuous-replication', 'weekly-full')",
                question_type="text",
                required=True,
                help_text="Critical for disaster recovery and data protection"
            ),
        ])
        
        # 8. API Design
        questions.extend([
            ArchitectureQuestion(
                id="api_style",
                category="API Design",
                question="API style? (REST, GraphQL, gRPC, WebSocket, mixed)",
                question_type="select",
                required=True,
                options=["REST", "GraphQL", "gRPC", "WebSocket", "mixed"],
                help_text="REST for CRUD, GraphQL for flexible queries, gRPC for internal services"
            ),
            ArchitectureQuestion(
                id="api_versioning_strategy",
                category="API Design",
                question="API versioning strategy? (url-path, header, query-param, none)",
                question_type="select",
                required=True,
                options=["url-path", "header", "query-param", "none"],
                help_text="Essential for API evolution and backward compatibility"
            ),
            ArchitectureQuestion(
                id="api_security_model",
                category="API Design",
                question="API security model? (OAuth2, JWT, API-Key, mTLS, none)",
                question_type="select",
                required=True,
                options=["OAuth2", "JWT", "API-Key", "mTLS", "none"],
                help_text="OAuth2 for third-party, JWT for stateless auth, API-Key for simple cases"
            ),
            ArchitectureQuestion(
                id="api_rate_limiting",
                category="API Design",
                question="Is API rate limiting required?",
                question_type="boolean",
                required=True,
                default=True,
                help_text="Prevents abuse and ensures fair usage"
            ),
        ])
        
        # 9. Observability
        questions.extend([
            ArchitectureQuestion(
                id="logging_strategy",
                category="Observability",
                question="Logging strategy? (centralized, distributed, structured, unstructured)",
                question_type="select",
                required=True,
                options=["centralized", "distributed", "structured", "unstructured"],
                help_text="Centralized logging essential for microservices"
            ),
            ArchitectureQuestion(
                id="monitoring_strategy",
                category="Observability",
                question="Monitoring strategy? (prometheus, datadog, cloudwatch, custom, none)",
                question_type="select",
                required=True,
                options=["prometheus", "datadog", "cloudwatch", "custom", "none"],
                help_text="Metrics, dashboards, and alerting are critical for production"
            ),
            ArchitectureQuestion(
                id="tracing_strategy",
                category="Observability",
                question="Distributed tracing? (jaeger, zipkin, datadog-apm, none)",
                question_type="select",
                required=False,
                options=["jaeger", "zipkin", "datadog-apm", "none"],
                help_text="Essential for microservices debugging and performance analysis"
            ),
            ArchitectureQuestion(
                id="alerting_strategy",
                category="Observability",
                question="Alerting strategy? (pagerduty, slack, email, oncall-rotation, none)",
                question_type="select",
                required=True,
                options=["pagerduty", "slack", "email", "oncall-rotation", "none"],
                help_text="Critical for incident response and system reliability"
            ),
        ])
        
        # 10. Development & Operations
        questions.extend([
            ArchitectureQuestion(
                id="ci_cd_strategy",
                category="DevOps",
                question="CI/CD strategy? (github-actions, gitlab-ci, jenkins, circleci, custom)",
                question_type="select",
                required=True,
                options=["github-actions", "gitlab-ci", "jenkins", "circleci", "custom"],
                help_text="Automated testing and deployment are essential"
            ),
            ArchitectureQuestion(
                id="testing_strategy",
                category="DevOps",
                question="Testing strategy? (comma-separated: unit, integration, e2e, performance, security)",
                question_type="text",
                required=True,
                help_text="Comprehensive testing prevents regressions and ensures quality"
            ),
            ArchitectureQuestion(
                id="code_review_process",
                category="DevOps",
                question="Code review process? (mandatory-pr, pair-programming, async-review, none)",
                question_type="select",
                required=True,
                options=["mandatory-pr", "pair-programming", "async-review", "none"],
                help_text="Code reviews improve quality and knowledge sharing"
            ),
            ArchitectureQuestion(
                id="deployment_frequency",
                category="DevOps",
                question="Deployment frequency? (multiple-daily, daily, weekly, monthly, on-demand)",
                question_type="select",
                required=True,
                options=["multiple-daily", "daily", "weekly", "monthly", "on-demand"],
                help_text="Affects infrastructure and process design"
            ),
        ])
        
        # 11. Features & Roadmap
        questions.extend([
            ArchitectureQuestion(
                id="must_have_features",
                category="Features",
                question="Must-have features for MVP? (comma-separated)",
                question_type="text",
                required=True,
                help_text="Core features that define the product"
            ),
            ArchitectureQuestion(
                id="nice_to_have_features",
                category="Features",
                question="Nice-to-have features? (comma-separated)",
                question_type="text",
                required=False,
                help_text="Features that enhance the product but aren't critical"
            ),
            ArchitectureQuestion(
                id="future_features",
                category="Features",
                question="Future features (post-MVP)? (comma-separated)",
                question_type="text",
                required=False,
                help_text="Features planned for future releases"
            ),
            ArchitectureQuestion(
                id="feature_flags_required",
                category="Features",
                question="Are feature flags required?",
                question_type="boolean",
                required=True,
                default=True,
                help_text="Enables gradual rollouts and A/B testing"
            ),
        ])
        
        # 12. Risk & Constraints
        questions.extend([
            ArchitectureQuestion(
                id="technical_constraints",
                category="Constraints",
                question="Technical constraints? (comma-separated: legacy-systems, language-restrictions, etc.)",
                question_type="text",
                required=False,
                help_text="Limitations that affect architecture choices"
            ),
            ArchitectureQuestion(
                id="business_constraints",
                category="Constraints",
                question="Business constraints? (comma-separated: budget, timeline, resources, etc.)",
                question_type="text",
                required=False,
                help_text="Business limitations affecting implementation"
            ),
            ArchitectureQuestion(
                id="risk_factors",
                category="Risk",
                question="Key risk factors? (comma-separated)",
                question_type="text",
                required=False,
                help_text="Identify potential issues early"
            ),
            ArchitectureQuestion(
                id="mitigation_strategies",
                category="Risk",
                question="Mitigation strategies? (comma-separated)",
                question_type="text",
                required=False,
                help_text="How to address identified risks"
            ),
        ])
        
        # 13. Team & Timeline
        questions.extend([
            ArchitectureQuestion(
                id="team_size",
                category="Team",
                question="Team size? (number of developers)",
                question_type="number",
                required=True,
                validation={"min": 1, "max": 1000}
            ),
            ArchitectureQuestion(
                id="timeline",
                category="Timeline",
                question="Project timeline? (e.g., '3 months MVP', '6 months v1.0', '1 year full-featured')",
                question_type="text",
                required=True,
                help_text="Affects architecture complexity and feature prioritization"
            ),
            ArchitectureQuestion(
                id="budget_constraints",
                category="Timeline",
                question="Budget constraints? (e.g., 'low', 'medium', 'high', 'unlimited')",
                question_type="select",
                required=False,
                options=["low", "medium", "high", "unlimited"],
                help_text="Affects infrastructure and tool choices"
            ),
        ])
        
        # 14. Additional Context
        questions.extend([
            ArchitectureQuestion(
                id="reference_architectures",
                category="References",
                question="Reference architectures or patterns? (comma-separated URLs or names)",
                question_type="text",
                required=False,
                help_text="Examples: AWS Well-Architected, solution-architecture-patterns, industry-specific patterns"
            ),
            ArchitectureQuestion(
                id="additional_notes",
                category="Additional",
                question="Any additional notes or context?",
                question_type="text",
                required=False,
                help_text="Any other information relevant to architecture decisions"
            ),
        ])
        
        return questions
    
    def get_all_questions(self) -> List[ArchitectureQuestion]:
        """Get all questions."""
        return self.questions
    
    def get_questions_by_category(self, category: Optional[str] = None) -> List[ArchitectureQuestion]:
        """Get questions by category."""
        if category:
            return [q for q in self.questions if q.category == category]
        else:
            # Return all questions organized by category
            categories = {}
            for question in self.questions:
                if question.category not in categories:
                    categories[question.category] = []
                categories[question.category].append(question)
            return categories
    
    def get_question_by_key(self, key: str) -> Optional[ArchitectureQuestion]:
        """Get a question by its key (id)."""
        for question in self.questions:
            if question.id == key or question.key == key:
                return question
        return None
    
    def get_question_by_id(self, question_id: str) -> Optional[ArchitectureQuestion]:
        """Get a question by its ID."""
        return self.get_question_by_key(question_id)
    
    def validate_answer(self, question: ArchitectureQuestion, answer: Any) -> bool:
        """Validate a single answer for a question."""
        if question.required and (answer is None or answer == ""):
            return False
        
        if answer is None:
            return True
        
        # Validate based on question type
        if question.question_type == "choice" or question.question_type == "select":
            if question.options and answer not in question.options:
                return False
        
        if question.question_type == "number" and question.validation:
            try:
                num_answer = float(answer)
                if "min" in question.validation and num_answer < question.validation["min"]:
                    return False
                if "max" in question.validation and num_answer > question.validation["max"]:
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def fill_with_ai(
        self,
        project_name: str,
        project_description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ArchitectureAnswers:
        """
        Fill questionnaire using AI/LLM (placeholder - uses intelligent defaults).
        
        Args:
            project_name: Name of the project
            project_description: Description of the project
            context: Additional context for AI
            
        Returns:
            ArchitectureAnswers object
        """
        # This is a placeholder - actual LLM integration would go here
        from .architecture_doc_generator import ArchitectureDocGenerator
        
        llm_context = {
            "project_name": project_name,
            "project_description": project_description or "",
            **(context or {})
        }
        
        # Use intelligent defaults for now
        return self._generate_intelligent_defaults(llm_context)
    
    def _generate_intelligent_defaults(self, context: Dict[str, Any]) -> ArchitectureAnswers:
        """Generate intelligent default answers."""
        answers = ArchitectureAnswers()
        
        project_name = context.get("project_name", "")
        project_description = context.get("project_description", "")
        template_stack = context.get("template_stack", "")
        
        # Set basic defaults
        answers.project_vision = project_description or f"Build {project_name}"
        answers.target_users = "End users and API consumers"
        
        # Infer architecture pattern from stack
        if "microservices" in template_stack.lower():
            answers.architecture_pattern = ArchitecturePattern.MICROSERVICES
        elif "serverless" in template_stack.lower():
            answers.architecture_pattern = ArchitecturePattern.SERVERLESS
        elif "event" in template_stack.lower():
            answers.architecture_pattern = ArchitecturePattern.EVENT_DRIVEN
        else:
            answers.architecture_pattern = ArchitecturePattern.MONOLITH
        
        # Default deployment model
        answers.deployment_model = DeploymentModel.CLOUD_NATIVE
        answers.scalability_requirement = ScalabilityRequirement.MEDIUM
        
        return answers
    
    def validate_answers(self, answers: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate questionnaire answers.
        
        Args:
            answers: Dictionary of answers
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for question in self.questions:
            if question.required and question.id not in answers:
                errors.append(f"Required question '{question.id}' ({question.question}) not answered")
                continue
            
            if question.id in answers:
                answer = answers[question.id]
                
                # Validate based on question type
                if question.question_type == "number" and question.validation:
                    if "min" in question.validation and answer < question.validation["min"]:
                        errors.append(f"'{question.id}': value must be >= {question.validation['min']}")
                    if "max" in question.validation and answer > question.validation["max"]:
                        errors.append(f"'{question.id}': value must be <= {question.validation['max']}")
                
                if question.question_type == "select" and question.options:
                    if answer not in question.options:
                        errors.append(f"'{question.id}': invalid option. Must be one of {question.options}")
        
        return len(errors) == 0, errors

