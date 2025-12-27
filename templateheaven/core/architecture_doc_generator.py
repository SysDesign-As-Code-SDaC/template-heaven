"""
Architecture Document Generator for Template Heaven.

Generates mandatory architecture and system design documents from
questionnaire answers to prevent architectural drift.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from jinja2 import Template

from .architecture_questionnaire import ArchitectureAnswers
from .diagram_generator import DiagramGenerator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ArchitectureDocGenerator:
    """Generates architecture documentation from questionnaire answers."""
    
    def __init__(self):
        """Initialize the generator."""
        self.logger = logger
        self.diagram_generator = DiagramGenerator()
    
    def generate_all_docs(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path,
        conversation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Generate all architecture documents.
        
        Args:
            project_name: Name of the project
            answers: Architecture questionnaire answers
            output_dir: Directory to write documents
            
        Returns:
            Dictionary mapping document names to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        docs_dir = output_dir / "docs" / "architecture"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create diagrams directory
        diagrams_dir = docs_dir / "diagrams"
        diagrams_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate diagrams
        diagrams = self.diagram_generator.generate_all_diagrams(project_name, answers)
        
        # Save diagram files
        for diagram_name, diagram_content in diagrams.items():
            diagram_path = diagrams_dir / f"{diagram_name}.mmd"
            diagram_path.write_text(diagram_content, encoding="utf-8")
        
        generated_docs = {}
        
        # Generate main architecture document (with diagrams and conversation data)
        arch_doc_path = self._generate_architecture_doc(project_name, answers, docs_dir, diagrams, conversation_data)
        generated_docs["architecture"] = arch_doc_path
        
        # Generate system design document (with diagrams)
        system_doc_path = self._generate_system_design_doc(project_name, answers, docs_dir, diagrams)
        generated_docs["system_design"] = system_doc_path
        
        # Generate roadmap
        roadmap_path = self._generate_roadmap(project_name, answers, docs_dir)
        generated_docs["roadmap"] = roadmap_path
        
        # Generate feature flags document
        if answers.feature_flags_required:
            feature_flags_path = self._generate_feature_flags_doc(project_name, answers, docs_dir)
            generated_docs["feature_flags"] = feature_flags_path
        
        # Generate API design document
        if answers.api_style:
            api_doc_path = self._generate_api_design_doc(project_name, answers, docs_dir)
            generated_docs["api_design"] = api_doc_path
        
        # Generate infrastructure document
        infra_doc_path = self._generate_infrastructure_doc(project_name, answers, docs_dir)
        generated_docs["infrastructure"] = infra_doc_path
        
        # Generate security document
        security_doc_path = self._generate_security_doc(project_name, answers, docs_dir)
        generated_docs["security"] = security_doc_path
        
        logger.info(f"Generated {len(generated_docs)} architecture documents in {docs_dir}")
        
        return generated_docs
    
    def _generate_architecture_doc(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path,
        diagrams: Dict[str, str] = None,
        conversation_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Generate main architecture document."""
        if diagrams is None:
            diagrams = {}
            
        template_content = """# {{ project_name }} - Architecture Document

**Generated:** {{ timestamp }}  
**Version:** 1.0.0

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Context](#system-context)
3. [Architecture Pattern](#architecture-pattern)
4. [System Components](#system-components)
5. [Data Architecture](#data-architecture)
6. [API Design](#api-design)
7. [Infrastructure](#infrastructure)
8. [Security](#security)
9. [Observability](#observability)
10. [Deployment](#deployment)
11. [Risk & Constraints](#risk--constraints)

---

## Project Overview

### Vision
{{ project_vision }}

### Target Users
{{ target_users }}

### Business Objectives
{% for objective in business_objectives %}
- {{ objective }}
{% endfor %}

### Success Metrics
{% for metric in success_metrics %}
- {{ metric }}
{% endfor %}

---

## System Context

### System Context Diagram

The following diagram shows the system and its relationships with users and external systems:

```mermaid
{{ system_context_diagram }}
```

---

## Architecture Pattern

**Selected Pattern:** {{ architecture_pattern|title if architecture_pattern else "Not specified" }}

### Rationale
The {{ architecture_pattern|title if architecture_pattern else "architecture" }} pattern was selected based on:
- Scalability requirements: {{ scalability_requirement|title if scalability_requirement else "Not specified" }}
- Deployment model: {{ deployment_model|title if deployment_model else "Not specified" }}
- Team size: {{ team_size if team_size else "Not specified" }} developers

### Pattern Benefits
{% if architecture_pattern == "microservices" %}
- Independent scaling of services
- Technology diversity
- Fault isolation
- Team autonomy
{% elif architecture_pattern == "serverless" %}
- No server management
- Pay-per-use pricing
- Automatic scaling
- Reduced operational overhead
{% elif architecture_pattern == "event-driven" %}
- Loose coupling
- Scalability
- Resilience
- Real-time processing
{% else %}
- Simplicity
- Easier debugging
- Lower operational complexity
{% endif %}

---

## System Components

### Core Components
{% if architecture_pattern == "microservices" %}
- **API Gateway**: Entry point for all client requests
- **Service Registry**: Service discovery and health checks
- **Microservices**: Domain-specific services
- **Message Broker**: Inter-service communication
- **Database per Service**: Data isolation
{% else %}
- **Application Layer**: Core business logic
- **Data Layer**: Data persistence
- **API Layer**: External interfaces
{% endif %}

### Integration Points
{% if integration_requirements %}
{% for integration in integration_requirements %}
- {{ integration }}
{% endfor %}
{% else %}
- No external integrations specified
{% endif %}

---

## Data Architecture

### Data Volume
**Expected Volume:** {{ data_volume }}

### Data Velocity
**Processing Speed:** {{ data_velocity }}

### Data Variety
**Data Types:** {{ data_variety|title if data_variety else "Not specified" }}

### Database Requirements
{% for db in database_requirements %}
- {{ db }}
{% endfor %}

### Data Retention
**Policy:** {{ data_retention_policy }}

### Backup Strategy
**Strategy:** {{ backup_strategy }}

---

## API Design

### API Style
**Style:** {{ api_style if api_style else "Not specified" }}

### Versioning Strategy
**Strategy:** {{ api_versioning_strategy|title if api_versioning_strategy else "Not specified" }}

### Security Model
**Model:** {{ api_security_model if api_security_model else "Not specified" }}

### Rate Limiting
**Enabled:** {{ "Yes" if api_rate_limiting else "No" }}

---

## Infrastructure

### Cloud Provider
**Provider:** {{ cloud_provider|upper if cloud_provider else "Not specified" }}

### Containerization
**Containerized:** {{ "Yes" if containerization else "No" }}
{% if containerization %}
- **Container Platform:** Docker
{% endif %}

### Orchestration
**Platform:** {{ orchestration_platform|title if orchestration_platform else "Not specified" }}

### Caching Strategy
**Strategy:** {{ caching_strategy|title if caching_strategy else "None" }}

### CDN
**Required:** {{ "Yes" if cdn_required else "No" }}

---

## Security

### Security Requirements
{% for req in security_requirements %}
- {{ req }}
{% endfor %}

### Compliance Requirements
{% if compliance_requirements %}
{% for req in compliance_requirements %}
- {{ req }}
{% endfor %}
{% else %}
- None specified
{% endif %}

---

## Observability

### Logging
**Strategy:** {{ logging_strategy|title if logging_strategy else "Not specified" }}

### Monitoring
**Strategy:** {{ monitoring_strategy|title if monitoring_strategy else "Not specified" }}

### Tracing
**Strategy:** {{ tracing_strategy|title if tracing_strategy else "None" }}

### Alerting
**Strategy:** {{ alerting_strategy|title if alerting_strategy else "Not specified" }}

---

## Deployment

### CI/CD Strategy
**Platform:** {{ ci_cd_strategy|title if ci_cd_strategy else "Not specified" }}

### Testing Strategy
{% for test_type in testing_strategy %}
- {{ test_type|title }}
{% endfor %}

### Deployment Frequency
**Frequency:** {{ deployment_frequency|title if deployment_frequency else "Not specified" }}

---

## Risk & Constraints

### Technical Constraints
{% if technical_constraints %}
{% for constraint in technical_constraints %}
- {{ constraint }}
{% endfor %}
{% else %}
- None specified
{% endif %}

### Business Constraints
{% if business_constraints %}
{% for constraint in business_constraints %}
- {{ constraint }}
{% endfor %}
{% else %}
- None specified
{% endif %}

### Risk Factors
{% if risk_factors %}
{% for risk in risk_factors %}
- {{ risk }}
{% endfor %}
{% else %}
- None identified
{% endif %}

### Mitigation Strategies
{% if mitigation_strategies %}
{% for mitigation in mitigation_strategies %}
- {{ mitigation }}
{% endfor %}
{% else %}
- None specified
{% endif %}

---

## References

{% if reference_architectures %}
{% for ref in reference_architectures %}
- {{ ref }}
{% endfor %}
{% else %}
- None specified
{% endif %}

---

## AI Consultation Summary

{% if conversation_session_id %}
**Consultation Session:** {{ conversation_session_id }}

This architecture was designed with the assistance of an AI system design consultant. The consultation process involved:

- Multi-turn conversation to understand requirements
- Architecture pattern recommendations
- Open-source solution analysis
- Integration strategy suggestions

### Key Insights from Consultation

{% if consultation_insights %}
{{ consultation_insights }}
{% else %}
Consultation insights are available in the conversation session.
{% endif %}

### Recommended Open-Source Integrations

{% if repo_recommendations %}
{% for rec in repo_recommendations %}
#### {{ rec.repository.name }}

- **Repository:** [{{ rec.repository.name }}]({{ rec.repository.url }})
- **Use Case:** {{ rec.use_case }}
- **Integration Approach:** {{ rec.integration_approach|title }}
- **Effort:** {{ rec.effort|title }}
- **Compatibility Score:** {{ (rec.compatibility * 100)|int }}%
- **Relevance Score:** {{ (rec.relevance_score * 100)|int }}%

**Pros:**
{% for pro in rec.pros %}
- {{ pro }}
{% endfor %}

**Cons:**
{% for con in rec.cons %}
- {{ con }}
{% endfor %}

**Integration Steps:**
{% for step in rec.integration_steps %}
1. {{ step }}
{% endfor %}

{% if rec.code_example %}
**Code Example:**
```python
{{ rec.code_example }}
```
{% endif %}

---
{% endfor %}
{% else %}
No specific open-source integrations were recommended during the consultation.
{% endif %}

{% else %}
This architecture was designed without AI consultation assistance.
{% endif %}

---

## Additional Notes

{{ additional_notes if additional_notes else "None" }}

---

**Document Status:** Draft  
**Last Updated:** {{ timestamp }}  
**Next Review:** {{ next_review_date }}
"""
        
        template = Template(template_content)
        
        # Extract conversation data
        conversation_session_id = None
        consultation_insights = None
        repo_recommendations = []
        
        if conversation_data:
            conversation_session_id = conversation_data.get("session_id")
            
            # Extract insights from conversation messages
            messages = conversation_data.get("messages", [])
            if messages:
                # Get last assistant message as insight summary
                assistant_messages = [m for m in messages if m.get("role") == "assistant"]
                if assistant_messages:
                    consultation_insights = assistant_messages[-1].get("content", "")[:500]  # First 500 chars
            
            # Extract repository recommendations
            repo_recommendations = conversation_data.get("repo_recommendations", [])
        
        # Also check answers for conversation reference
        if not conversation_session_id and answers.reference_architectures:
            for ref in answers.reference_architectures:
                if "Conversation:" in ref:
                    conversation_session_id = ref.split("Conversation:")[-1].strip()
                    break
        
        render_vars = answers.to_dict()
        render_vars.update({
            "project_name": project_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "next_review_date": (datetime.now().replace(day=1).replace(month=datetime.now().month + 1)).strftime("%Y-%m-%d"),
            "system_context_diagram": diagrams.get("system_context", ""),
            "container_diagram": diagrams.get("container", ""),
            "conversation_session_id": conversation_session_id,
            "consultation_insights": consultation_insights,
            "repo_recommendations": repo_recommendations,
        })
        content = template.render(**render_vars)
        
        file_path = output_dir / "ARCHITECTURE.md"
        file_path.write_text(content, encoding="utf-8")
        
        return file_path
    
    def _generate_system_design_doc(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path,
        diagrams: Dict[str, str] = None
    ) -> Path:
        """Generate system design document."""
        if diagrams is None:
            diagrams = {}
            
        template_content = """# {{ project_name }} - System Design Document

**Generated:** {{ timestamp }}

## Overview

This document describes the high-level system design for {{ project_name }}.

## System Architecture

### Architecture Pattern
**Pattern:** {{ architecture_pattern|title if architecture_pattern else "Not specified" }}

### Deployment Model
**Model:** {{ deployment_model|title if deployment_model else "Not specified" }}

### Scalability
**Requirement:** {{ scalability_requirement|title if scalability_requirement else "Not specified" }}

## Container Diagram

The following diagram shows the high-level technical building blocks and how they interact:

```mermaid
{{ container_diagram }}
```

## Component Diagram

The following diagram shows the components within the main container:

```mermaid
{{ component_diagram }}
```

## Data Flow

1. **Request Flow:**
   - Client sends request to API Gateway
   - Gateway authenticates and routes to appropriate service
   - Service processes request and queries database
   - Response is cached and returned to client

2. **Data Processing:**
   - Data velocity: {{ data_velocity }}
   - Data volume: {{ data_volume }}
   - Processing: {{ data_variety|title if data_variety else "Not specified" }}

## Performance Requirements

{% if performance_requirements %}
- Response Time: {{ performance_requirements.get('response_time_ms', 'N/A') }}ms
- Throughput: {{ performance_requirements.get('throughput_rps', 'N/A') }} req/s
- Concurrent Users: {{ performance_requirements.get('concurrent_users', 'N/A') }}
{% else %}
Not specified
{% endif %}

## Integration Points

{% if integration_requirements %}
{% for integration in integration_requirements %}
- {{ integration }}
{% endfor %}
{% else %}
- None specified
{% endif %}

## Technology Stack

- **API Style:** {{ api_style if api_style else "Not specified" }}
- **Databases:** {{ database_requirements|join(", ") if database_requirements else "Not specified" }}
- **Caching:** {{ caching_strategy|title if caching_strategy else "None" }}
- **Orchestration:** {{ orchestration_platform|title if orchestration_platform else "None" }}

---

**Last Updated:** {{ timestamp }}
"""
        
        template = Template(template_content)
        render_vars = answers.to_dict()
        render_vars.update({
            "project_name": project_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "container_diagram": diagrams.get("container", ""),
            "component_diagram": diagrams.get("component", ""),
        })
        content = template.render(**render_vars)
        
        file_path = output_dir / "SYSTEM_DESIGN.md"
        file_path.write_text(content, encoding="utf-8")
        
        return file_path
    
    def _generate_roadmap(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path
    ) -> Path:
        """Generate product roadmap."""
        template_content = """# {{ project_name }} - Product Roadmap

**Generated:** {{ timestamp }}  
**Timeline:** {{ timeline if timeline else "Not specified" }}

## Vision

{{ project_vision }}

## Success Metrics

{% for metric in success_metrics %}
- {{ metric }}
{% endfor %}

---

## MVP (Must-Have Features)

**Priority:** P0 - Critical  
**Timeline:** Initial release

{% for feature in must_have_features %}
- [ ] {{ feature }}
{% endfor %}

---

## Phase 1 (Nice-to-Have Features)

**Priority:** P1 - High  
**Timeline:** Post-MVP

{% for feature in nice_to_have_features %}
- [ ] {{ feature }}
{% endfor %}

---

## Future Features

**Priority:** P2 - Medium  
**Timeline:** Future releases

{% for feature in future_features %}
- [ ] {{ feature }}
{% endfor %}

---

## Feature Prioritization

### Must-Have (P0)
These features are essential for the product to function and deliver value.

{% for feature in must_have_features %}
- **{{ feature }}**
  - Priority: Critical
  - Impact: High
  - Effort: TBD
{% endfor %}

### Nice-to-Have (P1)
These features enhance the product but are not critical for initial release.

{% for feature in nice_to_have_features %}
- **{{ feature }}**
  - Priority: High
  - Impact: Medium-High
  - Effort: TBD
{% endfor %}

### Future (P2)
These features are planned for future releases.

{% for feature in future_features %}
- **{{ feature }}**
  - Priority: Medium
  - Impact: TBD
  - Effort: TBD
{% endfor %}

---

## Timeline

**Project Timeline:** {{ timeline if timeline else "Not specified" }}

### Milestones

1. **MVP Release**
   - Target: TBD
   - Features: All must-have features
   - Status: Planning

2. **Phase 1 Release**
   - Target: TBD
   - Features: Nice-to-have features
   - Status: Planned

3. **Future Releases**
   - Target: TBD
   - Features: Future features
   - Status: Backlog

---

## Risk Factors

{% if risk_factors %}
{% for risk in risk_factors %}
- {{ risk }}
{% endfor %}
{% else %}
- None identified
{% endif %}

## Mitigation Strategies

{% if mitigation_strategies %}
{% for mitigation in mitigation_strategies %}
- {{ mitigation }}
{% endfor %}
{% else %}
- None specified
{% endif %}

---

**Last Updated:** {{ timestamp }}  
**Next Review:** Monthly
"""
        
        template = Template(template_content)
        content = template.render(
            project_name=project_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **answers.to_dict()
        )
        
        file_path = output_dir / "ROADMAP.md"
        file_path.write_text(content, encoding="utf-8")
        
        return file_path
    
    def _generate_feature_flags_doc(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path
    ) -> Path:
        """Generate feature flags documentation."""
        template_content = """# {{ project_name }} - Feature Flags Strategy

**Generated:** {{ timestamp }}

## Overview

This document outlines the feature flags strategy for {{ project_name }}.

## Purpose

Feature flags enable:
- Gradual feature rollouts
- A/B testing
- Quick feature toggles
- Risk mitigation
- Canary deployments

## Feature Flags

### MVP Features
{% for feature in must_have_features %}
- **{{ feature }}**
  - Flag: `feature.{{ feature|lower|replace(" ", "_") }}`
  - Default: Enabled
  - Rollout: 100%
{% endfor %}

### Phase 1 Features
{% for feature in nice_to_have_features %}
- **{{ feature }}**
  - Flag: `feature.{{ feature|lower|replace(" ", "_") }}`
  - Default: Disabled
  - Rollout: Gradual (10% → 50% → 100%)
{% endfor %}

### Future Features
{% for feature in future_features %}
- **{{ feature }}**
  - Flag: `feature.{{ feature|lower|replace(" ", "_") }}`
  - Default: Disabled
  - Rollout: TBD
{% endfor %}

## Implementation

### Flag Management
- **Provider:** TBD (e.g., LaunchDarkly, Unleash, custom)
- **Storage:** TBD
- **Update Frequency:** Real-time

### Rollout Strategy

1. **Internal Testing (10%)**
   - Enable for internal team
   - Monitor metrics and errors

2. **Beta Users (25%)**
   - Enable for beta testers
   - Collect feedback

3. **Gradual Rollout (50% → 100%)**
   - Monitor performance metrics
   - Watch for errors and issues
   - Full rollout if stable

### Rollback Strategy

- Immediate disable via feature flag
- No code deployment required
- Monitor impact after rollback

---

**Last Updated:** {{ timestamp }}
"""
        
        template = Template(template_content)
        content = template.render(
            project_name=project_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **answers.to_dict()
        )
        
        file_path = output_dir / "FEATURE_FLAGS.md"
        file_path.write_text(content, encoding="utf-8")
        
        return file_path
    
    def _generate_api_design_doc(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path
    ) -> Path:
        """Generate API design document."""
        template_content = """# {{ project_name }} - API Design Document

**Generated:** {{ timestamp }}

## API Overview

### API Style
**Style:** {{ api_style if api_style else "Not specified" }}

### Versioning
**Strategy:** {{ api_versioning_strategy|title if api_versioning_strategy else "Not specified" }}

### Security
**Model:** {{ api_security_model if api_security_model else "Not specified" }}

### Rate Limiting
**Enabled:** {{ "Yes" if api_rate_limiting else "No" }}

## API Endpoints

### Base URL
```
https://api.{{ project_name|lower|replace(" ", "-") }}.com
```

### Versioning
{% if api_versioning_strategy == "url-path" %}
```
/api/v1/...
/api/v2/...
```
{% elif api_versioning_strategy == "header" %}
```
X-API-Version: v1
```
{% elif api_versioning_strategy == "query-param" %}
```
?version=v1
```
{% endif %}

## Authentication

**Model:** {{ api_security_model if api_security_model else "Not specified" }}

{% if api_security_model == "OAuth2" %}
### OAuth2 Flow
1. Client requests authorization
2. User authorizes
3. Client receives access token
4. Client uses token for API requests
{% elif api_security_model == "JWT" %}
### JWT Authentication
1. Client authenticates and receives JWT
2. Client includes JWT in Authorization header
3. Server validates JWT
{% elif api_security_model == "API-Key" %}
### API Key Authentication
- Include API key in header: `X-API-Key: <key>`
- Or in query parameter: `?api_key=<key>`
{% endif %}

## Rate Limiting

{% if api_rate_limiting %}
**Limits:**
- Per API Key: 1000 requests/hour
- Per IP: 100 requests/minute

**Headers:**
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time
{% else %}
Rate limiting not enabled.
{% endif %}

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {}
  }
}
```

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

---

**Last Updated:** {{ timestamp }}
"""
        
        template = Template(template_content)
        content = template.render(
            project_name=project_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **answers.to_dict()
        )
        
        file_path = output_dir / "API_DESIGN.md"
        file_path.write_text(content, encoding="utf-8")
        
        return file_path
    
    def _generate_infrastructure_doc(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path
    ) -> Path:
        """Generate infrastructure document."""
        template_content = """# {{ project_name }} - Infrastructure Document

**Generated:** {{ timestamp }}

## Infrastructure Overview

### Cloud Provider
**Provider:** {{ cloud_provider|upper if cloud_provider else "Not specified" }}

### Containerization
**Containerized:** {{ "Yes" if containerization else "No" }}
{% if containerization %}
- Platform: Docker
- Base Images: TBD
{% endif %}

### Orchestration
**Platform:** {{ orchestration_platform|title if orchestration_platform else "Not specified" }}

## Infrastructure Components

### Compute
- **Type:** TBD
- **Scaling:** Auto-scaling enabled
- **Regions:** TBD

### Storage
- **Databases:** {{ database_requirements|join(", ") if database_requirements else "Not specified" }}
- **Object Storage:** TBD
- **Backup Storage:** TBD

### Networking
- **CDN:** {{ "Enabled" if cdn_required else "Disabled" }}
- **Load Balancer:** TBD
- **DNS:** TBD

### Caching
**Strategy:** {{ caching_strategy|title if caching_strategy else "None" }}

## Deployment

### CI/CD
**Platform:** {{ ci_cd_strategy|title if ci_cd_strategy else "Not specified" }}

### Deployment Frequency
**Frequency:** {{ deployment_frequency|title if deployment_frequency else "Not specified" }}

## Monitoring & Observability

### Logging
**Strategy:** {{ logging_strategy|title if logging_strategy else "Not specified" }}

### Monitoring
**Platform:** {{ monitoring_strategy|title if monitoring_strategy else "Not specified" }}

### Tracing
**Platform:** {{ tracing_strategy|title if tracing_strategy else "None" }}

### Alerting
**Platform:** {{ alerting_strategy|title if alerting_strategy else "Not specified" }}

---

**Last Updated:** {{ timestamp }}
"""
        
        template = Template(template_content)
        content = template.render(
            project_name=project_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **answers.to_dict()
        )
        
        file_path = output_dir / "INFRASTRUCTURE.md"
        file_path.write_text(content, encoding="utf-8")
        
        return file_path
    
    def _generate_security_doc(
        self,
        project_name: str,
        answers: ArchitectureAnswers,
        output_dir: Path
    ) -> Path:
        """Generate security document."""
        template_content = """# {{ project_name }} - Security Document

**Generated:** {{ timestamp }}

## Security Overview

### Security Requirements
{% for req in security_requirements %}
- {{ req }}
{% endfor %}

### Compliance Requirements
{% if compliance_requirements %}
{% for req in compliance_requirements %}
- {{ req }}
{% endfor %}
{% else %}
- None specified
{% endif %}

## Authentication & Authorization

**Model:** {{ api_security_model if api_security_model else "Not specified" }}

## Data Protection

### Encryption
- **At Rest:** TBD
- **In Transit:** TLS 1.3

### Data Retention
**Policy:** {{ data_retention_policy }}

### Backup Strategy
**Strategy:** {{ backup_strategy }}

## API Security

### Rate Limiting
**Enabled:** {{ "Yes" if api_rate_limiting else "No" }}

### Input Validation
- All inputs validated
- SQL injection prevention
- XSS prevention

## Infrastructure Security

### Network Security
- VPC isolation
- Security groups
- Firewall rules

### Access Control
- IAM roles and policies
- Least privilege principle
- Regular access reviews

## Security Monitoring

### Logging
**Strategy:** {{ logging_strategy|title if logging_strategy else "Not specified" }}

### Alerting
**Platform:** {{ alerting_strategy|title if alerting_strategy else "Not specified" }}

## Incident Response

### Process
1. Detection
2. Containment
3. Eradication
4. Recovery
5. Post-incident review

---

**Last Updated:** {{ timestamp }}
"""
        
        template = Template(template_content)
        content = template.render(
            project_name=project_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **answers.to_dict()
        )
        
        file_path = output_dir / "SECURITY.md"
        file_path.write_text(content, encoding="utf-8")
        
        return file_path

