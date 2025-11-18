# Architecture Questionnaire System

## 🎯 Overview

Template Heaven includes a **mandatory architecture questionnaire system** that ensures every scaffolded project has comprehensive system design documentation. This prevents architectural drift and helps teams make informed decisions from the start.

## ✨ Features

- **47 Comprehensive Questions** covering all aspects of system design
- **17 Question Categories** organized by domain
- **Mandatory During Scaffolding** - Cannot skip to prevent drift
- **AI/LLM Integration** - API endpoints for intelligent auto-filling
- **Auto-Generated Documentation** - 7 architecture documents created automatically
- **C4 Model Diagrams** - Automatically generated Mermaid diagrams (System Context, Container, Component)
- **Architecture Decision Records (ADRs)** - Track architectural decisions
- **Request for Comments (RFCs)** - Propose and document major changes
- **Planning Templates** - Project Charter, Technical Design, Sprint Planning templates
- **Validation** - Ensures all required fields are answered

## 📋 Question Categories

1. **Project Overview** - Vision, users, objectives, success metrics
2. **Architecture Patterns** - Monolith, microservices, serverless, event-driven, etc.
3. **Performance** - Response time, throughput, concurrent users
4. **Security** - Security level, compliance standards, authentication
5. **Integration** - API requirements, third-party integrations
6. **Infrastructure** - Cloud provider, containerization, orchestration
7. **Data Architecture** - Volume, velocity, variety, retention
8. **API Design** - Style (REST/GraphQL/gRPC), versioning, security
9. **Observability** - Logging, monitoring, tracing, alerting
10. **DevOps** - CI/CD, testing strategy, deployment frequency
11. **Features** - Must-have vs nice-to-have, feature flags
12. **Risk & Constraints** - Technical/business constraints, risk factors
13. **Team & Timeline** - Team size, timeline, budget

## 🚀 Usage

### Interactive Wizard

When you run `templateheaven init`, the architecture questionnaire is Step 4:

```bash
uv run templateheaven init

# Flow:
# 1. Stack selection
# 2. Template selection
# 3. Project configuration
# 4. Architecture Questionnaire (MANDATORY) ← You are here
# 5. Confirmation
# 6. Project creation with auto-generated docs
```

### API Integration

#### Get Questionnaire Structure

```bash
curl http://localhost:8000/api/v1/architecture/questionnaire/structure
```

#### Get Questions by Category

```bash
curl http://localhost:8000/api/v1/architecture/questionnaire/category/project_overview
```

#### Fill Questionnaire with AI/LLM

```bash
curl -X POST http://localhost:8000/api/v1/architecture/questionnaire/fill \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "my-awesome-project",
    "project_description": "A scalable microservices platform",
    "template_stack": "backend",
    "context": {
      "expected_users": "100K+",
      "compliance": "SOC2"
    },
    "llm_provider": "openai",
    "llm_model": "gpt-4",
    "llm_api_key": "your-api-key"
  }'
```

#### Validate Answers

```bash
curl -X POST http://localhost:8000/api/v1/architecture/questionnaire/validate \
  -H "Content-Type: application/json" \
  -d '{
    "answers": {
      "project_vision": "Build scalable platform",
      "target_users": "Enterprise customers",
      "architecture_patterns": ["microservices"],
      "deployment_model": "cloud-native"
    }
  }'
```

## 📄 Generated Documents

When a project is scaffolded, the following architecture documents are automatically generated in `docs/architecture/`:

### 1. ARCHITECTURE.md
Complete architecture overview including:
- Project vision and objectives
- Architecture patterns selected
- System components
- Data architecture
- API design overview
- Infrastructure summary
- Security overview
- Observability strategy
- Deployment model
- Risk and constraints

### 2. SYSTEM_DESIGN.md
Detailed system design document with:
- System architecture diagrams (text-based)
- Component interactions
- Data flow
- Service boundaries
- Communication patterns
- Scalability considerations

### 3. ROADMAP.md
Feature roadmap and prioritization:
- Must-have features (Phase 1)
- Nice-to-have features (Phase 2)
- Future features (Phase 3+)
- Timeline estimates
- Dependencies
- Success metrics

### 4. FEATURE_FLAGS.md
Feature flagging strategy (if enabled):
- Feature flag framework recommendations
- Flag naming conventions
- Rollout strategies
- A/B testing approach
- Flag lifecycle management

### 5. INFRASTRUCTURE.md
Infrastructure requirements:
- Cloud provider recommendations
- Containerization strategy
- Orchestration platform
- Database requirements
- Caching strategy
- CDN requirements
- Backup and disaster recovery

### 6. SECURITY.md
Security architecture:
- Security level requirements
- Compliance standards
- Authentication and authorization
- Data encryption
- Network security
- Incident response plan

### 7. API_DESIGN.md
API design documentation (if API style specified):
- API style (REST/GraphQL/gRPC)
- Versioning strategy
- Security model
- Rate limiting
- Error handling
- Documentation standards

### 8. Architecture Diagrams
C4 model diagrams automatically generated in `docs/architecture/diagrams/`:
- **system_context.mmd** - System and its relationships with users and external systems
- **container.mmd** - High-level technical building blocks and interactions
- **component.mmd** - Components within containers

Diagrams are embedded in `ARCHITECTURE.md` and `SYSTEM_DESIGN.md` as Mermaid code blocks, which render automatically on GitHub and GitLab.

## 🔧 Implementation Details

### Architecture Questionnaire Module

Located in `templateheaven/core/architecture_questionnaire.py`:

- `ArchitectureQuestionnaire` - Main questionnaire class
- `ArchitectureAnswers` - Data container for answers
- `ArchitectureQuestion` - Individual question model
- Enums: `ArchitecturePattern`, `DeploymentModel`, `ScalabilityRequirement`, `SecurityLevel`, `ComplianceStandard`

### Document Generator

Located in `templateheaven/core/architecture_doc_generator.py`:

- `ArchitectureDocGenerator` - Generates all architecture documents
- Uses Jinja2 templates for document generation
- Creates documents in `docs/architecture/` directory

### Integration Points

1. **Wizard Integration** (`templateheaven/cli/wizard.py`):
   - Step 4: `_collect_architecture_answers()`
   - Collects answers interactively
   - Option for AI/LLM auto-filling

2. **Customizer Integration** (`templateheaven/core/customizer.py`):
   - Auto-generates architecture docs after project creation
   - Uses `ArchitectureDocGenerator.generate_all_docs()`

3. **API Integration** (`templateheaven/api/routes/architecture.py`):
   - REST endpoints for questionnaire operations
   - AI/LLM integration endpoints
   - Validation endpoints

## 🧪 Testing

Comprehensive test suite in `tests/`:

- `test_architecture_questionnaire.py` - 17 tests for questionnaire functionality
- `test_architecture_doc_generator.py` - 11 tests for document generation
- `test_architecture_api.py` - API endpoint tests

All tests passing ✅

## 📚 Best Practices

1. **Answer Honestly** - The questionnaire helps prevent architectural drift
2. **Use AI/LLM When Appropriate** - Can speed up filling for standard projects
3. **Review Generated Docs** - Always review and customize generated documents
4. **Update Regularly** - Keep architecture docs updated as project evolves
5. **Share with Team** - Architecture docs help align team understanding

## 🔗 Related Documentation

- [Wizard Explanation](../WIZARD_EXPLANATION.md) - How the wizard works
- [CLI Usage Guide](../CLI_USAGE_GUIDE.md) - CLI commands and usage
- [Planning Guide](./PLANNING_GUIDE.md) - Guide to using ADRs, RFCs, and planning templates
- [API Documentation](http://localhost:8000/docs) - Full API reference

## 🎯 Benefits

1. **Prevents Architectural Drift** - Mandatory questions ensure decisions are made upfront
2. **Comprehensive Documentation** - Auto-generated docs save time
3. **Visual Architecture** - C4 model diagrams provide clear visual representation
4. **Decision Tracking** - ADRs document why decisions were made
5. **Change Management** - RFCs provide structured process for major changes
6. **Team Alignment** - Shared understanding of architecture decisions
7. **Onboarding** - New team members can quickly understand the system
8. **Compliance** - Documents help meet compliance requirements
9. **Roadmap Planning** - Prioritization helps plan feature development

## 📊 Architecture Diagrams

The architecture questionnaire automatically generates C4 model diagrams using Mermaid syntax:

- **System Context Diagram**: Shows the system and its relationships with users and external systems
- **Container Diagram**: Shows high-level technical building blocks (applications, databases, services)
- **Component Diagram**: Shows components within containers

Diagrams are:
- Saved as `.mmd` files in `docs/architecture/diagrams/`
- Embedded in `ARCHITECTURE.md` and `SYSTEM_DESIGN.md`
- Rendered automatically on GitHub, GitLab, and other platforms that support Mermaid

## 📝 Planning Tools

Template Heaven provides additional planning tools:

### Architecture Decision Records (ADRs)

Document important architectural decisions in `docs/adr/`. Use the `ADRManager` class or copy templates from `docs/templates/ADR_TEMPLATE.md`.

### Request for Comments (RFCs)

Propose major changes using RFCs in `docs/rfc/`. Use the `RFCManager` class or copy templates from `docs/templates/RFC_TEMPLATE.md`.

### Planning Templates

Use planning templates for:
- **Project Charter** (`docs/templates/PROJECT_CHARTER.md`) - Define project objectives and scope
- **Technical Design** (`docs/templates/TECHNICAL_DESIGN.md`) - Document detailed technical designs
- **Sprint Plan** (`docs/templates/SPRINT_PLAN.md`) - Plan sprint backlogs and track progress

See [Planning Guide](./PLANNING_GUIDE.md) for detailed usage instructions.

