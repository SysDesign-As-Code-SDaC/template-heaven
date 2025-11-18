# Planning Guide

This guide explains how to use the planning tools available in Template Heaven for effective software development planning and architecture documentation.

## Overview

Template Heaven provides several planning tools to help you:

- **Document architecture decisions** with Architecture Decision Records (ADRs)
- **Propose major changes** with Request for Comments (RFCs)
- **Plan projects** using planning templates
- **Visualize architecture** with automatically generated C4 model diagrams

## Architecture Decision Records (ADRs)

### What are ADRs?

Architecture Decision Records are documents that capture important architectural decisions made during the project. They help:

- Track why decisions were made
- Understand the context and alternatives considered
- Prevent repeating discussions
- Onboard new team members

### When to Create an ADR

Create an ADR when making decisions about:

- Technology choices (frameworks, libraries, databases)
- Architecture patterns (microservices, monolith, serverless)
- Infrastructure decisions (cloud providers, deployment strategies)
- Design patterns and approaches
- Any decision that affects the system architecture

### How to Create an ADR

#### Using Python Code

```python
from pathlib import Path
from templateheaven.core.adr_manager import ADRManager

# Initialize ADR manager
adr_manager = ADRManager(project_dir=Path("."))

# Create an ADR
adr_path = adr_manager.create_adr(
    title="Use FastAPI for API",
    context="We need to choose a web framework for our API.",
    decision="We will use FastAPI for its async support and automatic documentation.",
    consequences="FastAPI provides excellent performance and developer experience.",
    alternatives=["Flask", "Django REST Framework"],
    status="Accepted"
)
```

#### Manually

1. Copy the template from `docs/templates/ADR_TEMPLATE.md`
2. Create a file in `docs/adr/` with format: `0001-title.md`
3. Fill in the template with your decision details

### ADR Statuses

- **Proposed**: Initial draft, under discussion
- **Accepted**: Decision has been made and approved
- **Deprecated**: Decision is no longer valid
- **Superseded**: Replaced by another ADR

### ADR Structure

Each ADR contains:

- **Status**: Current status of the decision
- **Context**: The issue or problem requiring a decision
- **Decision**: What was decided
- **Consequences**: Positive and negative impacts
- **Alternatives Considered**: Other options evaluated

## Request for Comments (RFCs)

### What are RFCs?

RFCs are documents used to propose and discuss major changes to the system. They provide:

- Structured way to propose changes
- Forum for discussion and feedback
- Documentation of design decisions
- Historical record of major changes

### When to Create an RFC

Create an RFC for:

- Major feature additions
- Significant architecture changes
- Breaking changes
- New system components
- Changes affecting multiple teams

### How to Create an RFC

#### Using Python Code

```python
from pathlib import Path
from templateheaven.core.rfc_manager import RFCManager

# Initialize RFC manager
rfc_manager = RFCManager(project_dir=Path("."))

# Create an RFC
rfc_path = rfc_manager.create_rfc(
    title="Multi-branch Architecture",
    summary="Propose a multi-branch architecture for organizing templates.",
    motivation="Current single-branch approach doesn't scale well.",
    design="Use separate branches for each technology stack.",
    alternatives=["Monorepo", "Separate repositories"],
    open_questions=["How to handle cross-stack dependencies?"],
    status="Draft"
)
```

#### Manually

1. Copy the template from `docs/templates/RFC_TEMPLATE.md`
2. Create a file in `docs/rfc/` with format: `RFC-0001-title.md`
3. Fill in the template with your proposal details

### RFC Statuses

- **Draft**: Initial proposal, work in progress
- **Review**: Under review by stakeholders
- **Accepted**: Proposal approved, ready for implementation
- **Rejected**: Proposal not approved

### RFC Structure

Each RFC contains:

- **Status**: Current status of the RFC
- **Summary**: Brief overview (2-3 sentences)
- **Motivation**: Why this change is needed
- **Detailed Design**: Comprehensive design description
- **Alternatives**: Other approaches considered
- **Open Questions**: Questions that need answers

## Planning Templates

Template Heaven provides several planning document templates:

### Project Charter

Use the Project Charter template (`docs/templates/PROJECT_CHARTER.md`) to:

- Define project objectives and scope
- Identify stakeholders
- Set timelines and milestones
- Document risks and constraints

**When to use**: At the start of a new project or major initiative.

### Technical Design Document

Use the Technical Design Document template (`docs/templates/TECHNICAL_DESIGN.md`) to:

- Document detailed technical designs
- Describe architecture and implementation
- Plan security and performance
- Define testing and deployment strategies

**When to use**: Before implementing a major feature or component.

### Sprint Plan

Use the Sprint Plan template (`docs/templates/SPRINT_PLAN.md`) to:

- Plan sprint backlog
- Track user stories and tasks
- Manage dependencies and blockers
- Conduct sprint reviews and retrospectives

**When to use**: At the start of each sprint in an agile development process.

## Architecture Diagrams

Template Heaven automatically generates C4 model diagrams when you scaffold a project using the architecture questionnaire.

### System Context Diagram

Shows the system and its relationships with users and external systems. Generated automatically in `ARCHITECTURE.md`.

### Container Diagram

Shows high-level technical building blocks and how they interact. Generated automatically in `SYSTEM_DESIGN.md`.

### Component Diagram

Shows components within a container. Generated automatically in `SYSTEM_DESIGN.md`.

### Diagram Files

Diagrams are saved as Mermaid files in `docs/architecture/diagrams/`:

- `system_context.mmd`
- `container.mmd`
- `component.mmd`

These can be viewed in:
- GitHub (renders Mermaid automatically)
- GitLab (renders Mermaid automatically)
- VS Code (with Mermaid extension)
- Online Mermaid editors

## Best Practices

### ADR Best Practices

1. **Create ADRs early**: Document decisions as they're made
2. **Keep them concise**: Focus on the decision and rationale
3. **Update status**: Keep ADR status current
4. **Link related ADRs**: Reference superseded or related ADRs
5. **Review regularly**: Periodically review ADRs for relevance

### RFC Best Practices

1. **Start with motivation**: Clearly explain why the change is needed
2. **Be thorough**: Include enough detail for informed discussion
3. **Consider alternatives**: Document why alternatives were rejected
4. **Address questions**: Answer open questions before implementation
5. **Keep it updated**: Update RFC status as it progresses

### Planning Template Best Practices

1. **Fill completely**: Complete all relevant sections
2. **Keep updated**: Update documents as plans evolve
3. **Version control**: Track changes in git
4. **Review regularly**: Review and update planning documents regularly
5. **Share widely**: Ensure all stakeholders have access

## Integration with Architecture Questionnaire

When you scaffold a project using `templateheaven init`, the architecture questionnaire automatically:

1. Generates architecture documents with embedded diagrams
2. Creates the `docs/architecture/` directory structure
3. Saves diagram files in `docs/architecture/diagrams/`

You can then:

- Create ADRs in `docs/adr/` for architecture decisions
- Create RFCs in `docs/rfc/` for major changes
- Use planning templates for project planning

## Directory Structure

A well-organized project should have:

```
docs/
├── architecture/
│   ├── ARCHITECTURE.md          # Auto-generated
│   ├── SYSTEM_DESIGN.md         # Auto-generated
│   ├── ROADMAP.md               # Auto-generated
│   └── diagrams/                # Auto-generated Mermaid files
│       ├── system_context.mmd
│       ├── container.mmd
│       └── component.mmd
├── adr/                         # Architecture Decision Records
│   ├── 0001-use-fastapi.md
│   └── 0002-postgresql-over-sqlite.md
├── rfc/                         # Request for Comments
│   ├── RFC-0001-multi-branch.md
│   └── RFC-0002-adr-system.md
└── templates/                   # Planning templates
    ├── ADR_TEMPLATE.md
    ├── RFC_TEMPLATE.md
    ├── PROJECT_CHARTER.md
    ├── TECHNICAL_DESIGN.md
    └── SPRINT_PLAN.md
```

## Examples

### Example ADR

See `docs/templates/ADR_TEMPLATE.md` for a complete example.

### Example RFC

See `docs/templates/RFC_TEMPLATE.md` for a complete example.

## Tools and Utilities

### Listing ADRs

```python
from templateheaven.core.adr_manager import ADRManager

adr_manager = ADRManager(Path("."))
adrs = adr_manager.list_adrs()
for adr in adrs:
    print(f"ADR-{adr['number']:04d}: {adr['title']} - {adr['status']}")
```

### Listing RFCs

```python
from templateheaven.core.rfc_manager import RFCManager

rfc_manager = RFCManager(Path("."))
rfcs = rfc_manager.list_rfcs()
for rfc in rfcs:
    print(f"RFC-{rfc['number']:04d}: {rfc['title']} - {rfc['status']}")
```

### Updating Status

```python
# Update ADR status
adr_manager.update_adr_status(1, "Accepted")

# Update RFC status
rfc_manager.update_rfc_status(1, "Review")
```

## Additional Resources

- [MADR (Markdown ADR) Format](https://adr.github.io/madr/)
- [C4 Model Documentation](https://c4model.com/)
- [Mermaid Diagram Syntax](https://mermaid.js.org/)

## Support

For questions or issues with planning tools:

1. Check this guide
2. Review template examples in `docs/templates/`
3. Check existing ADRs/RFCs in your project
4. Consult the architecture questionnaire documentation

---

**Last Updated**: [Date]  
**Version**: 1.0

