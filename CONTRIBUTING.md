# Contributing to Template Heaven

Thank you for your interest in contributing to Template Heaven! This document provides comprehensive guidelines for maintaining, updating, and extending our professional template management platform.

## 🎯 Mission & Impact

Template Heaven is a **fully operational enterprise-grade template management platform** with **140 tests passing** and extensive API integration capabilities. Our platform enables organizations to:

- **🚀 Professional Template Management**: REST API with FastAPI, comprehensive CLI, and 24 technology stack categories
- **🔍 Advanced Discovery**: GitHub-powered live search, automated trend detection, and quality scoring
- **🏗️ Enterprise Features**: PostgreSQL database, Redis caching, JWT authentication, and Docker deployment
- **🔌 API Integration**: Extensible plugin system for IDEs, CI/CD pipelines, and third-party tools
- **✅ Quality Assurance**: Automated validation, security scanning, and comprehensive testing suite
- **📊 Analytics & Monitoring**: Usage tracking, health monitoring, and performance metrics

## 🌿 Multi-Branch Architecture

This repository uses a hybrid multi-branch architecture:

- **`dev` branch**: Core infrastructure, documentation, scripts, and tools
- **Stack branches**: Dedicated branches for each technology stack (e.g., `stack/frontend`, `stack/backend`)
- **Automated workflows**: Daily trend detection and upstream syncing

### Available Stack Branches

| Stack | Branch | Description |
|-------|--------|-------------|
| **Fullstack** | [`stack/fullstack`](../../tree/stack/fullstack) | Full-stack applications (Next.js, T3 Stack, Remix) |
| **Frontend** | [`stack/frontend`](../../tree/stack/frontend) | Frontend frameworks (React, Vue, Svelte) |
| **Backend** | [`stack/backend`](../../tree/stack/backend) | Backend services (Express, FastAPI, Django) |
| **AI/ML** | [`stack/ai-ml`](../../tree/stack/ai-ml) | Traditional ML and data science |
| **Advanced AI** | [`stack/advanced-ai`](../../tree/stack/advanced-ai) | LLMs, RAG, vector databases |
| **Agentic AI** | [`stack/agentic-ai`](../../tree/stack/agentic-ai) | Autonomous systems and agents |
| **Mobile** | [`stack/mobile`](../../tree/stack/mobile) | Mobile and desktop applications |
| **DevOps** | [`stack/devops`](../../tree/stack/devops) | automation, infrastructure, Docker, K8s |
| **Web3** | [`stack/web3`](../../tree/stack/web3) | Blockchain and smart contracts |
| **Microservices** | [`stack/microservices`](../../tree/stack/microservices) | Microservices architecture |
| **Monorepo** | [`stack/monorepo`](../../tree/stack/monorepo) | Monorepo build systems |
| **Quantum Computing** | [`stack/quantum-computing`](../../tree/stack/quantum-computing) | Quantum frameworks |
| **Computational Biology** | [`stack/computational-biology`](../../tree/stack/computational-biology) | Bioinformatics pipelines |
| **Scientific Computing** | [`stack/scientific-computing`](../../tree/stack/scientific-computing) | HPC, CUDA, molecular dynamics |
| **Space Technologies** | [`stack/space-technologies`](../../tree/stack/space-technologies) | Satellite systems, orbital computing |
| **6G Wireless** | [`stack/6g-wireless`](../../tree/stack/6g-wireless) | Next-gen communication |
| **Structural Batteries** | [`stack/structural-batteries`](../../tree/stack/structural-batteries) | Energy storage integration |
| **Polyfunctional Robots** | [`stack/polyfunctional-robots`](../../tree/stack/polyfunctional-robots) | Multi-task robotic systems |
| **Generative AI** | [`stack/generative-ai`](../../tree/stack/generative-ai) | Content creation and generation |
| **Modern Languages** | [`stack/modern-languages`](../../tree/stack/modern-languages) | Rust, Zig, Mojo, Julia |
| **Serverless** | [`stack/serverless`](../../tree/stack/serverless) | Serverless and edge computing |
| **VSCode Extensions** | [`stack/vscode-extensions`](../../tree/stack/vscode-extensions) | VSCode extension development |
| **Documentation** | [`stack/docs`](../../tree/stack/docs) | Documentation templates |
| **Workflows** | [`stack/workflows`](../../tree/stack/workflows) | General workflows, software engineering best practices |

### 📖 Architecture & API Documentation

- **[Branch Strategy](./docs/BRANCH_STRATEGY.md)** - Complete architecture overview
- **[Stack Branch Guide](./docs/STACK_BRANCH_GUIDE.md)** - How to work with stack branches
- **[Trend Detection Integration](./docs/TREND_DETECTION_INTEGRATION.md)** - Automated template discovery
- **[Contributing to Stacks](./docs/CONTRIBUTING_TO_STACKS.md)** - Detailed contribution guidelines
- **[API Documentation](http://localhost:8000/docs)** - Interactive OpenAPI documentation
- **[API Reference](http://localhost:8000/redoc)** - Alternative API documentation

## 📋 Contribution Guidelines

### Adding New Templates

1. **Identify the Target Stack**
   - Determine which stack branch the template belongs to
   - Check the stack's configuration and requirements
   - Review the stack's trend detection keywords

2. **Research and Validate**
   - Identify the most popular and well-maintained upstream template
   - Verify the template follows current best practices
   - Check for active maintenance and community support
   - Ensure compatibility with our organization's standards

3. **Switch to the Target Stack Branch**
   ```bash
   # Switch to the appropriate stack branch
   git checkout stack/frontend  # or stack/backend, etc.
   
   # Or use the sync script which auto-detects and switches
   ./scripts/sync_template.sh template-name upstream-url
   ```

4. **Template Structure**
   ```
   stacks/[category]/[template-name]/
   ├── README.md              # Template-specific documentation
   ├── [template files]       # All template files
   └── .upstream-info         # Upstream source information
   ```

5. **Required Documentation**
   - **README.md**: Must include:
     - Template description and use cases
     - Prerequisites and dependencies
     - Setup and installation instructions
     - Development workflow
     - Testing instructions
     - Deployment guidelines
     - Upstream source attribution
   - **.upstream-info**: Contains:
     - Original repository URL
     - Last sync date
     - License information
     - Key maintainers

### Updating Existing Templates

1. **Regular Sync Process**
   - Use `scripts/sync_template.sh` for automated syncing
   - Manual sync when automated script isn't available
   - Test updates in a separate branch before merging
   - Update `.upstream-info` with new sync date

2. **Breaking Changes**
   - Document breaking changes in template README
   - Provide migration guide if necessary
   - Consider maintaining multiple versions for major changes

### Template Categories

- **fullstack/**: Complete application stacks (frontend + backend)
- **frontend/**: Frontend-only frameworks and libraries
- **backend/**: Backend services, APIs, and server applications
- **ai-ml/**: Machine learning, data science, and AI frameworks
- **mobile/**: Mobile and desktop application frameworks
- **devops/**: automation, infrastructure, and deployment tools
- **vscode-extensions/**: VSCode extension development templates
- **docs/**: Documentation and community templates
- **other/**: Specialized templates (monorepo, microservices, etc.)

## 🔄 Sync Process

### Automated Sync (Preferred)

```bash
# Sync a specific template
./scripts/sync_template.sh [template-name] [upstream-url]

# Example
./scripts/sync_template.sh t3-stack https://github.com/t3-oss/create-t3-app
```

### Manual Sync

```bash
# Add upstream remote
git remote add upstream [upstream-url]
git fetch upstream

# Checkout specific template files
git checkout upstream/main -- stacks/[category]/[template-name]

# Update upstream info
echo "Last sync: $(date)" >> stacks/[category]/[template-name]/.upstream-info
```

## 🧪 Testing and Validation

### 📊 **Platform Testing Status**
- **✅ Overall**: 140/140 tests passing across all components
- **✅ Unit Tests**: Individual component testing
- **✅ Integration Tests**: API endpoint and service testing
- **✅ CLI Tests**: Command-line interface validation
- **✅ Database Tests**: Data persistence and migration testing
- **✅ GitHub Integration Tests**: External API testing

### Before Submitting Changes

#### **1. Code Quality Requirements**
- [ ] **Type hints** required for all functions and methods
- [ ] **Comprehensive docstrings** following PEP 257
- [ ] **Unit tests** added for new functionality
- [ ] **Integration tests** for API endpoints
- [ ] **Linting passes** (Black, flake8, isort, mypy)
- [ ] **Security scan** passes without critical issues

#### **2. Template Validation**
- [ ] Template scaffolds successfully via CLI
- [ ] Template scaffolds successfully via API
- [ ] All dependencies install correctly (`uv sync` or `pip install`)
- [ ] Development server starts without errors
- [ ] Build process completes successfully
- [ ] Basic functionality works as expected
- [ ] Documentation is complete and accurate

#### **3. API Integration Testing**
- [ ] REST endpoints respond correctly
- [ ] OpenAPI schema validates
- [ ] Authentication works (if required)
- [ ] Rate limiting functions properly
- [ ] Error handling returns appropriate responses
- [ ] CORS headers set correctly

### Comprehensive Testing Checklist

#### **🔧 Development Environment**
- [ ] **Fresh Installation**: Test on clean environment with `uv sync`
- [ ] **Virtual Environment**: Proper isolation and dependency management
- [ ] **Multiple Python Versions**: Test compatibility (3.9+)
- [ ] **Environment Variables**: All required vars documented and tested

#### **🏗️ Build & Deployment**
- [ ] **Package Building**: `uv build` completes successfully
- [ ] **Docker Build**: Container builds without errors
- [ ] **Docker Compose**: Multi-service setup works
- [ ] **Production Deployment**: Staging deployment succeeds

#### **🔌 API & Integration**
- [ ] **Health Checks**: `/api/v1/health` endpoint responds
- [ ] **Template Operations**: CRUD operations work via API
- [ ] **Search Functionality**: Template search returns results
- [ ] **GitHub Integration**: External API calls work (when configured)
- [ ] **Authentication**: JWT tokens and API keys function
- [ ] **Rate Limiting**: Request throttling works correctly

#### **💾 Database & Caching**
- [ ] **Database Connection**: PostgreSQL connection successful
- [ ] **Migrations**: Schema updates apply correctly
- [ ] **Redis Caching**: Cache operations work
- [ ] **Data Persistence**: Template data survives restarts
- [ ] **Backup/Restore**: Database backup and restore works

#### **🔍 CLI & User Experience**
- [ ] **Interactive Wizard**: Guided setup works end-to-end
- [ ] **Command Line**: All CLI commands function properly
- [ ] **Error Messages**: Clear and helpful error reporting
- [ ] **Help System**: `--help` provides useful information
- [ ] **Configuration**: Settings management works correctly

## 🔌 API Integration Guidelines

### REST API Development

#### **Endpoint Design**
- Follow RESTful conventions for resource naming
- Use appropriate HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Implement proper status codes (200, 201, 400, 401, 403, 404, 500)
- Include comprehensive OpenAPI documentation
- Support both JSON and query parameter inputs

#### **Authentication & Security**
- JWT tokens for session-based authentication
- API keys for service-to-service communication
- Rate limiting with configurable thresholds
- CORS configuration for web applications
- Input validation and sanitization

#### **Error Handling**
- Consistent error response format across all endpoints
- Detailed error messages in development mode
- Proper logging of errors with request IDs
- Graceful degradation for external service failures

### Third-Party Integrations

#### **IDE Plugin Development**
```python
# Example IDE plugin integration
from templateheaven.api.client import TemplateHeavenClient

class TemplateHeavenIDEPlugin:
    async def get_available_templates(self, stack: str):
        """Fetch templates for IDE new project wizard"""
        async with TemplateHeavenClient() as client:
            templates = await client.list_templates(stack=stack)
            return [t.dict() for t in templates]

    async def scaffold_project(self, template_id: str, path: str):
        """Scaffold project using Template Heaven API"""
        # Implementation for IDE integration
        pass
```

#### **CI/CD Pipeline Integration**
```yaml
# Example GitHub Actions integration
name: Initialize Project
on: workflow_dispatch

jobs:
  setup:
    runs-on: ubuntu-latest
    steps:
      - name: Initialize from template
        run: |
          curl -X POST "${{ secrets.TH_API_URL }}/api/v1/populate/run" \
            -H "Authorization: Bearer ${{ secrets.TH_API_TOKEN }}" \
            -H "Content-Type: application/json" \
            -d '{"stack": "backend", "limit": 5}'
```

#### **Webhook Integration**
```python
# Example webhook handler for template updates
from fastapi import FastAPI, BackgroundTasks
from templateheaven.services.webhook_service import WebhookService

app = FastAPI()
webhook_service = WebhookService()

@app.post("/webhooks/template-updated")
async def handle_template_update(payload: dict, background_tasks: BackgroundTasks):
    """Handle template update webhooks"""
    background_tasks.add_task(
        webhook_service.process_template_update,
        payload
    )
    return {"status": "accepted"}
```

## 🚀 Development Workflow

### Using uv (Recommended)

#### **Setup Development Environment**
```bash
# Clone and setup
git clone https://github.com/template-heaven/templateheaven.git
cd templateheaven

# Install all dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Verify setup
uv run python -c "import templateheaven; print('✅ Ready!')"
```

#### **Development Commands**
```bash
# Run tests
uv run python -m pytest

# Run API server with hot reload
uv run uvicorn templateheaven.api.main:app --reload

# Format code
uv run black templateheaven tests
uv run isort templateheaven tests

# Type checking
uv run mypy templateheaven

# Security scanning
uv run bandit -r templateheaven
```

### Code Quality Automation

#### **Pre-commit Hooks**
```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
```

#### **Automated Testing**
```bash
# Full test suite
make test

# With coverage
make test-cov

# Specific test categories
uv run python -m pytest tests/test_api/ -v    # API tests
uv run python -m pytest tests/test_cli/ -v    # CLI tests
uv run python -m pytest tests/test_core/ -v   # Core functionality
```

## 📝 Documentation Standards

### Template README Requirements

Each template must include:

1. **Overview**
   - What the template provides
   - Target use cases
   - Technology stack details

2. **Quick Start**
   - Prerequisites
   - Installation steps
   - Basic usage example

3. **Development**
   - Available scripts
   - Development workflow
   - Code organization

4. **Testing**
   - How to run tests
   - Testing strategy
   - Coverage requirements

5. **Deployment**
   - Build process
   - Environment configuration
   - Deployment options

6. **Troubleshooting**
   - Common issues
   - Solutions and workarounds
   - Getting help

### Code Documentation

- **TypeScript**: Comprehensive type definitions and JSDoc comments
- **Python**: Docstrings following PEP 257
- **Go**: Godoc comments for all public functions
- **Rust**: Documentation comments for all public items
- **Configuration**: Clear comments for all config options

## 🔒 Security and Privacy

### Template Security

- Remove or replace any hardcoded secrets
- Use environment variables for sensitive configuration
- Include security best practices in documentation
- Regular security updates and dependency management

### Privacy Considerations

- No internal organization information in templates
- Generic examples and placeholder data only
- Respect upstream licenses and attributions
- Maintain clear separation between templates and internal code

## 🚀 Release Process

### Template Updates

1. **Development Branch**
   - Create feature branch for template updates
   - Test thoroughly in isolated environment
   - Update documentation as needed

2. **Review Process**
   - Peer review of template changes
   - Validation by team leads
   - Security review for sensitive templates

3. **Release**
   - Merge to main branch
   - Tag release with semantic versioning
   - Update organization documentation
   - Notify teams of new/updated templates

## 📞 Getting Help

### Questions and Support

- **Template Issues**: Create issue in this repository
- **General Questions**: Contact DevOps team
- **Security Concerns**: Contact security team directly
- **Upstream Issues**: Report to original template maintainers

### Review Process

All contributions require:

1. **Technical Review**: Code quality and functionality
2. **Documentation Review**: Completeness and accuracy
3. **Security Review**: Security best practices
4. **Team Approval**: Relevant team lead approval

## 🏆 Recognition

Contributors will be recognized in:

- Template README files
- Organization documentation
- Team meetings and announcements
- Annual contribution reviews

---

Thank you for helping maintain our organization's template repository! Your contributions help all teams work more efficiently and consistently.

## 📚 Additional Resources

### API & Integration
- [API Documentation](http://localhost:8000/docs) - Interactive OpenAPI docs
- [API Reference](http://localhost:8000/redoc) - Alternative API documentation
- [Template Sync Scripts](./scripts/README.md) - Automation tools

### Development Guidelines
- [Coding Standards](./docs/coding-standards.md) - Code quality guidelines
- [Security Guidelines](./docs/security-guidelines.md) - Security best practices
- [Deployment Guide](./docs/deployment-guide.md) - Production deployment
- [Branch Strategy](./docs/BRANCH_STRATEGY.md) - Git workflow guidelines

### Community & Support
- [GitHub Issues](../../issues) - Bug reports and feature requests
- [GitHub Discussions](../../discussions) - Community discussions
- [Discord/Slack Community](../../README.md#links) - Real-time support
