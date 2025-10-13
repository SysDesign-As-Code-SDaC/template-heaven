# Contributing to Organization Universal Template Repository

Thank you for your interest in contributing to our template repository! This document provides guidelines for maintaining, updating, and adding new templates to our organization's private template collection.

## 🎯 Purpose

This repository uses a **multi-branch stack architecture** to organize templates by technology stack. It enables our organization to:

- Maintain consistent project structures across teams
- Bootstrap new projects quickly with best practices
- Keep templates updated with latest upstream improvements
- Avoid exposing internal work through public forks
- Organize templates by technology stack for better discoverability
- Enable parallel development across different technology stacks

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
| **DevOps** | [`stack/devops`](../../tree/stack/devops) | CI/CD, infrastructure, Docker, K8s |
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

### 📖 Architecture Documentation

- **[Branch Strategy](./docs/BRANCH_STRATEGY.md)** - Complete architecture overview
- **[Stack Branch Guide](./docs/STACK_BRANCH_GUIDE.md)** - How to work with stack branches
- **[Trend Detection Integration](./docs/TREND_DETECTION_INTEGRATION.md)** - Automated template discovery
- **[Contributing to Stacks](./docs/CONTRIBUTING_TO_STACKS.md)** - Detailed contribution guidelines

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
- **devops/**: CI/CD, infrastructure, and deployment tools
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

### Before Submitting

1. **Template Validation**
   - [ ] Template can be successfully scaffolded
   - [ ] All dependencies install correctly
   - [ ] Development server starts without errors
   - [ ] Tests pass (if included)
   - [ ] Build process completes successfully
   - [ ] Documentation is complete and accurate

2. **Organization Standards**
   - [ ] Follows our coding standards and conventions
   - [ ] Includes proper TypeScript types and documentation
   - [ ] Has comprehensive docstrings and comments
   - [ ] Includes appropriate CI/CD configurations
   - [ ] Security best practices are followed

### Testing Checklist

- [ ] **Fresh Installation**: Test on clean environment
- [ ] **Dependency Management**: Verify package managers work correctly
- [ ] **Environment Variables**: Document all required env vars
- [ ] **Database Setup**: If applicable, test database initialization
- [ ] **Authentication**: Test auth flows if included
- [ ] **API Integration**: Verify API endpoints work correctly
- [ ] **Deployment**: Test deployment to staging environment

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

- [Template Sync Script Documentation](./scripts/README.md)
- [Organization Coding Standards](./docs/coding-standards.md)
- [Security Guidelines](./docs/security-guidelines.md)
- [Deployment Best Practices](./docs/deployment-guide.md)
