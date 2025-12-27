# 🌿 Branch Strategy

This document explains the multi-branch architecture strategy for template-heaven, designed to organize templates by technology stack while maintaining efficient development workflows.

## 🎯 Overview

Template-heaven uses a **hybrid multi-branch architecture** that separates templates by technology stack into dedicated branches while keeping core infrastructure in the main development branch.

## 🏗️ Architecture

### Main/Dev Branch
The `dev` branch contains only core infrastructure and organizational files:

- **Documentation**: README, CONTRIBUTING, and organizational docs
- **Scripts**: Template syncing, branch management, and utility scripts
- **Tools**: Trend detection and analysis tools
- **Automation**: Automation examples (GitHub Actions disabled) and local automation scripts
- **Configuration**: Repository-wide settings and policies

**No actual template files** are stored in the main branch - only references and pointers to stack branches.

### Stack Category Branches
Each major technology category has its own dedicated branch:

- `stack/fullstack` - Full-stack application templates
- `stack/frontend` - Frontend framework templates  
- `stack/backend` - Backend/API service templates
- `stack/ai-ml` - Traditional ML and data science
- `stack/advanced-ai` - LLMs, RAG, vector databases
- `stack/agentic-ai` - Autonomous systems and agents
- `stack/mobile` - Mobile and desktop applications
- `stack/devops` - automation, infrastructure, Docker, K8s
- `stack/web3` - Blockchain and smart contracts
- `stack/microservices` - Microservices architecture
- `stack/monorepo` - Monorepo build systems
- `stack/quantum-computing` - Quantum computing frameworks
- `stack/computational-biology` - Bioinformatics pipelines
- `stack/scientific-computing` - HPC, CUDA, molecular dynamics
- `stack/space-technologies` - Satellite systems, orbital computing
- `stack/6g-wireless` - Next-gen communication systems
- `stack/structural-batteries` - Energy storage integration
- `stack/polyfunctional-robots` - Multi-task robotic systems
- `stack/generative-ai` - Content creation and generation
- `stack/modern-languages` - Rust, Zig, Mojo, Julia
- `stack/serverless` - Serverless and edge computing
- `stack/vscode-extensions` - VSCode extension development
- `stack/docs` - Documentation templates
- `stack/workflows` - General workflows, software engineering best practices, GitHub templates

### Sub-Branches for Complex Stacks
For stacks with many templates or experimental features:

- `stack/fullstack/experimental` - Bleeding-edge fullstack templates
- `stack/ai-ml/research` - Research-focused ML frameworks
- `stack/quantum-computing/hardware-specific` - Hardware-specific implementations

## 🔄 Workflow

### Development Workflow

1. **Core Changes**: Made in `dev` branch
2. **Template Changes**: Made in appropriate stack branch
3. **Cross-Stack Changes**: Coordinated through `dev` branch
4. **Releases**: Merged from `dev` to `main`

### Branch Management

```bash
# Create a new stack branch
git checkout dev
git checkout -b stack/new-stack
# Add templates and configuration
git push origin stack/new-stack

# Work on a specific stack
git checkout stack/frontend
# Make changes to frontend templates
git push origin stack/frontend

# Sync core tools to all stacks
./scripts/sync_core_tools.sh
```

### Template Management

```bash
# Add a template to a specific stack
./scripts/sync_to_branch.sh template-name upstream-url target-stack

# Sync all templates in a stack
./scripts/sync_stack.sh frontend

# Update a template across all relevant stacks
./scripts/update_template.sh template-name
```

## 🛠️ Tool Integration

### Trend Detection
The trend detection system is branch-aware:

- **Core Tool**: Located in `dev` branch under `tools/`
- **Stack Configs**: Each stack has `.trend-detection-config.yml`
- **Auto-Discovery**: Creates PRs to appropriate stack branches
- **Cross-Stack**: Can suggest templates for multiple stacks

### Automated Workflows

1. **Upstream Sync** (Daily)
   - Checks upstream repositories for updates
   - Creates PRs to relevant stack branches
   - Uses trend detection to identify new templates

2. **Trend Detection** (Daily)
   - Monitors GitHub for trending repositories
   - Flags templates for review
   - Creates issues with recommendations

3. **Branch Sync** (Weekly)
   - Syncs core tools to all stack branches
   - Updates documentation across branches
   - Maintains configuration consistency

## 📊 Benefits

### Organization
- **Clear Separation**: Templates grouped by technology stack
- **Focused Development**: Teams work on relevant stacks only
- **Scalable**: Easy to add new stack categories

### Performance
- **Faster Clones**: Smaller branch sizes
- **Parallel Development**: Independent stack development
- **Efficient Automation**: Targeted builds and tests

### Maintenance
- **Cleaner History**: Stack-specific change tracking
- **Easier Reviews**: Focused pull requests
- **Better Ownership**: Clear stack maintainer responsibilities

## 🔧 Configuration

### Stack Configuration
Each stack branch includes:

```yaml
# .stack-config.yml
stack_name: "Frontend Frameworks"
category: "frontend"
description: "Modern frontend framework templates"
maintainers: ["@frontend-team"]
upstream_sources:
  - github.com/vitejs/vite
  - github.com/sveltejs/kit
trend_keywords:
  - "react"
  - "vue"
  - "svelte"
auto_sync: true
```

### Trend Detection Configuration
```yaml
# .trend-detection-config.yml
stack_name: "frontend"
enabled: true
keywords:
  - "frontend"
  - "react"
  - "vue"
thresholds:
  stars:
    trending: 1000
    critical: 5000
auto_sync:
  enabled: true
  require_approval: true
```

## 🚀 Getting Started

### For Developers

1. **Choose Your Stack**: Identify the relevant stack branch
2. **Clone and Switch**: `git checkout stack/your-stack`
3. **Make Changes**: Add or update templates
4. **Create PR**: Submit changes back to `dev`

### For Maintainers

1. **Monitor Trends**: Review daily trend detection reports
2. **Sync Upstream**: Use automated sync workflows
3. **Review PRs**: Approve template additions
4. **Update Docs**: Keep stack documentation current

### For Teams

1. **Stack Ownership**: Assign teams to specific stacks
2. **Regular Reviews**: Weekly stack maintenance
3. **Cross-Stack**: Coordinate shared templates
4. **Best Practices**: Share learnings across stacks

## 📈 Metrics and Monitoring

### Branch Health
- Template count per stack
- Last update timestamps
- Upstream sync status
- Trend detection accuracy

### Performance Metrics
- Clone times by branch
- automation execution times
- PR review times
- Merge frequency

### Quality Metrics
- Template test coverage
- Documentation completeness
- Security scan results
- Upstream compliance

## 🔮 Future Enhancements

### Planned Features
- **Stack Dependencies**: Cross-stack template relationships
- **Automated Testing**: Stack-specific test suites
- **Performance Monitoring**: Branch size optimization
- **Advanced Analytics**: Usage patterns and trends

### Experimental Features
- **AI-Powered Categorization**: Automatic stack assignment
- **Dynamic Branching**: On-demand stack creation
- **Cross-Stack Templates**: Templates spanning multiple stacks
- **Real-time Sync**: Live upstream monitoring

## 📚 Related Documentation

- [Stack Branch Guide](./STACK_BRANCH_GUIDE.md) - How to use stack branches
- [Trend Detection Integration](./TREND_DETECTION_INTEGRATION.md) - Automated template discovery
- [Contributing to Stacks](./CONTRIBUTING_TO_STACKS.md) - Contribution guidelines
- [Branch Management Scripts](../scripts/README.md) - Tool documentation

---

**Last Updated**: 2024-01-15  
**Version**: 1.0  
**Maintainer**: Template Team
