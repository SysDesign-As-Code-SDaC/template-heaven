# 🚀 Stack Branch Guide

This guide explains how to work with stack branches in the template-heaven multi-branch architecture.

## 🎯 Overview

Stack branches are dedicated branches for specific technology stacks, containing only the templates and configuration relevant to that stack. This guide covers how to use, maintain, and contribute to stack branches.

## 📋 Available Stack Branches

### Core Development Stacks
- **`stack/fullstack`** - Full-stack application templates (Next.js, T3 Stack, Remix)
- **`stack/frontend`** - Frontend frameworks (React, Vue, Svelte, Vite)
- **`stack/backend`** - Backend services (Express, FastAPI, Django, Go)
- **`stack/mobile`** - Mobile development (React Native, Flutter, Electron)

### AI/ML Stacks
- **`stack/ai-ml`** - Traditional ML and data science
- **`stack/advanced-ai`** - LLMs, RAG, vector databases
- **`stack/agentic-ai`** - Autonomous systems and agents
- **`stack/generative-ai`** - Content creation and generation

### Infrastructure Stacks
- **`stack/devops`** - automation, infrastructure, Docker, Kubernetes
- **`stack/microservices`** - Microservices architecture
- **`stack/monorepo`** - Monorepo build systems
- **`stack/serverless`** - Serverless and edge computing

### Specialized Stacks
- **`stack/web3`** - Blockchain and smart contracts
- **`stack/quantum-computing`** - Quantum computing frameworks
- **`stack/computational-biology`** - Bioinformatics pipelines
- **`stack/scientific-computing`** - HPC, CUDA, molecular dynamics
- **`stack/space-technologies`** - Satellite systems, orbital computing
- **`stack/6g-wireless`** - Next-gen communication systems
- **`stack/structural-batteries`** - Energy storage integration
- **`stack/polyfunctional-robots`** - Multi-task robotic systems
- **`stack/modern-languages`** - Rust, Zig, Mojo, Julia
- **`stack/vscode-extensions`** - VSCode extension development
- **`stack/docs`** - Documentation templates
- **`stack/workflows`** - General workflows, software engineering best practices, GitHub templates

## 🔧 Working with Stack Branches

### Switching to a Stack Branch

```bash
# List available stack branches
git branch -r | grep "origin/stack/"

# Switch to a specific stack branch
git checkout stack/frontend

# Or create a local tracking branch
git checkout -b stack/frontend origin/stack/frontend
```

### Exploring Stack Contents

```bash
# View stack structure
ls -la stacks/[stack-name]/

# Check stack configuration
cat stacks/[stack-name]/.stack-config.yml

# View available templates
cat stacks/[stack-name]/TEMPLATES.md

# Check trend detection config
cat stacks/[stack-name]/.trend-detection-config.yml
```

### Adding Templates to a Stack

#### Using the Sync Script (Recommended)

```bash
# Sync a template from upstream
./scripts/sync_to_branch.sh template-name upstream-url target-stack

# Example: Add a React template to frontend stack
./scripts/sync_to_branch.sh react-vite https://github.com/vitejs/vite frontend

# With additional options
./scripts/sync_to_branch.sh nextjs-app https://github.com/vercel/next.js fullstack \
  --subdir examples/hello-world \
  --create-pr
```

#### Manual Addition

```bash
# 1. Switch to the target stack branch
git checkout stack/frontend

# 2. Create template directory
mkdir -p stacks/frontend/new-template

# 3. Copy template files
cp -r /path/to/template/* stacks/frontend/new-template/

# 4. Create upstream info file
cat > stacks/frontend/new-template/.upstream-info << EOF
# Upstream Template Information
Upstream URL: https://github.com/example/template
Branch: main
Last Sync: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF

# 5. Update stack documentation
# Edit stacks/frontend/TEMPLATES.md to add the new template

# 6. Commit changes
git add .
git commit -m "feat: add new-template to frontend stack

- Add template from upstream
- Update stack documentation
- Configure upstream tracking"

# 7. Push changes
git push origin stack/frontend
```

### Updating Existing Templates

```bash
# 1. Switch to the stack branch
git checkout stack/frontend

# 2. Update the template
./scripts/sync_to_branch.sh existing-template upstream-url frontend --force

# 3. Review changes
git diff

# 4. Commit and push
git add .
git commit -m "feat: update existing-template in frontend stack"
git push origin stack/frontend
```

### Removing Templates

```bash
# 1. Switch to the stack branch
git checkout stack/frontend

# 2. Remove template directory
rm -rf stacks/frontend/old-template

# 3. Update documentation
# Edit stacks/frontend/TEMPLATES.md to remove the template

# 4. Commit changes
git add .
git commit -m "feat: remove old-template from frontend stack

- Remove deprecated template
- Update stack documentation"

# 5. Push changes
git push origin stack/frontend
```

## 📊 Stack Configuration

### Stack Configuration File

Each stack has a `.stack-config.yml` file:

```yaml
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
created_at: "2024-01-15 10:00:00 UTC"
last_updated: "2024-01-15 10:00:00 UTC"
```

### Trend Detection Configuration

Each stack has a `.trend-detection-config.yml` file:

```yaml
stack_name: "frontend"
enabled: true
keywords:
  - "frontend"
  - "react"
  - "vue"
  - "svelte"
characteristics:
  - has_readme: true
  - has_license: true
  - has_ci: true
thresholds:
  stars:
    minimum: 100
    trending: 1000
    critical: 5000
auto_sync:
  enabled: true
  require_approval: true
```

## 🔄 Branch Management

### Creating a New Stack Branch

```bash
# Use the branch creation script
./scripts/create_stack_branch.sh new-stack \
  --description "Description of the new stack" \
  --maintainers "@team1,@team2"

# Or create manually
git checkout dev
git checkout -b stack/new-stack
# Add stack configuration and templates
git push origin stack/new-stack
```

### Merging Stack Changes to Dev

```bash
# 1. Switch to dev branch
git checkout dev

# 2. Merge stack branch
git merge stack/frontend

# 3. Resolve any conflicts
# 4. Push to dev
git push origin dev
```

### Syncing Core Tools to Stack Branches

```bash
# Sync all core tools to all stack branches
./scripts/sync_core_tools.sh

# Or sync to a specific stack
git checkout stack/frontend
git checkout dev -- scripts/
git checkout dev -- tools/
git add .
git commit -m "feat: sync core tools from dev branch"
git push origin stack/frontend
```

## 🛠️ Development Workflow

### Daily Workflow

1. **Check for Updates**
   ```bash
   git checkout stack/your-stack
   git pull origin stack/your-stack
   ```

2. **Review Trend Detection**
   - Check GitHub issues for trending templates
   - Review automated PRs from trend detection

3. **Update Templates**
   - Sync upstream changes
   - Test template functionality
   - Update documentation

### Weekly Workflow

1. **Stack Maintenance**
   ```bash
   # Update all templates in the stack
   ./scripts/update_stack.sh your-stack
   
   # Review and merge PRs
   # Update stack documentation
   ```

2. **Cross-Stack Coordination**
   - Review shared templates
   - Coordinate with other stack maintainers
   - Update best practices

### Monthly Workflow

1. **Stack Review**
   - Audit template quality
   - Remove deprecated templates
   - Update stack configuration

2. **Performance Optimization**
   - Optimize branch size
  - Review automation pipeline performance
   - Update automation workflows

## 📈 Monitoring and Metrics

### Stack Health Metrics

```bash
# Check stack status
python scripts/branch_manager.py status frontend

# Validate stack configuration
python scripts/branch_manager.py validate

# Generate stack report
python scripts/branch_manager.py report
```

### Template Statistics

- **Template Count**: Number of templates in the stack
- **Last Updated**: When templates were last synced
- **Upstream Compliance**: How up-to-date templates are
- **Test Coverage**: Percentage of templates with tests

### Performance Metrics

- **Branch Size**: Size of the stack branch
- **Clone Time**: Time to clone the stack branch
- **Automation Time**: Time for automated workflows
- **Review Time**: Average PR review time

## 🚨 Troubleshooting

### Common Issues

#### Branch Not Found
```bash
# Check if branch exists remotely
git ls-remote origin | grep stack/your-stack

# Create local tracking branch
git checkout -b stack/your-stack origin/stack/your-stack
```

#### Merge Conflicts
```bash
# Resolve conflicts during merge
git status
# Edit conflicted files
git add .
git commit -m "resolve: merge conflicts"
```

#### Template Sync Issues
```bash
# Check upstream repository
curl -s https://api.github.com/repos/owner/repo

# Verify sync script permissions
chmod +x scripts/sync_to_branch.sh

# Check network connectivity
ping github.com
```

### Getting Help

1. **Check Documentation**: Review this guide and related docs
2. **Search Issues**: Look for similar problems in GitHub issues
3. **Contact Maintainers**: Reach out to stack maintainers
4. **Create Issue**: Report bugs or request features

## 📚 Best Practices

### Template Management
- **Keep Templates Updated**: Regularly sync with upstream
- **Test Before Adding**: Verify templates work correctly
- **Document Changes**: Update README and documentation
- **Follow Standards**: Adhere to organization coding standards

### Branch Management
- **Use Descriptive Commits**: Clear commit messages
- **Create PRs**: Use pull requests for changes
- **Review Changes**: Thoroughly review before merging
- **Keep Branches Clean**: Remove unnecessary files

### Documentation
- **Update READMEs**: Keep template documentation current
- **Maintain Indexes**: Update TEMPLATES.md regularly
- **Document Changes**: Record significant changes
- **Share Knowledge**: Document best practices

## 🔗 Related Resources

- [Branch Strategy](./BRANCH_STRATEGY.md) - Overall architecture
- [Trend Detection Integration](./TREND_DETECTION_INTEGRATION.md) - Automated discovery
- [Contributing to Stacks](./CONTRIBUTING_TO_STACKS.md) - Contribution guidelines
- [Scripts Documentation](../scripts/README.md) - Tool reference

---

**Last Updated**: 2024-01-15  
**Version**: 1.0  
**Maintainer**: Template Team
