# 🤝 Contributing to Stacks

This guide explains how to contribute to stack branches in the template-heaven multi-branch architecture.

## 🎯 Overview

Contributing to stack branches involves adding new templates, updating existing ones, improving documentation, and maintaining stack quality. This guide covers the complete contribution workflow.

## 🚀 Getting Started

### Prerequisites

- Git installed and configured
- GitHub account with repository access
- Basic understanding of the technology stack you're contributing to
- Familiarity with the template-heaven architecture

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/template-heaven.git
   cd template-heaven
   ```

2. **Switch to Target Stack Branch**
   ```bash
   # List available stack branches
   git branch -r | grep "origin/stack/"
   
   # Switch to your target stack
   git checkout stack/frontend
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/add-new-template
   ```

## 📋 Contribution Types

### Adding New Templates

#### 1. Research and Validation

Before adding a template, ensure it meets our standards:

- **Popularity**: Minimum 100 stars, preferably 1000+
- **Maintenance**: Active development and recent commits
- **Documentation**: Comprehensive README and setup instructions
- **Quality**: Follows best practices and coding standards
- **License**: Compatible with organization requirements
- **Security**: No known security vulnerabilities

#### 2. Template Addition Process

```bash
# Use the automated sync script (recommended)
./scripts/sync_to_branch.sh template-name upstream-url target-stack

# Example: Add a React template to frontend stack
./scripts/sync_to_branch.sh react-vite https://github.com/vitejs/vite frontend

# With additional options
./scripts/sync_to_branch.sh nextjs-app https://github.com/vercel/next.js fullstack \
  --subdir examples/hello-world \
  --create-pr
```

#### 3. Manual Template Addition

If the automated script doesn't work:

```bash
# 1. Create template directory
mkdir -p stacks/frontend/new-template

# 2. Copy template files
cp -r /path/to/template/* stacks/frontend/new-template/

# 3. Create upstream info file
cat > stacks/frontend/new-template/.upstream-info << EOF
# Upstream Template Information
Upstream URL: https://github.com/example/template
Branch: main
Subdirectory: /
Last Sync: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Sync Script: scripts/sync_to_branch.sh
Sync Command: ./scripts/sync_to_branch.sh new-template https://github.com/example/template frontend

# License Information
# Please check the upstream repository for license details
# and ensure compliance with the original license terms.

# Attribution
# This template is based on work from the upstream repository.
# Please maintain proper attribution when using this template.
EOF

# 4. Update stack documentation
# Edit stacks/frontend/TEMPLATES.md to add the new template

# 5. Test the template
cd stacks/frontend/new-template
npm install  # or appropriate package manager
npm run dev  # or appropriate start command

# 6. Commit changes
git add .
git commit -m "feat: add new-template to frontend stack

- Add template from upstream: https://github.com/example/template
- Update stack documentation
- Configure upstream tracking

Template: new-template
Upstream: https://github.com/example/template
Stack: frontend"
```

### Updating Existing Templates

#### 1. Check for Updates

```bash
# Check upstream repository for updates
cd stacks/frontend/existing-template
git remote -v  # Check upstream remote
git fetch upstream
git log HEAD..upstream/main --oneline  # See new commits
```

#### 2. Update Template

```bash
# Use the sync script with --force flag
./scripts/sync_to_branch.sh existing-template upstream-url frontend --force

# Or update manually
cd stacks/frontend/existing-template
git pull upstream main
```

#### 3. Test Updates

```bash
# Test the updated template
npm install
npm run test
npm run build
```

#### 4. Commit Changes

```bash
git add .
git commit -m "feat: update existing-template in frontend stack

- Sync with upstream changes
- Update dependencies
- Fix compatibility issues

Upstream: https://github.com/example/template
Last commit: abc1234"
```

### Improving Documentation

#### 1. Template README Updates

```bash
# Edit template README
vim stacks/frontend/template-name/README.md

# Ensure it includes:
# - Clear description and use cases
# - Prerequisites and installation
# - Usage examples
# - API documentation (if applicable)
# - Contributing guidelines
# - License information
```

#### 2. Stack Documentation Updates

```bash
# Update stack README
vim stacks/frontend/README.md

# Update template index
vim stacks/frontend/TEMPLATES.md

# Update stack configuration
vim stacks/frontend/.stack-config.yml
```

#### 3. Commit Documentation Changes

```bash
git add .
git commit -m "docs: improve frontend stack documentation

- Update template READMEs
- Add usage examples
- Fix typos and formatting
- Update template index"
```

### Removing Deprecated Templates

#### 1. Identify Deprecated Templates

Templates should be removed if they:
- Are no longer maintained upstream
- Have security vulnerabilities
- Are superseded by better alternatives
- Have compatibility issues

#### 2. Remove Template

```bash
# Remove template directory
rm -rf stacks/frontend/deprecated-template

# Update documentation
# Edit stacks/frontend/TEMPLATES.md to remove the template

# Commit changes
git add .
git commit -m "feat: remove deprecated-template from frontend stack

- Remove unmaintained template
- Update stack documentation
- Add deprecation notice to changelog

Reason: No longer maintained upstream
Alternative: new-better-template"
```

## 🔍 Quality Assurance

### Template Testing Checklist

Before submitting a template, ensure it passes these tests:

- [ ] **Installation**: Dependencies install without errors
- [ ] **Build**: Project builds successfully
- [ ] **Tests**: All tests pass (if included)
- [ ] **Linting**: Code passes linting checks
- [ ] **Documentation**: README is comprehensive and accurate
- [ ] **Security**: No known security vulnerabilities
- [ ] **Performance**: Build and startup times are reasonable
- [ ] **Compatibility**: Works with specified Node.js/Python/etc. versions

### Code Quality Standards

#### TypeScript/JavaScript
- Use TypeScript where applicable
- Follow ESLint configuration
- Use Prettier for formatting
- Include proper type definitions
- Write unit tests

#### Python
- Follow PEP 8 style guide
- Use type hints
- Include docstrings
- Write unit tests with pytest
- Use black for formatting

#### Documentation
- Use clear, concise language
- Include code examples
- Provide setup instructions
- Document all configuration options
- Include troubleshooting section

### Security Considerations

- **Dependencies**: Use only trusted, well-maintained packages
- **Secrets**: Never commit API keys or passwords
- **Vulnerabilities**: Regularly update dependencies
- **Permissions**: Follow principle of least privilege
- **Validation**: Validate all user inputs

## 📝 Pull Request Process

### Creating a Pull Request

1. **Push Your Changes**
   ```bash
   git push origin feature/add-new-template
   ```

2. **Create Pull Request**
   - Go to GitHub repository
   - Click "New Pull Request"
   - Select your feature branch
   - Choose target stack branch
   - Fill out PR template

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] New template
   - [ ] Template update
   - [ ] Documentation improvement
   - [ ] Bug fix
   - [ ] Other

   ## Testing
   - [ ] Template installs successfully
   - [ ] Template builds without errors
   - [ ] All tests pass
   - [ ] Documentation is updated

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)

   ## Screenshots (if applicable)
   Add screenshots to help explain your changes

   ## Additional Notes
   Any additional information for reviewers
   ```

### Review Process

1. **Automated Checks**
   - automation pipeline runs
   - Code quality checks
   - Security scans
   - Dependency audits

2. **Human Review**
   - Stack maintainer reviews
   - Code quality assessment
   - Documentation review
   - Testing verification

3. **Approval Process**
   - At least one approval required
   - All checks must pass
   - No unresolved conversations

### Merging

1. **Squash and Merge** (recommended)
   - Combines all commits into one
   - Clean commit history
   - Descriptive commit message

2. **Merge Commit**
   - Preserves commit history
   - Shows merge point
   - Good for complex changes

3. **Rebase and Merge**
   - Linear commit history
   - No merge commit
   - Good for simple changes

## 🏷️ Release Process

### Versioning

We use semantic versioning for stack releases:

- **Major** (1.0.0): Breaking changes, new stack structure
- **Minor** (0.1.0): New templates, new features
- **Patch** (0.0.1): Bug fixes, documentation updates

### Release Workflow

1. **Update Version**
   ```bash
   # Update stack configuration
   vim stacks/frontend/.stack-config.yml
   # Update version field
   ```

2. **Create Release Notes**
   ```markdown
   # Frontend Stack v1.2.0

   ## New Templates
   - Added react-vite template
   - Added vue-vite template

   ## Updates
   - Updated nextjs-app to v14
   - Improved documentation

   ## Bug Fixes
   - Fixed build issues in svelte template
   - Resolved dependency conflicts
   ```

3. **Create Release**
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```

4. **Update Main Branch**
   ```bash
   git checkout dev
   git merge stack/frontend
   git push origin dev
   ```

## 🛠️ Development Tools

### Useful Scripts

```bash
# Check stack status
python scripts/branch_manager.py status frontend

# Validate stack configuration
python scripts/branch_manager.py validate

# Generate stack report
python scripts/branch_manager.py report

# Sync template from upstream
./scripts/sync_to_branch.sh template-name upstream-url stack-name

# Create new stack branch
./scripts/create_stack_branch.sh new-stack --description "Description"
```

### IDE Configuration

#### VS Code
```json
{
  "recommendations": [
    "ms-vscode.vscode-typescript-next",
    "esbenp.prettier-vscode",
    "ms-python.python",
    "ms-python.pylint"
  ],
  "settings": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.eslint": true
    }
  }
}
```

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

## 🚨 Troubleshooting

### Common Issues

#### Merge Conflicts
```bash
# Resolve conflicts
git status
# Edit conflicted files
git add .
git commit -m "resolve: merge conflicts"
```

#### Template Sync Issues
```bash
# Check upstream repository
curl -s https://api.github.com/repos/owner/repo

# Verify sync script
chmod +x scripts/sync_to_branch.sh
./scripts/sync_to_branch.sh --help
```

#### Build Failures
```bash
# Check Node.js version
node --version

# Clear cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Getting Help

1. **Check Documentation**: Review this guide and related docs
2. **Search Issues**: Look for similar problems in GitHub issues
3. **Contact Maintainers**: Reach out to stack maintainers
4. **Create Issue**: Report bugs or request features
5. **Join Discussions**: Participate in GitHub discussions

## 📚 Best Practices

### General Guidelines
- **Start Small**: Begin with simple contributions
- **Follow Standards**: Adhere to organization guidelines
- **Test Thoroughly**: Always test your changes
- **Document Changes**: Update documentation as needed
- **Be Patient**: Reviews take time

### Communication
- **Be Respectful**: Treat everyone with respect
- **Be Constructive**: Provide helpful feedback
- **Be Clear**: Use clear, concise language
- **Be Responsive**: Respond to feedback promptly

### Code Quality
- **Write Clean Code**: Follow best practices
- **Add Tests**: Include tests for new features
- **Update Documentation**: Keep docs current
- **Review Your Code**: Self-review before submitting

## 🔗 Related Resources

- [Branch Strategy](./BRANCH_STRATEGY.md) - Overall architecture
- [Stack Branch Guide](./STACK_BRANCH_GUIDE.md) - Working with stacks
- [Trend Detection Integration](./TREND_DETECTION_INTEGRATION.md) - Automated discovery
- [Scripts Documentation](../scripts/README.md) - Tool reference
- [Main CONTRIBUTING.md](../CONTRIBUTING.md) - General contribution guidelines

---

**Last Updated**: 2024-01-15  
**Version**: 1.0  
**Maintainer**: Template Team
