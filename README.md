# Template Heaven

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Interactive template management for modern software development**

Template Heaven is a Python package that provides an interactive CLI and Python API for discovering, customizing, and initializing projects from templates. This simplified repo focuses on a single fullstack starter (Next.js) and interactive discovery via GitHub. The intention is to keep the code lightweight and easy to extend.

## 🚀 Features

This repo always includes a Gold Standard template (`gold-standard-python-service`) with README, LICENSE, CONTRIBUTING and other best-practice files for new projects; if a user lacks these files, the customizer will add them during scaffolding.

### MVP (Current)

- **Interactive CLI** with beautiful terminal output using Rich
- **Wizard-style project initialization** with guided prompts
- **Fullstack (Next.js)** starter template and GitHub discovery-focused features
- **Local template bundling** with metadata caching
- **Template search and filtering** by stack, tags, and keywords
**github-actions-python**: GitHub Actions template example (automation disabled)
- **Configuration management** with YAML storage
- **Comprehensive documentation** and type hints

### Coming Soon (Phase 2+)

- **GitHub live search** for real-time template discovery
- **Web UI** with Streamlit for visual template browsing
- **Trend detection integration** for discovering popular templates
- **Custom template management** for organization-specific templates
- **Advanced customization options** and template validation

## 📦 Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/template-heaven/templateheaven.git
cd templateheaven

# Install in development mode
pip install -e ".[dev]"

# Or use the Makefile
make install-dev
```

### From PyPI (Coming Soon)

```bash
pip install templateheaven
```

## 🎯 Quick Start

### Interactive Wizard (Recommended)

```bash
# Launch the interactive wizard
templateheaven init

# The wizard will guide you through:
# 1. Selecting a technology stack
# 2. Choosing a template
# 3. Configuring your project
# 4. Creating the project
```

### Command Line Mode

```bash
# Initialize with specific template
templateheaven init --template react-vite --name my-app

# List available templates
templateheaven list
By default, if `prefer_github` is true in your config, the CLI will perform live discovery on GitHub (if a token is present); local bundled templates are used as a fallback.

# List templates by stack
templateheaven list --stack frontend

# Force listing from a specific source (auto, github, local)
templateheaven list --source local

# Include archived local templates in the result
templateheaven list --include-archived --source local

# Search templates (local)
templateheaven search "machine learning"

# Search templates on GitHub (live)
templateheaven search --source github "nextjs"

# Get template information
templateheaven info react-vite

# Configure settings
templateheaven config set default_author "John Doe"
```

### Yeoman Generator (experimental)

There is an experimental Yeoman generator included to provide a simple Node-based scaffolding workflow that reads the same bundled template metadata and copies template files into your directory.

To run the generator locally, install the generator dependencies and link it locally:

```bash
cd generators/templateheaven-generator
npm install
npm link
yo templateheaven
```

The generator is a small starter that lists stacks and templates from `templateheaven/data/stacks.yaml` and scaffolds a selected template into the current working directory. It is a simple and user-friendly alternative for quick scaffolds and will remain a companion to the Python CLI and API.


### API Examples

```python
from templateheaven import TemplateManager, Wizard, Config

# Initialize components
config = Config()
manager = TemplateManager(config)

# List templates
templates = manager.list_templates(stack='frontend')
for template in templates:
    print(f"{template.name}: {template.description}")

# Search templates
results = manager.search_templates("react typescript")
for result in results:
    print(f"{result.template.name} (score: {result.score})")

# Use the wizard programmatically
wizard = Wizard(manager, config)
wizard.run()
```

## 🏗️ Technology stacks

This repository is simplified to focus on an exemplary Fullstack template and GitHub-powered discovery. The canonical bundled template is the Next.js fullstack starter.

### Fullstack

- **nextjs-fullstack-app**: Next.js fullstack starter with Prisma, tRPC, and NextAuth

## 📋 Available Templates

### Frontend Templates

- **react-vite**: React + Vite + TypeScript starter
- **vue-vite**: Vue 3 + Vite + TypeScript starter
- **svelte-kit**: SvelteKit full-stack application

### Backend Templates

- **fastapi**: FastAPI + PostgreSQL + Docker starter
- **express-typescript**: Express.js + TypeScript + MongoDB
- **django-rest**: Django REST Framework + PostgreSQL

### Fullstack Templates

- **nextjs-fullstack**: Next.js 14 + TypeScript + Prisma + PostgreSQL
- **t3-stack**: T3 Stack - Next.js + tRPC + Prisma + NextAuth

### AI/ML Templates

- **pytorch-lightning**: PyTorch Lightning + Hydra configuration
- **cookiecutter-datascience**: Cookiecutter Data Science project

### DevOps Templates

- **github-actions-python**: GitHub Actions template example (automation disabled)
- **docker-compose-stack**: Docker Compose multi-service stack

### Mobile Templates

- **react-native-expo**: React Native + Expo + TypeScript
- **flutter-clean**: Flutter clean architecture template

### Workflow Templates

- **python-package**: Python package with modern tooling
- **typescript-library**: TypeScript library with modern tooling

## ⚙️ Configuration

Template Heaven stores configuration in `~/.templateheaven/config.yaml`:

```yaml
# Cache settings
cache_dir: "~/.templateheaven/cache"

# Default project settings
default_author: "Your Name"
default_license: "MIT"

# Package manager preferences
package_managers:
  python: "pip"
  node: "npm"
  rust: "cargo"
  go: "go"

# Optional GitHub integration
github_token: null  # Set for live search (Phase 2)
  # Whether to prefer GitHub live discovery over local bundled templates
  prefer_github: true

# UI preferences
auto_update: true
log_level: "INFO"
```

### Configuration Commands

```bash
# List all configuration
templateheaven config --list-all

# Set configuration values
templateheaven config --key default_author --value "John Doe"
templateheaven config --key github_token --value "ghp_xxxxx"

# Remove configuration
templateheaven config --unset github_token

# Reset to defaults
templateheaven config --reset
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/template-heaven/templateheaven.git
cd templateheaven
make setup

# Or manually
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_models.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check
```

### Building

```bash
# Build package
make build

# Clean build artifacts
make clean
```

## 📚 Documentation

### CLI Reference

```bash
# Get help for any command
templateheaven --help
templateheaven init --help
templateheaven list --help
templateheaven config --help
```

### Python API

```python
# Core classes
from templateheaven import TemplateManager, Config, Wizard
from templateheaven.core.models import Template, ProjectConfig, StackCategory

# Template management
manager = TemplateManager()
templates = manager.list_templates(stack='frontend')
template = manager.get_template('react-vite')

# Search functionality
results = manager.search_templates('machine learning', limit=10)

# Configuration
config = Config()
config.set('default_author', 'John Doe')
author = config.get('default_author')

# Project creation
from templateheaven.core.customizer import Customizer
customizer = Customizer()
# ... customize and create project
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `make check`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards

- **Type hints** required for all functions and methods
- **Comprehensive docstrings** with examples
- **Unit tests** with 80%+ coverage
- **Manual validation** before automated testing
- **Black formatting** and **flake8 linting**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Template Heaven Team** for the vision and architecture
- **Open Source Community** for the amazing templates we bundle
- **Rich** and **Click** for beautiful terminal interfaces
- **Jinja2** for powerful templating capabilities

## 🔗 Links

- **Documentation**: [templateheaven.dev/docs](https://templateheaven.dev/docs)
- **GitHub**: [github.com/template-heaven/templateheaven](https://github.com/template-heaven/templateheaven)
- **Issues**: [github.com/template-heaven/templateheaven/issues](https://github.com/template-heaven/templateheaven/issues)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## 🚧 Roadmap

- ### Phase 1: MVP ✅

- [x] Core CLI with interactive wizard
- [x] Template management and search
- [x] Project customization with Jinja2
- [x] Configuration system
- [x] Comprehensive testing

- ### Phase 2: Enhanced Discovery

- [ ] GitHub live search integration
- [ ] Template validation engine
- [ ] Advanced customization options
- [ ] Repository handler for git operations

- ### Phase 3: Advanced Features

- [ ] Web UI with Streamlit
- [ ] Trend detection integration
- [ ] Custom template management
- [ ] Analytics and recommendations

- ### Phase 4: Gold Standard Stack

- [ ] Workflows & best practices templates
- [ ] Repository structure templates
- [ ] Automation and code quality configs
- [ ] Documentation and project management templates

---

> **Made with ❤️ by the Template Heaven Team**

Start every project with a template. Build faster, build better.
