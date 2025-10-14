# How Template Heaven Works

## 🏗️ Architecture Overview

Template Heaven is a comprehensive Python package that provides an interactive CLI and Python API for discovering, customizing, and initializing projects from templates across 24+ technology stacks. Here's how it works:

## 📋 Core Components

### 1. **Template Manager** (`templateheaven/core/template_manager.py`)
The central component that handles template discovery, filtering, and search.

```python
# Core functionality
class TemplateManager:
    def list_templates(self, stack=None, tags=None, search=None):
        """List and filter templates based on criteria"""
        
    def get_template(self, name):
        """Get a specific template by name"""
        
    def search_templates(self, query, limit=10, min_score=0.1):
        """Search templates with relevance scoring"""
        
    def get_stacks(self):
        """Get all available technology stacks"""
```

**How it works:**
- Loads template metadata from `templateheaven/data/stacks.yaml`
- Provides filtering by stack category, tags, and search terms
- Implements relevance scoring for search results
- Caches template information for performance

### 2. **Configuration System** (`templateheaven/config/settings.py`)
Manages user preferences and application settings.

```python
class Config:
    def __init__(self, config_dir=None):
        """Initialize configuration from YAML file"""
        
    def get(self, key, default=None):
        """Get configuration value"""
        
    def set(self, key, value):
        """Set configuration value"""
        
    def save_config(self):
        """Save configuration to disk"""
```

**Configuration locations:**
- `~/.templateheaven/config.yaml` - User settings
- `~/.templateheaven/cache/` - Template cache
- Environment variables for sensitive data

### 3. **Customizer** (`templateheaven/core/customizer.py`)
Handles template customization and project generation.

```python
class Customizer:
    def customize(self, template, config, output_dir):
        """Customize and initialize project from template"""
        
    def process_template_file(self, content, variables):
        """Process template content with Jinja2"""
        
    def get_template_variables(self, config):
        """Get all template variables for a project"""
```

**Template processing:**
- Uses Jinja2 for variable substitution
- Supports custom filters (snake_case, kebab_case, etc.)
- Handles file copying and directory creation
- Generates package files (package.json, pyproject.toml, etc.)

### 4. **Interactive Wizard** (`templateheaven/cli/wizard.py`)
Provides a guided, step-by-step project creation experience.

```python
class Wizard:
    def run(self, output_dir=Path('.')):
        """Run the complete wizard flow"""
        
    def _select_stack(self):
        """Interactive stack selection"""
        
    def _select_template(self, stack):
        """Interactive template selection"""
        
    def _configure_project(self, template, output_dir):
        """Configure project settings"""
```

**Wizard flow:**
1. **Welcome message** with project overview
2. **Stack selection** - Choose technology category
3. **Template selection** - Pick specific template
4. **Project configuration** - Set name, author, license, etc.
5. **Confirmation** - Review and confirm creation
6. **Project creation** - Generate the project

### 5. **CLI Interface** (`templateheaven/cli/main.py`)
Command-line interface using Click framework.

```python
@click.group()
def cli():
    """Template Heaven - Interactive template management"""

@cli.command()
def init():
    """Initialize a new project from a template"""

@cli.command()
def list():
    """List available templates"""

@cli.command()
def search():
    """Search templates with relevance scoring"""
```

## 🔄 Workflow Examples

### Example 1: Interactive Project Creation

```bash
# User runs the wizard
$ templateheaven init

# 1. Welcome screen appears
Welcome to Template Heaven! 🎉
This wizard will help you create a new project from one of our templates.

# 2. Stack selection
Step 1: Select Technology Stack
Choose a technology stack:
> Frontend Frameworks (15 templates)
  Backend Services (12 templates)
  Fullstack Applications (8 templates)
  AI/ML & Data Science (6 templates)
  Gold Standard Templates (5 templates)

# 3. Template selection
Step 2: Select Template
Choose a template from frontend:
> React + Vite + TypeScript starter
  Vue 3 + Vite + TypeScript starter
  SvelteKit full-stack application

# 4. Project configuration
Step 3: Configure Project
Project name: my-awesome-app
Author: John Doe
License: MIT
Package manager: npm

# 5. Confirmation and creation
Step 4: Confirm Creation
Project Preview:
- Name: my-awesome-app
- Template: react-vite
- Stack: frontend
- Author: John Doe
- License: MIT

Create this project? [Y/n]: Y

Creating Project...
✅ Project created successfully: ./my-awesome-app
```

### Example 2: Command-Line Usage

```bash
# List all templates
$ templateheaven list

# List templates by stack
$ templateheaven list --stack frontend

# Search templates
$ templateheaven search "machine learning"

# Get template info
$ templateheaven info react-vite

# Initialize with specific template
$ templateheaven init --template react-vite --name my-app --author "John Doe"
```

### Example 3: Python API Usage

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

## 🗂️ Template Structure

### Template Metadata (`stacks.yaml`)

```yaml
stacks:
  frontend:
    name: "Frontend Frameworks"
    description: "Frontend frameworks and UI libraries"
    templates:
      - name: "react-vite"
        description: "React + Vite + TypeScript starter"
        tags: ["react", "vite", "typescript", "frontend"]
        dependencies:
          react: "^18.2.0"
          typescript: "^5.0.0"
          vite: "^4.4.0"
        upstream_url: "https://github.com/vitejs/vite"
        version: "1.0.0"
        author: "Vite Team"
        license: "MIT"
        features: ["TypeScript", "Hot Reload", "ESLint"]
        min_node_version: "16.0.0"
```

### Template Files

```
templates/gold-standard-python-service/
├── README.md                    # Template documentation
├── pyproject.toml              # Python package config
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   └── models/                 # Data models
├── tests/                      # Test suite
├── docker/                     # Docker configs
└── .github/                    # CI/CD workflows
```

## 🔧 Template Processing

### 1. **Variable Substitution**
Templates use Jinja2 syntax for customization:

```python
# In template files
{{ project_name | title }}  # "my-app" → "My App"
{{ project_name | snake_case }}  # "my-app" → "my_app"
{{ author }}  # "John Doe"
{{ license }}  # "MIT"
```

### 2. **File Generation**
The customizer generates project files:

```python
def _create_package_json(self, project_path, template, variables):
    package_data = {
        "name": variables['project_name'],
        "version": variables['version'],
        "description": variables['project_description'],
        "author": variables['author'],
        "license": variables['license'],
        "dependencies": template.dependencies,
    }
    
    package_json_path = project_path / 'package.json'
    self.file_ops.write_file(
        package_json_path,
        json.dumps(package_data, indent=2)
    )
```

### 3. **Directory Structure Creation**
```python
def _create_basic_project_structure(self, project_path, template, variables):
    # Create directories based on template stack
    if template.stack.value in ['frontend', 'fullstack']:
        self.file_ops.create_directory(project_path / 'src')
        self.file_ops.create_directory(project_path / 'public')
    
    if template.stack.value in ['backend', 'fullstack']:
        self.file_ops.create_directory(project_path / 'src')
        self.file_ops.create_directory(project_path / 'tests')
```

## 🔍 Search and Discovery

### Relevance Scoring Algorithm

```python
def _calculate_relevance_score(self, template, query):
    score = 0.0
    query_lower = query.lower()
    
    # Exact name match (highest priority)
    if template.name.lower() == query_lower:
        return 1.0
    
    # Name contains query
    if query_lower in template.name.lower():
        score = max(score, 0.8)
    
    # Description contains query
    if query_lower in template.description.lower():
        score = max(score, 0.6)
    
    # Tag matches
    tag_matches = sum(1 for tag in template.tags if query_lower in tag.lower())
    if tag_matches > 0:
        score = max(score, 0.4 * (tag_matches / len(template.tags)))
    
    # Stack category match
    if query_lower in template.stack.value.lower():
        score = max(score, 0.3)
    
    return min(score, 1.0)
```

### Search Results

```python
class TemplateSearchResult:
    template: Template
    score: float  # 0.0 to 1.0
    match_reason: str  # "Exact name match", "Tag match", etc.
```

## 🛡️ Security and Quality

### 1. **Input Validation**
```python
def validate_project_name(name: str) -> None:
    if not name:
        raise ValueError("Project name cannot be empty")
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError("Project name can only contain letters, numbers, hyphens, and underscores")
    if len(name) > 50:
        raise ValueError("Project name must be 50 characters or less")
```

### 2. **Error Handling**
```python
class TemplateNotFoundError(TemplateError):
    def __init__(self, template_name: str, available_templates: Optional[list] = None):
        message = f"Template '{template_name}' not found"
        suggestion = "Use 'templateheaven list' to see available templates"
        super().__init__(message, "TEMPLATE_NOT_FOUND", details, suggestion)
```

### 3. **Security Scanning**
- **Dependency scanning** with Safety and pip-audit
- **Code analysis** with Bandit and Semgrep
- **Container scanning** with Trivy
- **Secrets detection** with TruffleHog

## 🚀 Deployment and CI/CD

### Docker Multi-stage Build

```dockerfile
# Build stage
FROM python:3.11-slim as builder
# Install dependencies and build

# Production stage
FROM python:3.11-slim as production
# Copy built artifacts and run as non-root user
USER templateheaven
CMD ["templateheaven", "--help"]
```

### GitHub Actions Pipeline

```yaml
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - name: Code formatting check
        run: black --check templateheaven tests
      - name: Linting
        run: flake8 templateheaven tests
      - name: Security scan
        run: bandit -r templateheaven/
  
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    steps:
      - name: Run tests
        run: pytest --cov=templateheaven
```

## 📊 Monitoring and Observability

### Structured Logging
```python
import structlog

logger = structlog.get_logger(__name__)

logger.info(
    "Template created successfully",
    template_name=template.name,
    project_name=config.name,
    output_dir=str(output_dir)
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'templateheaven_requests_total',
    'Total requests',
    ['operation', 'status']
)

OPERATION_DURATION = Histogram(
    'templateheaven_operation_duration_seconds',
    'Operation duration',
    ['operation']
)
```

## 🔄 Data Flow

### 1. **Template Discovery**
```
User Request → TemplateManager → stacks.yaml → Filter/Search → Results
```

### 2. **Project Creation**
```
Template Selection → Configuration → Customizer → File Generation → Project
```

### 3. **Error Handling**
```
Exception → Custom Exception → User-Friendly Message → Suggestion
```

## 🎯 Key Features

### **Interactive Experience**
- Beautiful terminal output with Rich
- Step-by-step wizard guidance
- Real-time validation and feedback
- Helpful error messages with suggestions

### **Flexible Usage**
- CLI commands for quick operations
- Python API for programmatic use
- Interactive wizard for guided setup
- Configuration management

### **Production Ready**
- Comprehensive testing
- Security scanning
- Docker containerization
- CI/CD automation
- Monitoring and observability

### **Extensible Architecture**
- Plugin system for custom templates
- Configurable validation rules
- Customizable output formats
- Modular component design

## 🚀 Getting Started

### Installation
```bash
# Development installation
git clone https://github.com/template-heaven/templateheaven.git
cd templateheaven
pip install -e ".[dev]"
```

### Basic Usage
```bash
# Interactive wizard
templateheaven init

# Command line
templateheaven init --template react-vite --name my-app

# List templates
templateheaven list --stack frontend

# Search
templateheaven search "machine learning"
```

### Python API
```python
from templateheaven import TemplateManager, Config

config = Config()
manager = TemplateManager(config)
templates = manager.list_templates(stack='frontend')
```

## 🎉 Summary

Template Heaven works by providing a comprehensive, interactive system for template discovery and project initialization. It combines:

- **Template management** with metadata and search
- **Interactive wizard** for guided project creation
- **Template customization** with Jinja2 processing
- **Configuration management** for user preferences
- **Security and quality** with comprehensive scanning
- **Production readiness** with Docker and CI/CD

The result is a powerful tool that helps developers quickly start new projects with production-ready templates while following software engineering best practices.

---

**Template Heaven: Start every project with a template. Build faster, build better.**
