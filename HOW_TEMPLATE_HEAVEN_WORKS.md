# How Template Heaven Works - Complete Explanation

## 🎯 Overview

Template Heaven is a comprehensive Python package that provides an interactive CLI and Python API for discovering, customizing, and initializing projects from templates across 24+ technology stacks. It's designed to help developers quickly start new projects with production-ready templates while following software engineering best practices.

## 🏗️ Architecture

### Core Components

```
Template Heaven
├── Template Manager (Discovery & Search)
├── Configuration System (User Preferences)
├── Customizer (Template Processing)
├── Interactive Wizard (User Experience)
├── CLI Interface (Command Line)
└── Data Layer (Template Metadata)
```

### Data Flow

```
User Input → CLI/Wizard → Template Manager → Customizer → Project Output
     ↓              ↓              ↓              ↓
Configuration → Validation → Template Selection → File Generation
```

## 🔍 How It Works - Step by Step

### 1. **Template Discovery**

Template Heaven loads template metadata from `templateheaven/data/stacks.yaml`:

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

**What happens:**
- `TemplateManager` loads and parses the YAML data
- Creates `Template` objects with all metadata
- Caches templates for performance
- Provides filtering and search capabilities

### 2. **Template Search & Filtering**

The system provides multiple ways to find templates:

```python
# List all templates
templates = manager.list_templates()

# Filter by stack
frontend_templates = manager.list_templates(stack="frontend")

# Search with relevance scoring
results = manager.search_templates("react typescript", limit=5)
```

**Search Algorithm:**
- **Exact name match**: Score = 1.0 (highest priority)
- **Name contains query**: Score = 0.8
- **Description contains query**: Score = 0.6
- **Tag matches**: Score = 0.4 × (matches/total_tags)
- **Stack category match**: Score = 0.3

### 3. **Interactive Wizard Flow**

When you run `templateheaven init`, here's what happens:

```
Step 1: Welcome Screen
├── Display project overview
├── Show available stacks
└── Guide user to selection

Step 2: Stack Selection
├── List all technology stacks
├── Show template counts per stack
└── Allow user to choose category

Step 3: Template Selection
├── List templates in selected stack
├── Show descriptions and features
└── Allow user to pick template

Step 4: Project Configuration
├── Collect project name
├── Set author information
├── Choose license
└── Configure package manager

Step 5: Confirmation
├── Show project summary
├── Display configuration
└── Confirm creation

Step 6: Project Creation
├── Generate project structure
├── Process template files
├── Apply customizations
└── Create final project
```

### 4. **Template Customization**

The `Customizer` class handles template processing:

```python
class Customizer:
    def customize(self, template, config, output_dir):
        # 1. Create project directory structure
        self._create_project_structure(output_dir, template)
        
        # 2. Process template files with Jinja2
        self._process_template_files(template, config, output_dir)
        
        # 3. Generate package configuration files
        self._generate_package_files(template, config, output_dir)
        
        # 4. Apply customizations
        self._apply_customizations(template, config, output_dir)
```

**Template Processing:**
- **Variable Substitution**: Uses Jinja2 for `{{ variable }}` replacement
- **File Generation**: Creates `package.json`, `pyproject.toml`, etc.
- **Directory Structure**: Creates appropriate folder hierarchy
- **Customization**: Applies user preferences and settings

### 5. **Configuration Management**

The system manages user preferences:

```python
class Config:
    def __init__(self):
        # Load from ~/.templateheaven/config.yaml
        self.config = self._load_config()
    
    def get(self, key, default=None):
        # Get configuration value with fallback
        return self.config.get(key, default)
    
    def set(self, key, value):
        # Set configuration value and save
        self.config[key] = value
        self._save_config()
```

**Configuration includes:**
- Default author name
- Preferred license
- Package manager preferences
- Cache settings
- Log level
- Custom template sources

## 🚀 Usage Examples

### Command Line Interface

```bash
# Interactive wizard
templateheaven init

# List all templates
templateheaven list

# List templates by stack
templateheaven list --stack frontend

# Search templates
templateheaven search --query "react typescript"

# Get template information
templateheaven info react-vite

# Manage configuration
templateheaven config set author "John Doe"
templateheaven config get default_license
```

### Python API

```python
from templateheaven import TemplateManager, Config, Wizard

# Initialize components
config = Config()
manager = TemplateManager(config)

# List templates
templates = manager.list_templates(stack='frontend')
for template in templates:
    print(f"{template.name}: {template.description}")

# Search templates
results = manager.search_templates("machine learning")
for result in results:
    print(f"{result.template.name} (score: {result.score})")

# Use the wizard
wizard = Wizard(manager, config)
wizard.run()
```

## 🎨 Template Structure

### Template Metadata

Each template includes:

```python
@dataclass
class Template:
    name: str                    # Unique identifier
    display_name: str           # User-friendly name
    description: str            # Template description
    stack: StackCategory        # Technology stack
    tags: List[str]            # Search tags
    dependencies: Dict[str, str] # Package dependencies
    upstream_url: str          # Source repository
    version: str               # Template version
    author: str                # Template author
    license: str               # License type
    features: List[str]        # Key features
    min_node_version: str      # Minimum Node.js version
    min_python_version: str    # Minimum Python version
```

### Template Files

Templates are stored in the `templates/` directory:

```
templates/
├── gold-standard-python-service/
│   ├── README.md
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   └── models/
│   ├── tests/
│   ├── docker/
│   └── .github/
├── modern-react-app/
│   ├── README.md
│   ├── package.json
│   ├── src/
│   ├── tests/
│   └── docker/
└── fastapi-microservice/
    ├── README.md
    ├── app/
    ├── tests/
    └── docker/
```

## 🔧 Template Processing

### Jinja2 Variable Substitution

Templates use Jinja2 syntax for customization:

```python
# In template files
{{ project_name | title }}     # "my-app" → "My App"
{{ project_name | snake_case }} # "my-app" → "my_app"
{{ author }}                   # "John Doe"
{{ license }}                  # "MIT"
{{ version }}                  # "1.0.0"
```

### File Generation

The customizer generates project files:

```python
def _generate_package_json(self, template, config, output_dir):
    package_data = {
        "name": config.project_name,
        "version": config.version,
        "description": config.description,
        "author": config.author,
        "license": config.license,
        "dependencies": template.dependencies,
    }
    
    package_json_path = output_dir / 'package.json'
    self.file_ops.write_file(
        package_json_path,
        json.dumps(package_data, indent=2)
    )
```

### Directory Structure Creation

```python
def _create_project_structure(self, output_dir, template):
    # Create base directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create stack-specific directories
    if template.stack in [StackCategory.FRONTEND, StackCategory.FULLSTACK]:
        (output_dir / 'src').mkdir(exist_ok=True)
        (output_dir / 'public').mkdir(exist_ok=True)
    
    if template.stack in [StackCategory.BACKEND, StackCategory.FULLSTACK]:
        (output_dir / 'src').mkdir(exist_ok=True)
        (output_dir / 'tests').mkdir(exist_ok=True)
```

## 🛡️ Security & Quality

### Input Validation

```python
def validate_project_name(name: str) -> None:
    if not name:
        raise ValueError("Project name cannot be empty")
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError("Project name can only contain letters, numbers, hyphens, and underscores")
    if len(name) > 50:
        raise ValueError("Project name must be 50 characters or less")
```

### Error Handling

```python
class TemplateNotFoundError(TemplateError):
    def __init__(self, template_name: str, available_templates: Optional[list] = None):
        message = f"Template '{template_name}' not found"
        suggestion = "Use 'templateheaven list' to see available templates"
        super().__init__(message, "TEMPLATE_NOT_FOUND", details, suggestion)
```

### Security Scanning

- **Dependency scanning** with Safety and pip-audit
- **Code analysis** with Bandit and Semgrep
- **Container scanning** with Trivy
- **Secrets detection** with TruffleHog

## 📊 Performance & Caching

### Template Caching

```python
class Cache:
    def __init__(self, cache_dir: Path, default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.default_ttl = timedelta(seconds=default_ttl)
    
    def get(self, key: str) -> Optional[Any]:
        # Check if cached data exists and is not expired
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if datetime.now() < datetime.fromisoformat(data["expiry"]):
                    return data["value"]
        return None
```

### Performance Optimization

- **Lazy loading** of template data
- **LRU caching** for frequently accessed templates
- **Parallel processing** for multiple template operations
- **Efficient search** with indexed metadata

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
name: CI/CD Pipeline
on: [push, pull_request]

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

### Docker Support

```dockerfile
# Multi-stage build for security
FROM python:3.11-slim as builder
# Install dependencies and build

FROM python:3.11-slim as production
# Copy built artifacts and run as non-root user
USER templateheaven
CMD ["templateheaven", "--help"]
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
templateheaven search --query "machine learning"
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
