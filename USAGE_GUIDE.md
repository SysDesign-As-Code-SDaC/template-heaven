# Template Heaven Usage Guide

## 🚀 Quick Start with Direct Python Installation

Template Heaven is now installed and ready to use! Here's how to get started:

## 📋 Available Commands

### **1. List Available Templates**
```bash
# List all templates
python -m templateheaven.cli.main list

# List templates by stack
python -m templateheaven.cli.main list --stack frontend
python -m templateheaven.cli.main list --stack backend
python -m templateheaven.cli.main list --stack fullstack
```

### **2. Search Templates**
```bash
# Search for specific technologies
python -m templateheaven.cli.main search --query "react"
python -m templateheaven.cli.main search --query "python"
python -m templateheaven.cli.main search --query "machine learning"

# Limit results
python -m templateheaven.cli.main search --query "typescript" --limit 5
```

### **3. Get Template Information**
```bash
# Get detailed info about a template
python -m templateheaven.cli.main info react-vite
python -m templateheaven.cli.main info fastapi
python -m templateheaven.cli.main info nextjs-fullstack
```

### **4. Initialize Projects**

#### **Interactive Wizard (Recommended)**
```bash
# Launch the interactive wizard
python -m templateheaven.cli.main init
```

The wizard will guide you through:
1. **Stack Selection** - Choose technology category
2. **Template Selection** - Pick specific template
3. **Project Configuration** - Set name, author, license
4. **Confirmation** - Review and create project

#### **Command Line Mode**
```bash
# Create project with specific template
python -m templateheaven.cli.main init --template react-vite --name my-react-app

# Create project with stack selection
python -m templateheaven.cli.main init --stack frontend --name my-frontend-app

# Full command line configuration
python -m templateheaven.cli.main init \
  --template react-vite \
  --name my-awesome-app \
  --author "John Doe" \
  --license MIT \
  --package-manager npm
```

### **5. Configuration Management**
```bash
# View current configuration
python -m templateheaven.cli.main config show

# Set configuration values
python -m templateheaven.cli.main config set author "Your Name"
python -m templateheaven.cli.main config set default_license "MIT"

# Get specific configuration
python -m templateheaven.cli.main config get author
```

### **6. Template Statistics**
```bash
# View template statistics
python -m templateheaven.cli.main stats
```

## 🎯 Available Templates

### **Frontend Templates**
- **react-vite** - React + Vite + TypeScript starter
- **vue-vite** - Vue 3 + Vite + TypeScript starter  
- **svelte-kit** - SvelteKit full-stack application

### **Backend Templates**
- **fastapi** - FastAPI + PostgreSQL + Docker starter
- **express-typescript** - Express.js + TypeScript + MongoDB
- **django-rest** - Django REST Framework + PostgreSQL

### **Fullstack Templates**
- **nextjs-fullstack** - Next.js 14 + TypeScript + Prisma + PostgreSQL
- **t3-stack** - T3 Stack (Next.js + tRPC + Prisma + NextAuth)

### **AI/ML Templates**
- **pytorch-lightning** - PyTorch Lightning + Hydra configuration
- **cookiecutter-datascience** - Cookiecutter Data Science project

### **DevOps Templates**
- **github-actions-python** - GitHub Actions workflow for Python
- **docker-compose-stack** - Docker Compose multi-service stack

### **Mobile Templates**
- **react-native-expo** - React Native + Expo + TypeScript
- **flutter-clean** - Flutter clean architecture

### **Workflow Templates**
- **python-package** - Python package with modern tooling
- **typescript-library** - TypeScript library with modern tooling

## 🐍 Python API Usage

### **Basic Usage**
```python
from templateheaven import TemplateManager, Config

# Initialize components
config = Config()
manager = TemplateManager(config)

# List templates
templates = manager.list_templates()
for template in templates:
    print(f"{template.name}: {template.description}")

# Search templates
results = manager.search_templates("react")
for result in results:
    print(f"{result.template.name} (score: {result.score})")

# Get template by name
template = manager.get_template("react-vite")
if template:
    print(f"Found: {template.name}")
```

### **Advanced Usage**
```python
# Filter by stack
frontend_templates = manager.list_templates(stack="frontend")

# Get template statistics
stats = manager.get_template_stats()
print(f"Total templates: {stats['total_templates']}")

# Get stack information
stacks = manager.get_stacks()
for stack in stacks:
    stack_info = manager.get_stack_info(stack)
    print(f"{stack_info['name']}: {stack_info['template_count']} templates")
```

## ⚙️ Configuration

Template Heaven stores configuration in `~/.templateheaven/config.yaml`:

```yaml
# Default settings
default_author: "Your Name"
default_license: "MIT"
package_managers:
  python: "pip"
  node: "npm"
cache_dir: "~/.templateheaven/cache"
log_level: "INFO"

# Template sources
template_sources:
  bundled:
    enabled: true
  github:
    enabled: false
    token: ""
```

## 🔧 Customization

### **Template Variables**
Templates support Jinja2 variable substitution:

```python
# Available variables
{
    "project_name": "my-awesome-app",
    "author": "John Doe",
    "license": "MIT",
    "version": "1.0.0",
    "description": "My awesome application"
}
```

### **Custom Templates**
You can add custom templates by:

1. **Creating template directory** in `templates/`
2. **Adding metadata** to `stacks.yaml`
3. **Using Jinja2 syntax** in template files

## 🛠️ Development

### **Install Development Dependencies**
```bash
pip install -e ".[dev]"
```

### **Run Tests**
```bash
pytest tests/
```

### **Code Quality**
```bash
# Format code
black templateheaven tests

# Lint code
flake8 templateheaven tests

# Type checking
mypy templateheaven
```

## 🐳 Docker Usage (Optional)

### **Build and Run**
```bash
# Build Docker image
docker build -t templateheaven .

# Run with Docker
docker run -it templateheaven init

# Use Docker Compose
docker-compose up --build
```

### **Development with Docker**
```bash
# Development environment
docker-compose --profile dev up --build

# Testing environment
docker-compose --profile test up --build
```

## 🎉 Examples

### **Create a React App**
```bash
python -m templateheaven.cli.main init --template react-vite --name my-react-app
```

### **Create a FastAPI Backend**
```bash
python -m templateheaven.cli.main init --template fastapi --name my-api
```

### **Create a Next.js Fullstack App**
```bash
python -m templateheaven.cli.main init --template nextjs-fullstack --name my-fullstack-app
```

### **Search and Create**
```bash
# Search for Python templates
python -m templateheaven.cli.main search --query "python"

# Create with found template
python -m templateheaven.cli.main init --template fastapi --name my-python-api
```

## 🆘 Troubleshooting

### **Common Issues**

1. **Command not found**
   ```bash
   # Use Python module approach
   python -m templateheaven.cli.main --help
   ```

2. **Template not found**
   ```bash
   # List available templates
   python -m templateheaven.cli.main list
   ```

3. **Permission errors**
   ```bash
   # Check file permissions
   ls -la ~/.templateheaven/
   ```

### **Get Help**
```bash
# General help
python -m templateheaven.cli.main --help

# Command-specific help
python -m templateheaven.cli.main init --help
python -m templateheaven.cli.main list --help
python -m templateheaven.cli.main search --help
```

## 🎯 Next Steps

1. **Try the interactive wizard**: `python -m templateheaven.cli.main init`
2. **Explore templates**: `python -m templateheaven.cli.main list`
3. **Search for your tech stack**: `python -m templateheaven.cli.main search --query "your-tech"`
4. **Create your first project**: `python -m templateheaven.cli.main init --template react-vite --name my-app`

---

**Template Heaven: Start every project with a template. Build faster, build better!**
