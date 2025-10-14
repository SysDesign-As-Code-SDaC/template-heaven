"""
Stack Documentation Generator for Template Heaven.

This module provides automated generation of comprehensive documentation
for all stack branches based on stack configurations and template data.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .stack_config import get_stack_config_manager
from .template_manager import TemplateManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StackDocumentationGenerator:
    """
    Generates comprehensive documentation for stack branches.

    Creates detailed README files with stack-specific information,
    technology overviews, usage guides, and template listings.
    """

    def __init__(self, config=None):
        """
        Initialize the documentation generator.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.stack_config = get_stack_config_manager()
        self.template_manager = TemplateManager(config)

    def generate_all_stack_documentation(self) -> Dict[str, Any]:
        """
        Generate documentation for all stack branches.

        Returns:
            Generation results summary
        """
        results = {
            "stacks_processed": 0,
            "successful_generations": 0,
            "errors": [],
            "stack_results": {}
        }

        all_stacks = self.stack_config.get_all_stacks()
        results["stacks_processed"] = len(all_stacks)

        for stack_name in all_stacks:
            try:
                logger.info(f"Generating documentation for stack: {stack_name}")
                result = self.generate_stack_documentation(stack_name)
                results["stack_results"][stack_name] = result

                if result["success"]:
                    results["successful_generations"] += 1
                else:
                    results["errors"].extend(result["errors"])

            except Exception as e:
                logger.error(f"Failed to generate documentation for {stack_name}: {e}")
                results["errors"].append(f"{stack_name}: {str(e)}")
                results["stack_results"][stack_name] = {
                    "success": False,
                    "error": str(e)
                }

        logger.info(f"Documentation generation completed: {results['successful_generations']}/{results['stacks_processed']} stacks successful")
        return results

    def generate_stack_documentation(self, stack_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive documentation for a specific stack.

        Args:
            stack_name: Name of the stack to document

        Returns:
            Generation result
        """
        result = {
            "success": False,
            "files_generated": [],
            "errors": []
        }

        try:
            # Get stack configuration
            stack_config = self.stack_config.get_stack_config(stack_name)
            if not stack_config:
                raise ValueError(f"Stack configuration not found: {stack_name}")

            # Create stacks directory structure
            stacks_dir = Path("stacks")
            stack_dir = stacks_dir / stack_name
            stack_dir.mkdir(parents=True, exist_ok=True)

            # Generate main README
            readme_content = self._generate_main_readme(stack_name, stack_config)
            readme_path = stack_dir / "README.md"
            readme_path.write_text(readme_content)
            result["files_generated"].append(str(readme_path))

            # Generate TEMPLATES.md if templates exist
            templates_content = self._generate_templates_md(stack_name)
            if templates_content:
                templates_path = stack_dir / "TEMPLATES.md"
                templates_path.write_text(templates_content)
                result["files_generated"].append(str(templates_path))

            result["success"] = True
            logger.info(f"Successfully generated documentation for stack {stack_name}")

        except Exception as e:
            logger.error(f"Failed to generate documentation for {stack_name}: {e}")
            result["errors"].append(str(e))

        return result

    def _generate_main_readme(self, stack_name: str, stack_config: Any) -> str:
        """
        Generate the main README content for a stack.

        Args:
            stack_name: Stack name
            stack_config: Stack configuration object

        Returns:
            README content as string
        """
        # Get stack metadata
        stack_info = self._get_stack_info(stack_name)
        templates = self._get_stack_templates(stack_name)

        # Generate content sections
        header = self._generate_header_section(stack_name, stack_config, stack_info)
        overview = self._generate_overview_section(stack_name, stack_config)
        templates_section = self._generate_templates_section(templates, stack_name)
        quick_start = self._generate_quick_start_section(stack_name, stack_config)
        technologies = self._generate_technologies_section(stack_name, stack_config)
        workflow = self._generate_workflow_section(stack_name, stack_config)
        deployment = self._generate_deployment_section(stack_name, stack_config)
        contributing = self._generate_contributing_section(stack_name)
        statistics = self._generate_statistics_section(templates, stack_info)

        # Combine all sections
        content = "\n".join([
            header,
            overview,
            templates_section,
            quick_start,
            technologies,
            workflow,
            deployment,
            contributing,
            statistics
        ])

        return content

    def _generate_header_section(self, stack_name: str, stack_config: Any, stack_info: Dict) -> str:
        """Generate the header section."""
        title = stack_info.get("display_name", stack_name.replace("-", " ").title())
        description = stack_config.description

        return f"""# {title} Stack

{description}

## 🚀 Overview

{self._get_stack_overview_description(stack_name)}

## 📋 Available Templates

| Template | Description | Technologies | Stars | Last Updated |
|----------|-------------|--------------|-------|--------------|
{self._generate_templates_table(templates, stack_name)}

## 🛠️ Technology Focus

{self._get_technology_focus_description(stack_name)}
"""

    def _generate_overview_section(self, stack_name: str, stack_config: Any) -> str:
        """Generate the overview section."""
        return f"""
### Core Technologies
{chr(10).join(f"- **{tech.title()}**" for tech in stack_config.technologies[:8])}

### Quality Standards
{chr(10).join(f"- {standard}" for standard in stack_config.quality_standards[:5])}

### Use Cases
{self._get_use_cases_description(stack_name)}
"""

    def _generate_templates_section(self, templates: List[Dict], stack_name: str) -> str:
        """Generate the templates section."""
        if not templates:
            return """
## 📦 Templates

*Templates will be added as they are discovered and approved through our automated quality validation process.*
"""

        template_sections = []
        for template in templates:
            template_sections.append(self._generate_template_entry(template))

        return f"""
## 📦 Available Templates

{chr(10).join(template_sections)}
"""

    def _generate_quick_start_section(self, stack_name: str, stack_config: Any) -> str:
        """Generate the quick start section."""
        return f"""
## 🚀 Quick Start

### Using a Template

1. **Browse Templates**: Check the available templates above
2. **Choose Template**: Select the template that best fits your needs
3. **Copy Template**: Copy the template to your new project
4. **Install Dependencies**: Install required packages
5. **Start Development**: Begin building your application

### Example Usage

```bash
# Copy a template to your new project
cp -r stacks/{stack_name}/template-name ../my-new-project
cd ../my-new-project

# Install dependencies and run
{self._get_installation_commands(stack_name)}
```

### Environment Setup

{self._get_environment_setup_description(stack_name)}
"""

    def _generate_technologies_section(self, stack_name: str, stack_config: Any) -> str:
        """Generate the technologies section."""
        return f"""
## 🛠️ Technology Stack

### Frameworks & Libraries
{chr(10).join(f"- **{tech.title()}**: {self._get_tech_description(tech)}" for tech in stack_config.technologies[:10])}

### Development Tools
{self._get_development_tools_description(stack_name)}

### Best Practices
{chr(10).join(f"- {standard}" for standard in stack_config.quality_standards)}
"""

    def _generate_workflow_section(self, stack_name: str, stack_config: Any) -> str:
        """Generate the development workflow section."""
        return f"""
## 🔧 Development Workflow

### Local Development
1. **Clone Template**: Copy the template to your project
2. **Install Dependencies**: Run appropriate package manager
3. **Start Development**: Run development server
4. **Open Browser**: Navigate to localhost
5. **Hot Reload**: Make changes and see them instantly

### Testing
{self._get_testing_description(stack_name)}

### Code Quality
{self._get_code_quality_description(stack_name)}

### Building
{self._get_building_description(stack_name)}
"""

    def _generate_deployment_section(self, stack_name: str, stack_config: Any) -> str:
        """Generate the deployment section."""
        return f"""
## 🚀 Deployment

### Supported Platforms
{self._get_deployment_platforms(stack_name)}

### Deployment Steps
1. **Build Application**: Create production build
2. **Test Production Build**: Verify functionality
3. **Deploy to Platform**: Use platform-specific deployment
4. **Configure CDN**: Set up content delivery
5. **Monitor Performance**: Set up monitoring

### Performance Optimization
{self._get_performance_optimization_description(stack_name)}
"""

    def _generate_contributing_section(self, stack_name: str) -> str:
        """Generate the contributing section."""
        return f"""
## 🤝 Contributing

### Adding New Templates
1. **Research**: Find trending templates in {stack_name}
2. **Review**: Ensure quality and best practices
3. **Test**: Verify the template works correctly
4. **Document**: Add comprehensive documentation
5. **Submit PR**: Create a pull request for review

### Template Requirements
{self._get_template_requirements(stack_name)}

### Quality Standards
{self._get_quality_standards_description(stack_name)}
"""

    def _generate_statistics_section(self, templates: List[Dict], stack_info: Dict) -> str:
        """Generate the statistics section."""
        total_stars = sum(t.get("stars", 0) for t in templates)

        return f"""
## 📊 Statistics

- **Total templates**: {len(templates)}
- **Total stars**: {total_stars}+
- **Last updated**: {datetime.now().strftime('%Y-%m-%d')}
- **Next review**: {(datetime.now().replace(day=15) if datetime.now().day < 15 else datetime.now().replace(month=datetime.now().month + 1, day=15)).strftime('%Y-%m-%d')}

## 🔗 Related Resources

### Stack Branches
- **[Fullstack Stack](../../tree/stack/fullstack)** - Full-stack application templates
- **[Frontend Stack](../../tree/stack/frontend)** - Frontend application templates
- **[Backend Stack](../../tree/stack/backend)** - Backend service templates

### Documentation
- **[Branch Strategy](../../docs/BRANCH_STRATEGY.md)** - Architecture overview
- **[Stack Branch Guide](../../docs/STACK_BRANCH_GUIDE.md)** - Working with stacks
- **[Contributing Guide](../../docs/CONTRIBUTING_TO_STACKS.md)** - Contribution guidelines

### Tools
- **[Sync Scripts](../../scripts/)** - Template synchronization tools
- **[Trend Detection](../../tools/trending-flagger/)** - Automated template discovery
- **[GitHub Actions](../../.github/workflows/)** - Automated workflows

## 📞 Support

### Getting Help
- **Documentation**: Check template-specific README files
- **Issues**: Create GitHub issues for problems
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our community channels

### Reporting Issues
When reporting issues:
1. **Check existing issues**: Search for similar problems
2. **Provide details**: Include error messages and steps to reproduce
3. **Include environment**: OS, runtime versions, etc.
4. **Use templates**: Follow the issue template format

---

**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
**Maintainer**: {stack_info.get('maintainer', 'Template Team')}
**Version**: 1.0

*This README is automatically updated when templates are added or modified.*
"""

    def _generate_templates_md(self, stack_name: str) -> Optional[str]:
        """Generate TEMPLATES.md content."""
        templates = self._get_stack_templates(stack_name)
        if not templates:
            return None

        content = f"""# {stack_name.title()} Stack Templates

Complete list of available templates in the {stack_name} stack.

## 📦 Templates

"""

        for template in templates:
            content += self._generate_detailed_template_entry(template)

        content += f"""
## 🤝 Contributing

To add new templates to this stack:

1. Use the automated population system:
   ```bash
   templateheaven populate discover {stack_name}
   templateheaven populate run --stack {stack_name}
   ```

2. Or manually add templates following the [contribution guidelines](../../docs/CONTRIBUTING_TO_STACKS.md)

## 📊 Template Statistics

- **Total Templates**: {len(templates)}
- **Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
- **Stack**: {stack_name}

---

*This file is automatically generated. Manual edits will be overwritten.*
"""

        return content

    # Helper methods for content generation

    def _get_stack_info(self, stack_name: str) -> Dict[str, Any]:
        """Get stack information."""
        stack_info_map = {
            "frontend": {
                "display_name": "Frontend",
                "maintainer": "Frontend Team",
                "category": "core"
            },
            "backend": {
                "display_name": "Backend",
                "maintainer": "Backend Team",
                "category": "core"
            },
            "ai-ml": {
                "display_name": "AI/ML",
                "maintainer": "AI/ML Team",
                "category": "ai-ml"
            },
            "devops": {
                "display_name": "DevOps",
                "maintainer": "DevOps Team",
                "category": "infrastructure"
            },
            "mobile": {
                "display_name": "Mobile",
                "maintainer": "Mobile Team",
                "category": "core"
            },
            "fullstack": {
                "display_name": "Fullstack",
                "maintainer": "Fullstack Team",
                "category": "core"
            },
            # Add more stack info as needed
        }

        return stack_info_map.get(stack_name, {
            "display_name": stack_name.replace("-", " ").title(),
            "maintainer": "Template Team",
            "category": "specialized"
        })

    def _get_stack_templates(self, stack_name: str) -> List[Dict[str, Any]]:
        """Get templates for a stack."""
        # For now, return empty list as templates haven't been populated yet
        # In a real implementation, this would query the template manager
        return []

    def _get_stack_overview_description(self, stack_name: str) -> str:
        """Get stack overview description."""
        descriptions = {
            "frontend": "The Frontend Stack provides comprehensive templates for building modern frontend applications using the latest frameworks, tools, and best practices. These templates are designed to get you up and running quickly with optimized development workflows.",
            "backend": "The Backend Stack provides robust templates for building scalable backend services and APIs using modern frameworks and architectures. Focus on performance, security, and maintainability.",
            "ai-ml": "The AI/ML Stack provides templates for machine learning workflows, data science projects, and AI applications using popular frameworks like TensorFlow, PyTorch, and scikit-learn.",
            "devops": "The DevOps Stack provides templates for infrastructure automation, CI/CD pipelines, containerization, and cloud deployment using tools like Docker, Kubernetes, and Terraform.",
        }

        return descriptions.get(stack_name, f"The {stack_name.title()} Stack provides specialized templates for {stack_name} development and deployment.")

    def _generate_templates_table(self, templates: List[Dict], stack_name: str) -> str:
        """Generate templates table for header."""
        if not templates:
            return "| _No templates available yet_ | | | | |"

        rows = []
        for template in templates[:5]:  # Show first 5 templates
            name = template.get("name", "")
            description = template.get("description", "")[:50] + "..." if len(template.get("description", "")) > 50 else template.get("description", "")
            technologies = ", ".join(template.get("technologies", [])[:3])
            stars = template.get("stars", 0)
            updated = template.get("updated_at", datetime.now().strftime("%Y-%m-%d"))

            rows.append(f"| [{name}](./{name}/) | {description} | {technologies} | {stars}k+ | {updated} |")

        if len(templates) > 5:
            rows.append(f"| ... and {len(templates) - 5} more | | | | |")

        return "\n".join(rows)

    def _get_technology_focus_description(self, stack_name: str) -> str:
        """Get technology focus description."""
        focuses = {
            "frontend": """
### Core Frameworks
- **React**: Modern frontend library with hooks and functional components
- **Vue**: Progressive JavaScript framework
- **Svelte**: Compile-time optimized framework
- **Angular**: Full-featured TypeScript framework

### Build Tools
- **Vite**: Next-generation frontend build tool
- **Webpack**: Module bundler and build tool
- **Rollup**: ES6 module bundler
- **esbuild**: Extremely fast JavaScript bundler

### Styling Solutions
- **Tailwind CSS**: Utility-first CSS framework
- **Styled Components**: CSS-in-JS styling
- **Emotion**: CSS-in-JS library
- **CSS Modules**: Scoped CSS modules
""",
            "backend": """
### API Frameworks
- **FastAPI**: Modern, fast web framework for building APIs
- **Express**: Minimal and flexible Node.js web application framework
- **Django**: High-level Python web framework
- **Spring Boot**: Java framework for building enterprise applications

### Languages
- **Python**: Versatile language for backend development
- **JavaScript/Node.js**: Runtime for server-side JavaScript
- **Go**: Efficient language for cloud services
- **Rust**: Memory-safe systems programming

### Databases
- **PostgreSQL**: Advanced open source relational database
- **MongoDB**: Document-oriented NoSQL database
- **Redis**: In-memory data structure store
- **MySQL**: Popular relational database
""",
            "ai-ml": """
### Machine Learning Frameworks
- **TensorFlow**: End-to-end ML platform
- **PyTorch**: Deep learning research platform
- **scikit-learn**: Machine learning in Python
- **XGBoost**: Gradient boosting framework

### Data Processing
- **pandas**: Data manipulation and analysis
- **NumPy**: Scientific computing with Python
- **Dask**: Parallel computing with Python
- **Apache Spark**: Unified analytics engine

### MLOps Tools
- **MLflow**: ML lifecycle management
- **Kubeflow**: ML on Kubernetes
- **DVC**: Data version control
- **Weights & Biases**: Experiment tracking
""",
            "devops": """
### Containerization
- **Docker**: Container platform
- **Podman**: Daemonless container engine
- **Kubernetes**: Container orchestration
- **Docker Compose**: Multi-container applications

### Infrastructure as Code
- **Terraform**: Infrastructure provisioning
- **Ansible**: Configuration management
- **Pulumi**: Infrastructure as code SDK
- **CloudFormation**: AWS infrastructure templates

### CI/CD
- **GitHub Actions**: Workflow automation
- **Jenkins**: Automation server
- **GitLab CI**: Integrated CI/CD
- **CircleCI**: Cloud-based CI/CD
"""
        }

        return focuses.get(stack_name, f"Specialized technologies for {stack_name} development and deployment.")

    def _get_use_cases_description(self, stack_name: str) -> str:
        """Get use cases description."""
        use_cases = {
            "frontend": """
### Perfect For
- **Single Page Applications**: SPAs with client-side routing
- **Progressive Web Apps**: PWAs with offline capabilities
- **Component Libraries**: Reusable UI component libraries
- **Design Systems**: Comprehensive design system implementations
- **Static Sites**: Static site generation and deployment
- **Interactive Dashboards**: Data visualization and analytics
- **E-commerce Frontends**: Online store interfaces
- **Content Management**: CMS frontend interfaces
""",
            "backend": """
### Perfect For
- **REST APIs**: Scalable web service APIs
- **GraphQL Services**: Flexible API development
- **Microservices**: Distributed system architecture
- **Serverless Functions**: Event-driven computing
- **Data Processing**: ETL and analytics pipelines
- **Real-time Applications**: WebSocket and streaming services
- **IoT Backends**: Connected device management
- **Enterprise Applications**: Large-scale business systems
""",
            "ai-ml": """
### Perfect For
- **Predictive Analytics**: Forecasting and trend analysis
- **Computer Vision**: Image recognition and processing
- **Natural Language Processing**: Text analysis and generation
- **Recommendation Systems**: Personalized content delivery
- **Anomaly Detection**: Fraud detection and monitoring
- **Automated Decision Making**: Intelligent business processes
- **Research Projects**: Academic and scientific computing
- **Production ML Systems**: Scalable model deployment
"""
        }

        return use_cases.get(stack_name, f"Specialized applications and use cases for {stack_name} development.")

    def _get_installation_commands(self, stack_name: str) -> str:
        """Get installation commands for a stack."""
        commands = {
            "frontend": """# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview""",
            "backend": """# Install dependencies
pip install -r requirements.txt
# or
npm install

# Start development server
python main.py
# or
npm run dev

# Run tests
pytest
# or
npm test""",
            "ai-ml": """# Install dependencies
pip install -r requirements.txt

# Start Jupyter notebook (if applicable)
jupyter notebook

# Run training script
python train.py

# Run inference
python predict.py""",
            "devops": """# Install dependencies
pip install -r requirements.txt

# Initialize infrastructure
terraform init

# Plan deployment
terraform plan

# Apply changes
terraform apply"""
        }

        return commands.get(stack_name, """# Install dependencies and run
# (See template README for specific commands)""")

    def _get_environment_setup_description(self, stack_name: str) -> str:
        """Get environment setup description."""
        setups = {
            "frontend": """
Most frontend templates require minimal environment setup:

```bash
# Development server
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run test         # Run tests
npm run lint         # Run linter
npm run format       # Format code
```""",
            "backend": """
Backend templates typically require:

```bash
# Python environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt

# Node.js environment
npm install

# Database setup (if required)
# See template README for database configuration
```""",
            "ai-ml": """
AI/ML templates typically require:

```bash
# Python environment with ML libraries
conda create -n ml-env python=3.9
conda activate ml-env
pip install -r requirements.txt

# GPU support (optional)
# Install CUDA and cuDNN for GPU acceleration

# Jupyter environment
pip install jupyter
jupyter notebook
```"""
        }

        return setups.get(stack_name, f"See individual template documentation for environment setup requirements specific to {stack_name}.")

    def _get_development_tools_description(self, stack_name: str) -> str:
        """Get development tools description."""
        tools = {
            "frontend": """
- **ESLint & Prettier**: Code linting and formatting
- **TypeScript**: Type-safe JavaScript development
- **Vitest**: Fast unit testing framework
- **Cypress**: End-to-end testing
- **Storybook**: Component development environment
- **Parcel**: Zero-configuration application bundler
- **Snowpack**: Lightning-fast frontend build tool""",
            "backend": """
- **pytest**: Testing framework for Python
- **Jest**: Testing framework for Node.js
- **Postman**: API testing and documentation
- **Swagger/OpenAPI**: API documentation
- **Docker**: Containerization platform
- **PostgreSQL/MySQL**: Database systems
- **Redis**: Caching and session storage""",
            "ai-ml": """
- **Jupyter**: Interactive computing environment
- **MLflow**: Experiment tracking and model management
- **DVC**: Data version control
- **Weights & Biases**: ML experiment tracking
- **TensorBoard**: Visualization toolkit for TensorFlow
- **Kubeflow**: ML pipelines on Kubernetes
- **Ray**: Distributed computing framework"""
        }

        return tools.get(stack_name, f"Specialized development tools for {stack_name} workflows.")

    def _get_testing_description(self, stack_name: str) -> str:
        """Get testing description."""
        tests = {
            "frontend": """- **Unit Tests**: `npm run test`
- **Component Tests**: `npm run test:components`
- **E2E Tests**: `npm run test:e2e`
- **Visual Tests**: `npm run test:visual`
- **Type Checking**: `npm run type-check`""",
            "backend": """- **Unit Tests**: `pytest tests/unit/` or `npm test`
- **Integration Tests**: `pytest tests/integration/`
- **API Tests**: Using Postman or automated test suites
- **Load Tests**: Using tools like Locust or Artillery
- **Security Tests**: Automated vulnerability scanning"""
        }

        return tests.get(stack_name, f"- **Unit Tests**: Run test suites specific to {stack_name}\n- **Integration Tests**: Verify component interactions\n- **Quality Assurance**: Automated testing pipelines")

    def _get_code_quality_description(self, stack_name: str) -> str:
        """Get code quality description."""
        quality = {
            "frontend": """- **Linting**: `npm run lint`
- **Formatting**: `npm run format`
- **Type Checking**: `npm run type-check`
- **Bundle Analysis**: `npm run analyze`
- **Accessibility**: Automated a11y testing""",
            "backend": """- **Linting**: `flake8` or `eslint`
- **Formatting**: `black` or `prettier`
- **Type Checking**: `mypy` or `TypeScript`
- **Security Scanning**: `bandit` or `snyk`
- **Performance Analysis**: Code profiling tools"""
        }

        return quality.get(stack_name, f"- **Linting**: Automated code quality checks\n- **Formatting**: Consistent code formatting\n- **Security**: Vulnerability scanning\n- **Performance**: Code analysis and optimization")

    def _get_building_description(self, stack_name: str) -> str:
        """Get building description."""
        builds = {
            "frontend": """- **Development Build**: `npm run build:dev`
- **Production Build**: `npm run build`
- **Static Export**: `npm run export`
- **Bundle Analysis**: `npm run analyze`
- **Preview Build**: `npm run preview`""",
            "backend": """- **Development Build**: Local development server
- **Production Build**: Optimized build with `python -m py_compile` or build tools
- **Container Build**: `docker build`
- **Deployment Package**: `pip wheel` or deployment scripts
- **Documentation Build**: `sphinx-build` or similar"""
        }

        return builds.get(stack_name, f"- **Development Build**: Local development setup\n- **Production Build**: Optimized production builds\n- **Deployment Ready**: Prepared for deployment pipelines")

    def _get_deployment_platforms(self, stack_name: str) -> str:
        """Get deployment platforms."""
        platforms = {
            "frontend": """- **Vercel**: Recommended for React and Next.js applications
- **Netlify**: Great for static sites and JAMstack
- **GitHub Pages**: Free hosting for static sites
- **AWS S3**: Scalable static site hosting
- **Cloudflare Pages**: Global edge deployment
- **Docker**: Containerized deployment""",
            "backend": """- **Heroku**: Platform as a Service
- **AWS Elastic Beanstalk**: Managed application platform
- **Google App Engine**: Serverless application platform
- **Azure App Service**: Managed web apps
- **DigitalOcean App Platform**: Cloud application platform
- **Docker/Kubernetes**: Container orchestration""",
            "ai-ml": """- **Google AI Platform**: Managed ML platform
- **AWS SageMaker**: ML model building and deployment
- **Azure Machine Learning**: Cloud ML platform
- **Hugging Face Spaces**: ML model hosting
- **Streamlit Cloud**: Data app deployment
- **Gradio**: ML model interfaces"""
        }

        return platforms.get(stack_name, f"- **Cloud Platforms**: Major cloud provider services\n- **Container Platforms**: Docker and Kubernetes\n- **Specialized Hosting**: Platform-specific deployment options")

    def _get_performance_optimization_description(self, stack_name: str) -> str:
        """Get performance optimization description."""
        optimizations = {
            "frontend": """- **Code Splitting**: Automatic code splitting for optimal loading
- **Lazy Loading**: Load components and routes on demand
- **Image Optimization**: Optimized image loading and formats
- **Caching**: Aggressive caching strategies
- **Compression**: Gzip and Brotli compression
- **CDN**: Content delivery network integration""",
            "backend": """- **Database Optimization**: Query optimization and indexing
- **Caching**: Redis and in-memory caching
- **Load Balancing**: Request distribution and scaling
- **Compression**: Response compression and optimization
- **Connection Pooling**: Database and external service connections
- **Async Processing**: Non-blocking operations and queuing"""
        }

        return optimizations.get(stack_name, f"- **Performance Monitoring**: Application performance tracking\n- **Optimization Tools**: Specialized performance analysis\n- **Scaling Strategies**: Horizontal and vertical scaling approaches")

    def _get_template_requirements(self, stack_name: str) -> str:
        """Get template requirements."""
        requirements = {
            "frontend": """- **TypeScript Support**: Full TypeScript configuration
- **Testing Setup**: Unit and integration tests
- **Documentation**: Comprehensive README and docs
- **Performance**: Optimized bundle size and loading
- **Accessibility**: WCAG compliance
- **Responsive Design**: Mobile-first approach
- **SEO**: Search engine optimization
- **Build Optimization**: Fast build times""",
            "backend": """- **API Documentation**: OpenAPI/Swagger specs
- **Testing**: Unit and integration test suites
- **Security**: Input validation and authentication
- **Performance**: Optimized queries and responses
- **Error Handling**: Comprehensive error management
- **Logging**: Structured logging implementation
- **Database**: Proper schema and migrations"""
        }

        return requirements.get(stack_name, f"- **Quality Standards**: Meet {stack_name} best practices\n- **Documentation**: Comprehensive setup and usage guides\n- **Testing**: Automated test suites included\n- **Performance**: Optimized for production use")

    def _get_quality_standards_description(self, stack_name: str) -> str:
        """Get quality standards description."""
        standards = {
            "frontend": """- **Minimum stars**: 300
- **Minimum forks**: 30
- **Growth rate**: >8% weekly
- **Documentation score**: >8/10
- **Performance score**: >7/10
- **Accessibility score**: >8/10""",
            "backend": """- **Minimum stars**: 200
- **Minimum forks**: 20
- **Growth rate**: >5% weekly
- **Documentation score**: >8/10
- **Security score**: >7/10
- **Performance score**: >7/10"""
        }

        return standards.get(stack_name, f"- **Community Engagement**: Minimum stars and forks requirements\n- **Growth Rate**: Sustained development activity\n- **Documentation**: Comprehensive setup and usage guides\n- **Quality Metrics**: Automated quality scoring")

    def _generate_template_entry(self, template: Dict[str, Any]) -> str:
        """Generate a template entry for the templates section."""
        name = template.get("name", "")
        description = template.get("description", "")
        technologies = template.get("technologies", [])
        stars = template.get("stars", 0)

        return f"""### {name}

{description}

**Technologies**: {", ".join(technologies[:5])}
**Stars**: {stars}
**Path**: [`stacks/{template.get('stack', '')}/{name}/`](./{name}/)

---

"""

    def _generate_detailed_template_entry(self, template: Dict[str, Any]) -> str:
        """Generate a detailed template entry for TEMPLATES.md."""
        name = template.get("name", "")
        description = template.get("description", "")
        technologies = template.get("technologies", [])
        stars = template.get("stars", 0)
        url = template.get("upstream_url", "")

        return f"""## {name}

{description}

### Specifications

- **Stars**: {stars}
- **Technologies**: {", ".join(technologies)}
- **Upstream**: [{url}]({url})
- **Last Updated**: {template.get('updated_at', 'Unknown')}

### Quick Start

```bash
cp -r stacks/{template.get('stack', '')}/{name} ../my-project
cd ../my-project
# Follow README.md instructions
```

---

"""
