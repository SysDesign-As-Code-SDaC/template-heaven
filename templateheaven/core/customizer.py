"""
Template Customizer for Template Heaven.

This module provides template customization functionality using Jinja2
for variable substitution and file processing.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from jinja2 import Environment, FileSystemLoader, Template, TemplateError

from .models import Template, ProjectConfig
from ..utils.logger import get_logger
from ..utils.file_ops import FileOperations
from ..utils.helpers import sanitize_project_name, validate_project_name

logger = get_logger(__name__)


class Customizer:
    """
    Handles template customization and file generation.
    
    MVP features:
    - Copy template files to destination
    - Replace placeholders using Jinja2 templating
    - Update package.json/pyproject.toml with project info
    - Basic file filtering and processing
    - Safe file operations with validation
    
    Attributes:
        file_ops: File operations utility
        jinja_env: Jinja2 environment for templating
    """
    
    def __init__(self, config=None):
        """
        Initialize the customizer.

        Args:
            config: Configuration instance (optional)
        """
        self.config = config
        self.file_ops = FileOperations()
        self.jinja_env = Environment(
            loader=FileSystemLoader('.'),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Add custom filters
        self.jinja_env.filters['snake_case'] = self._snake_case_filter
        self.jinja_env.filters['kebab_case'] = self._kebab_case_filter
        self.jinja_env.filters['pascal_case'] = self._pascal_case_filter
        self.jinja_env.filters['camel_case'] = self._camel_case_filter
        
        logger.debug("Customizer initialized")
    
    def _copy_template_files(
        self,
        project_path: Path,
        template: Template,
        variables: Dict[str, Any]
    ) -> None:
        """
        Copy template files from templates directory to project directory.
        
        Args:
            project_path: Destination project directory
            template: Template to copy
            variables: Template variables for substitution
        """
        # Get template source directory
        template_source = Path(__file__).parent.parent.parent / "templates" / template.name
        
        if not template_source.exists():
            logger.warning(f"Template source not found: {template_source}")
            # Fall back to basic structure
            self._create_basic_project_structure(project_path, template, variables)
            return
        
        logger.info(f"Copying template files from: {template_source}")
        
        # Copy all files and directories
        self._copy_directory_recursive(template_source, project_path, variables)
        
        logger.info(f"Successfully copied template files to: {project_path}")
    
    def _copy_directory_recursive(
        self,
        source: Path,
        destination: Path,
        variables: Dict[str, Any]
    ) -> None:
        """
        Recursively copy directory structure with Jinja2 templating.
        
        Args:
            source: Source directory
            destination: Destination directory
            variables: Template variables
        """
        for item in source.iterdir():
            dest_item = destination / item.name
            
            if item.is_dir():
                # Skip certain directories
                if item.name in ['.git', '__pycache__', '.pytest_cache', 'node_modules']:
                    continue
                
                # Create directory and recurse
                self.file_ops.create_directory(dest_item)
                self._copy_directory_recursive(item, dest_item, variables)
            
            elif item.is_file():
                # Skip certain files
                if item.name in ['.DS_Store', 'Thumbs.db']:
                    continue
                
                # Process file with Jinja2 templating
                self._process_template_file(item, dest_item, variables)
    
    def _process_template_file(
        self,
        source_file: Path,
        dest_file: Path,
        variables: Dict[str, Any]
    ) -> None:
        """
        Process a single template file with Jinja2 templating.
        
        Args:
            source_file: Source file path
            dest_file: Destination file path
            variables: Template variables
        """
        try:
            # Read source file
            content = source_file.read_text(encoding='utf-8')
            
            # Check if file contains Jinja2 template syntax and is not a YAML file
            # YAML files with GitHub Actions syntax (${{ }}) should not be processed as Jinja2
            is_yaml_file = source_file.suffix.lower() in ['.yml', '.yaml']
            contains_jinja_syntax = ('{{' in content or '{%' in content) and not is_yaml_file

            if contains_jinja_syntax:
                # Process with Jinja2
                template = self.jinja_env.from_string(content)
                processed_content = template.render(**variables)
            else:
                # Simple variable substitution for non-Jinja2 files
                processed_content = content
                for key, value in variables.items():
                    processed_content = processed_content.replace(f'{{{{ {key} }}}}', str(value))
                    processed_content = processed_content.replace(f'{{{key}}}', str(value))
            
            # Write processed content to destination
            self.file_ops.write_file(dest_file, processed_content)
            
        except Exception as e:
            logger.error(f"Error processing template file {source_file}: {e}")
            # Fall back to simple copy
            self.file_ops.copy_file(source_file, dest_file)
    
    def customize(
        self,
        template: Template,
        config: ProjectConfig,
        output_dir: Path
    ) -> bool:
        """
        Customize and initialize project from template.
        
        Args:
            template: Template to use
            config: Project configuration
            output_dir: Destination directory
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If output directory exists or is invalid
            OSError: If file operations fail
            TemplateError: If Jinja2 templating fails
        """
        logger.info(f"Customizing template '{template.name}' to '{output_dir}'")
        
        # Validate inputs
        self._validate_inputs(template, config, output_dir)
        
        # Get template variables
        variables = config.get_template_variables()
        
        # Create project directory
        project_path = output_dir / config.name
        if project_path.exists():
            raise ValueError(f"Project directory already exists: {project_path}")
        
        try:
            # Create project directory
            self.file_ops.create_directory(project_path)
            
            # Copy actual template files from templates directory
            self._copy_template_files(project_path, template, variables)
            
            # Update package files
            self._update_package_files(project_path, template, variables)
            
            # Create README
            self._create_readme(project_path, template, variables)
            # Create LICENSE if missing
            self._create_license(project_path, template, variables)
            # Create CONTRIBUTING.md if missing
            self._create_contributing(project_path, template, variables)
            
            # Create .gitignore if needed
            self._create_gitignore(project_path, template, variables)
            
            logger.info(f"Successfully created project: {project_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to customize template: {e}")
            # Clean up on failure
            if project_path.exists():
                self.file_ops.remove_directory(project_path)
            raise

    def customize_from_repo_dir(
        self,
        source_dir: Path,
        config: ProjectConfig,
        output_dir: Path
    ) -> bool:
        """
        Customize and initialize project from a source directory (e.g., a cloned repo).

        This method is similar to `customize` but uses an explicit source directory
        rather than a template identifier inside the `templates/` directory.
        """
        logger.info(f"Customizing from repo directory '{source_dir}' to '{output_dir}'")

        # Validate inputs (reuse existing validation)
        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        # We re-create a minimal Template object for compatibility
        # There may not be all metadata available; we will attempt to read a template manifest.
        try:
            # Create project_dir and copy files
            project_path = output_dir / config.name
            if project_path.exists():
                raise ValueError(f"Project directory already exists: {project_path}")

            self.file_ops.create_directory(project_path)
            self._copy_directory_recursive(source_dir, project_path, config.get_template_variables())

            # Update package files, README, gitignore as in normal flow
            template_meta = config.template if hasattr(config, 'template') and config.template else None
            self._update_package_files(project_path, template_meta, config.get_template_variables())
            self._create_readme(project_path, template_meta, config.get_template_variables())
            self._create_license(project_path, template_meta, config.get_template_variables())
            self._create_contributing(project_path, template_meta, config.get_template_variables())
            self._create_gitignore(project_path, template_meta, config.get_template_variables())

            logger.info(f"Successfully created project from repo: {project_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to customize from repo: {e}")
            # Clean up on failure
            project_path = output_dir / config.name
            if project_path.exists():
                self.file_ops.remove_directory(project_path)
            raise
    
    def _validate_inputs(
        self,
        template: Template,
        config: ProjectConfig,
        output_dir: Path
    ) -> None:
        """
        Validate input parameters.
        
        Args:
            template: Template to validate
            config: Project configuration to validate
            output_dir: Output directory to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not template:
            raise ValueError("Template cannot be None")
        
        if not config:
            raise ValueError("Project configuration cannot be None")
        
        if not output_dir:
            raise ValueError("Output directory cannot be None")
        
        # Validate project name
        try:
            validate_project_name(config.name)
        except ValueError as e:
            raise ValueError(f"Invalid project name: {e}")
        
        # Check if output directory is writable
        if not output_dir.parent.exists():
            raise ValueError(f"Parent directory does not exist: {output_dir.parent}")
        
        if not os.access(output_dir.parent, os.W_OK):
            raise ValueError(f"Parent directory is not writable: {output_dir.parent}")
    
    def _create_basic_project_structure(
        self,
        project_path: Path,
        template: Template,
        variables: Dict[str, Any]
    ) -> None:
        """
        Create basic project structure for MVP.
        
        Args:
            project_path: Project directory path
            template: Template being used
            variables: Template variables
        """
        # Create basic directories based on template stack
        if template.stack.value in ['frontend', 'fullstack']:
            self.file_ops.create_directory(project_path / 'src')
            self.file_ops.create_directory(project_path / 'public')
        
        if template.stack.value in ['backend', 'fullstack']:
            self.file_ops.create_directory(project_path / 'src')
            self.file_ops.create_directory(project_path / 'tests')
        
        if template.stack.value == 'ai-ml':
            self.file_ops.create_directory(project_path / 'data')
            self.file_ops.create_directory(project_path / 'notebooks')
            self.file_ops.create_directory(project_path / 'src')
        
        # Create common directories
        self.file_ops.create_directory(project_path / 'docs')
        
        logger.debug(f"Created basic project structure for {template.stack.value}")
    
    def _update_package_files(
        self,
        project_path: Path,
        template: Optional[Template],
        variables: Dict[str, Any]
    ) -> None:
        """
        Update package configuration files.
        
        Args:
            project_path: Project directory path
            template: Template being used
            variables: Template variables
        """
        # Create package.json for Node.js projects
        template_tags = template.tags if template else []
        if any(tag in template_tags for tag in ['nodejs', 'react', 'vue', 'nextjs', 'typescript']):
            self._create_package_json(project_path, template, variables)

        # Create pyproject.toml for Python projects
        if any(tag in template_tags for tag in ['python', 'fastapi', 'django', 'pytorch']):
            self._create_pyproject_toml(project_path, template, variables)

        # Create requirements.txt for Python projects
        if any(tag in template_tags for tag in ['python', 'fastapi', 'django', 'pytorch']):
            self._create_requirements_txt(project_path, template, variables)
    
    def _create_package_json(
        self,
        project_path: Path,
        template: Template,
        variables: Dict[str, Any]
    ) -> None:
        """Create package.json file."""
        package_data = {
            "name": variables['project_name'],
            "version": variables['version'],
            "description": variables['project_description'],
            "author": variables['author'],
            "license": variables['license'],
            "main": "src/index.js",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview",
                "test": "jest"
            },
            "dependencies": template.dependencies,
            "devDependencies": {
                "typescript": "^5.0.0",
                "vite": "^4.4.0",
                "@types/node": "^20.0.0"
            }
        }
        
        # Add template-specific dependencies
        if 'react' in template.tags:
            package_data['dependencies']['react'] = '^18.2.0'
            package_data['dependencies']['react-dom'] = '^18.2.0'
            package_data['devDependencies']['@types/react'] = '^18.2.0'
            package_data['devDependencies']['@types/react-dom'] = '^18.2.0'
        
        if 'vue' in template.tags:
            package_data['dependencies']['vue'] = '^3.3.0'
            package_data['devDependencies']['@vitejs/plugin-vue'] = '^4.2.0'
        
        import json
        package_json_path = project_path / 'package.json'
        self.file_ops.write_file(
            package_json_path,
            json.dumps(package_data, indent=2)
        )
        
        logger.debug("Created package.json")
    
    def _create_pyproject_toml(
        self,
        project_path: Path,
        template: Template,
        variables: Dict[str, Any]
    ) -> None:
        """Create pyproject.toml file."""
        pyproject_content = f"""[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{variables['project_name']}"
version = "{variables['version']}"
description = "{variables['project_description']}"
authors = [
    {{name = "{variables['author']}", email = "user@example.com"}},
]
license = {{text = "{variables['license']}"}}
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
"""
        
        # Add template dependencies
        for dep, version in template.dependencies.items():
            pyproject_content += f'    "{dep}{version}",\n'
        
        pyproject_content += """]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "mypy>=1.0",
    "flake8>=6.0",
]

[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
disallow_untyped_defs = true
"""
        
        pyproject_path = project_path / 'pyproject.toml'
        self.file_ops.write_file(pyproject_path, pyproject_content)
        
        logger.debug("Created pyproject.toml")
    
    def _create_requirements_txt(
        self,
        project_path: Path,
        template: Template,
        variables: Dict[str, Any]
    ) -> None:
        """Create requirements.txt file."""
        requirements_content = "# Core dependencies\n"
        
        for dep, version in template.dependencies.items():
            requirements_content += f"{dep}{version}\n"
        
        requirements_content += """
# Development dependencies
pytest>=7.0
black>=23.0
mypy>=1.0
flake8>=6.0
"""
        
        requirements_path = project_path / 'requirements.txt'
        self.file_ops.write_file(requirements_path, requirements_content)
        
        logger.debug("Created requirements.txt")
    
    def _create_readme(
        self,
        project_path: Path,
        template: Template,
        variables: Dict[str, Any]
    ) -> None:
        """Create README.md file."""
        readme_content = f"""# {variables['project_name']}

{variables['project_description']}

## 🚀 Features

"""
        
        # Add template features
        for feature in template.features:
            readme_content += f"- {feature}\n"
        
        readme_content += f"""
## 📋 Prerequisites

- Python 3.8+ (if applicable)
- Node.js 16+ (if applicable)

## 🛠️ Installation

"""
        
        # Add installation instructions based on template
        if any(tag in template.tags for tag in ['python', 'fastapi', 'django']):
            readme_content += """```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```
"""
        
        if any(tag in template.tags for tag in ['nodejs', 'react', 'vue', 'typescript']):
            readme_content += """```bash
# Install dependencies
npm install
```
"""
        
        readme_content += f"""
## 🚀 Usage

```bash
# Development
npm run dev  # or python app.py
```

## 📚 Documentation

See the [docs/](docs/) directory for detailed documentation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the {variables['license']} License.

## 🔗 Upstream Source

- **Original Template**: {template.name}
- **Upstream URL**: {template.upstream_url or 'N/A'}
- **Template Version**: {template.version or 'N/A'}
"""
        
        readme_path = project_path / 'README.md'
        self.file_ops.write_file(readme_path, readme_content)
        
        logger.debug("Created README.md")
    
    def _create_gitignore(
        self,
        project_path: Path,
        template: Template,
        variables: Dict[str, Any]
    ) -> None:
        """Create .gitignore file."""
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn-integrity

# Build outputs
dist/
build/
.next/
.nuxt/
.vuepress/dist

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
logs
*.log

# Database
*.db
*.sqlite
*.sqlite3

# Temporary files
*.tmp
*.temp
"""
        
        gitignore_path = project_path / '.gitignore'
        self.file_ops.write_file(gitignore_path, gitignore_content)
        
        logger.debug("Created .gitignore")

    def _create_license(
        self,
        project_path: Path,
        template: Optional[Template],
        variables: Dict[str, Any]
    ) -> None:
        """Create a LICENSE file in the project.

        Uses template LICENSE if provided, otherwise writes a minimal MIT license.
        """
        license_path = project_path / 'LICENSE'
        # If LICENSE already exists, do nothing
        if license_path.exists():
            logger.debug("LICENSE already exists; skipping")
            return

        # Check if template contains a LICENSE
        template_source = Path(__file__).parent.parent.parent / 'templates' / (template.name if template else '')
        template_license = template_source / 'LICENSE'
        if template_license.exists():
            try:
                self.file_ops.copy_file(template_license, license_path)
                logger.debug("Copied template LICENSE into project")
                return
            except Exception:
                logger.debug("Failed to copy template LICENSE; falling back to default")

        # Default MIT license
        year = variables.get('current_year', '2024')
        author = variables.get('author', 'Template Heaven')
        mit_license = f"MIT License\n\nCopyright (c) {year} {author}\n\nPermission is hereby granted, free of charge, to any person obtaining a copy..."
        self.file_ops.write_file(license_path, mit_license)
        logger.debug("Created default MIT LICENSE")

    def _create_contributing(
        self,
        project_path: Path,
        template: Optional[Template],
        variables: Dict[str, Any]
    ) -> None:
        """Create a CONTRIBUTING.md file in the project.

        Uses template CONTRIBUTING.md if provided, otherwise writes a small CONTRIBUTING template.
        """
        contributing_path = project_path / 'CONTRIBUTING.md'
        if contributing_path.exists():
            logger.debug("CONTRIBUTING.md already exists; skipping")
            return

        # Check for template CONTRIBUTING.md
        template_source = Path(__file__).parent.parent.parent / 'templates' / (template.name if template else '')
        template_contrib = template_source / 'CONTRIBUTING.md'
        if template_contrib.exists():
            try:
                self.file_ops.copy_file(template_contrib, contributing_path)
                logger.debug("Copied template CONTRIBUTING.md into project")
                return
            except Exception:
                logger.debug("Failed to copy template CONTRIBUTING.md; falling back to default")

        contributing_text = (
            "# Contributing\n\n" 
            "Thank you for contributing to this project!\n\n" 
            "Please read the following guidelines before opening a pull request:\n\n" 
            "1. Fork the repo and create a feature branch.\n"
            "2. Add tests and update documentation.\n"
            "3. Run the test suite and linters.\n"
        )

        self.file_ops.write_file(contributing_path, contributing_text)
        logger.debug("Created default CONTRIBUTING.md")
    
    def _snake_case_filter(self, text: str) -> str:
        """Convert text to snake_case."""
        # Insert underscores between camelCase or PascalCase boundaries then normalize
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return re.sub(r'[^a-zA-Z0-9]+', '_', s2).lower().strip('_')
    
    def _kebab_case_filter(self, text: str) -> str:
        """Convert text to kebab-case."""
        # Insert hyphens between camelCase or PascalCase boundaries then normalize
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', text)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1-\2', s1)
        return re.sub(r'[^a-zA-Z0-9]+', '-', s2).lower().strip('-')
    
    def _pascal_case_filter(self, text: str) -> str:
        """Convert text to PascalCase."""
        words = re.findall(r'[a-zA-Z0-9]+', text)
        return ''.join(word.capitalize() for word in words)
    
    def _camel_case_filter(self, text: str) -> str:
        """Convert text to camelCase."""
        words = re.findall(r'[a-zA-Z0-9]+', text)
        if not words:
            return text
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    def process_template_file(
        self,
        content: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Process template content with Jinja2.
        
        Args:
            content: Template content
            variables: Template variables
            
        Returns:
            Processed content
            
        Raises:
            TemplateError: If templating fails
        """
        try:
            template = self.jinja_env.from_string(content)
            return template.render(**variables)
        except TemplateError as e:
            logger.error(f"Template processing failed: {e}")
            raise
    
    def get_template_variables(self, config: ProjectConfig) -> Dict[str, Any]:
        """
        Get all template variables for a project configuration.
        
        Args:
            config: Project configuration
            
        Returns:
            Dictionary of template variables
        """
        variables = config.get_template_variables()
        
        # Add additional computed variables
        variables.update({
            'project_name_snake': self._snake_case_filter(config.name),
            'project_name_kebab': self._kebab_case_filter(config.name),
            'project_name_pascal': self._pascal_case_filter(config.name),
            'project_name_camel': self._camel_case_filter(config.name),
            'current_year': str(2024),  # Could be dynamic
        })
        
        return variables

    # Backwards-compatible alias used in tests
    def _get_template_variables(self, config: ProjectConfig) -> Dict[str, Any]:
        return self.get_template_variables(config)
