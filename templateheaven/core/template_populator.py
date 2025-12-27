"""
Template Population System for Template Heaven.

This module provides automated template discovery, validation, and population
for all stack branches using GitHub API integration and stack-specific validation.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import tempfile

from .github_search import GitHubSearchService
from .stack_config import get_stack_config_manager
from .template_manager import TemplateManager
from .customizer import Customizer
from ..utils.logger import get_logger
from ..utils.cache import Cache
from ..config.settings import Config

logger = get_logger(__name__)


class TemplatePopulator:
    """
    Automated template population system for stack branches.

    Discovers high-quality templates from GitHub, validates them against
    stack-specific requirements, and populates stack branches with
    properly formatted templates and documentation.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the template populator.

        Args:
            config: Configuration instance
        """
        self.config = config or Config()
        self.github_search = GitHubSearchService(self.config)
        self.stack_config = get_stack_config_manager()
        self.template_manager = TemplateManager(self.config)
        self.cache = Cache(
            cache_dir=self.config.get_metadata_cache_dir(),
            default_ttl=3600
        )
        self.customizer = Customizer(self.config)

        # Template population settings
        self.templates_per_stack = 5  # Target templates per stack
        self.min_template_quality = 0.7  # Minimum quality score
        self.max_concurrent_downloads = 3  # Limit concurrent operations

    async def populate_all_stacks(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Populate all stack branches with templates.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            Population results summary
        """
        logger.info("Starting template population for all stacks")

        results = {
            "total_stacks": 0,
            "successful_populations": 0,
            "failed_populations": 0,
            "templates_added": 0,
            "errors": [],
            "stack_results": {}
        }

        # Get all stack configurations
        all_stacks = self.stack_config.get_all_stacks()
        results["total_stacks"] = len(all_stacks)

        for stack_name, stack_config in all_stacks.items():
            try:
                logger.info(f"Populating stack: {stack_name}")
                stack_result = await self.populate_stack(stack_name, force_refresh)
                results["stack_results"][stack_name] = stack_result

                if stack_result["success"]:
                    results["successful_populations"] += 1
                    results["templates_added"] += stack_result["templates_added"]
                else:
                    results["failed_populations"] += 1
                    results["errors"].extend(stack_result["errors"])

            except Exception as e:
                logger.error(f"Failed to populate stack {stack_name}: {e}")
                results["failed_populations"] += 1
                results["errors"].append(f"{stack_name}: {str(e)}")
                results["stack_results"][stack_name] = {
                    "success": False,
                    "error": str(e),
                    "templates_added": 0,
                    "errors": [str(e)]
                }

        logger.info(f"Template population completed: {results['successful_populations']}/{results['total_stacks']} stacks successful")
        return results

    async def populate_stack(self, stack_name: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Populate a specific stack branch with templates.

        Args:
            stack_name: Name of the stack to populate
            force_refresh: Force refresh of cached data

        Returns:
            Population results for this stack
        """
        result = {
            "success": False,
            "templates_added": 0,
            "templates_skipped": 0,
            "errors": [],
            "templates": []
        }

        try:
            # Get stack configuration
            stack_config = self.stack_config.get_stack_config(stack_name)
            if not stack_config:
                raise ValueError(f"Stack configuration not found: {stack_name}")

            # Discover template candidates
            logger.info(f"Discovering templates for stack: {stack_name}")
            candidates = await self.github_search.discover_templates_for_stack(
                stack_name, limit=self.templates_per_stack * 2  # Get more candidates for selection
            )

            if not candidates:
                logger.warning(f"No template candidates found for stack: {stack_name}")
                result["errors"].append("No template candidates found")
                return result

            # Filter and validate candidates
            validated_candidates = []
            for candidate in candidates:
                if self._validate_candidate_for_stack(candidate, stack_name):
                    validated_candidates.append(candidate)
                else:
                    result["templates_skipped"] += 1

            # Limit to target number of templates
            validated_candidates = validated_candidates[:self.templates_per_stack]

            if not validated_candidates:
                logger.warning(f"No valid template candidates for stack: {stack_name}")
                result["errors"].append("No valid template candidates after validation")
                return result

            # Populate templates
            templates_added = 0
            for candidate in validated_candidates:
                try:
                    template_result = await self._add_template_to_stack(candidate, stack_name)
                    if template_result["success"]:
                        templates_added += 1
                        result["templates"].append({
                            "name": template_result["template_name"],
                            "source_url": template_result["source_url"],
                            "quality_score": candidate.get("quality_score", 0.0)
                        })
                    else:
                        result["errors"].extend(template_result["errors"])
                        result["templates_skipped"] += 1

                except Exception as e:
                    logger.error(f"Failed to add template to stack {stack_name}: {e}")
                    result["errors"].append(f"Template addition failed: {str(e)}")
                    result["templates_skipped"] += 1

            # Update stack documentation
            if templates_added > 0:
                await self._update_stack_documentation(stack_name, result["templates"])

            result["success"] = True
            result["templates_added"] = templates_added

            logger.info(f"Successfully populated stack {stack_name} with {templates_added} templates")

        except Exception as e:
            logger.error(f"Stack population failed for {stack_name}: {e}")
            result["errors"].append(str(e))

        return result

    def _validate_candidate_for_stack(self, candidate: Dict[str, Any], stack_name: str) -> bool:
        """
        Validate a template candidate for a specific stack.

        Args:
            candidate: Template candidate data
            stack_name: Target stack name

        Returns:
            True if candidate is valid for the stack
        """
        try:
            # Check basic requirements
            repo_data = candidate.get("repository", {})
            stars = repo_data.get("stargazers_count", 0)
            forks = repo_data.get("forks_count", 0)
            quality_score = candidate.get("quality_score", 0.0)

            # Get stack requirements
            stack_config = self.stack_config.get_stack_config(stack_name)
            if not stack_config:
                return False

            requirements = stack_config.requirements

            # Validate against requirements
            if stars < requirements.min_stars:
                logger.debug(f"Template {repo_data.get('name')} rejected: insufficient stars ({stars} < {requirements.min_stars})")
                return False

            if forks < requirements.min_forks:
                logger.debug(f"Template {repo_data.get('name')} rejected: insufficient forks ({forks} < {requirements.min_forks})")
                return False

            if quality_score < self.min_template_quality:
                logger.debug(f"Template {repo_data.get('name')} rejected: insufficient quality ({quality_score} < {self.min_template_quality})")
                return False

            # Check stack validation
            stack_validation = candidate.get("stack_validation", {})
            if not stack_validation.get("valid", False):
                logger.debug(f"Template {repo_data.get('name')} rejected: failed stack validation")
                return False

            # Check for duplicates
            if self._is_template_already_added(repo_data.get("full_name"), stack_name):
                logger.debug(f"Template {repo_data.get('name')} rejected: already exists in stack")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating candidate: {e}")
            return False

    async def _add_template_to_stack(self, candidate: Dict[str, Any], stack_name: str) -> Dict[str, Any]:
        """
        Add a validated template candidate to a stack branch.

        Args:
            candidate: Validated template candidate
            stack_name: Target stack name

        Returns:
            Template addition result
        """
        result = {
            "success": False,
            "template_name": None,
            "source_url": None,
            "errors": []
        }

        try:
            repo_data = candidate["repository"]
            repo_name = repo_data.get("name", "")
            repo_full_name = repo_data.get("full_name", "")
            repo_url = repo_data.get("html_url", "")

            # Generate template name
            template_name = self._generate_template_name(repo_name, repo_full_name)
            result["template_name"] = template_name
            result["source_url"] = repo_url

            # Create template directory structure
            template_dir = Path("stacks") / stack_name / template_name
            template_dir.mkdir(parents=True, exist_ok=True)

            # Download and process template
            await self._download_and_process_template(repo_data, template_dir, stack_name)

            # Create template documentation
            await self._create_template_documentation(candidate, template_dir, stack_name)

            # Store template metadata
            self._store_template_metadata(candidate, stack_name, template_name)

            result["success"] = True
            logger.info(f"Successfully added template {template_name} to stack {stack_name}")

        except Exception as e:
            logger.error(f"Failed to add template to stack: {e}")
            result["errors"].append(str(e))

        return result

    async def _download_and_process_template(self, repo_data: Dict[str, Any], template_dir: Path, stack_name: str):
        """
        Download and process a template repository.

        Args:
            repo_data: GitHub repository data
            template_dir: Target template directory
            stack_name: Stack name for processing
        """
        repo_full_name = repo_data.get("full_name", "")
        repo_url = repo_data.get("html_url", "")

        logger.info(f"Downloading template from {repo_full_name}")

        # For now, create a minimal template structure
        # In a full implementation, this would clone/download the actual repository
        # and process it according to stack-specific rules

        # Create basic template structure
        self._create_template_structure(template_dir, repo_data, stack_name)

        # Create template configuration
        self._create_template_config(template_dir, repo_data, stack_name)

    def _create_template_structure(self, template_dir: Path, repo_data: Dict[str, Any], stack_name: str):
        """
        Create the basic template structure.

        Args:
            template_dir: Template directory
            repo_data: Repository data
            stack_name: Stack name
        """
        # Create basic files that would be in any template
        files_to_create = {
            "README.md": self._generate_template_readme(repo_data, stack_name),
            ".gitignore": self._generate_gitignore(stack_name),
            "requirements.txt": self._generate_requirements(repo_data, stack_name),
            "package.json": self._generate_package_json(repo_data, stack_name) if self._is_frontend_stack(stack_name) else None,
        }

        for filename, content in files_to_create.items():
            if content:
                (template_dir / filename).write_text(content)

        # Create src directory structure
        src_dir = template_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Create basic source files based on stack
        if stack_name == "frontend":
            self._create_frontend_template_files(src_dir, repo_data)
        elif stack_name == "backend":
            self._create_backend_template_files(src_dir, repo_data)
        elif stack_name == "ai-ml":
            self._create_ai_ml_template_files(src_dir, repo_data)
        elif stack_name == "devops":
            self._create_devops_template_files(src_dir, repo_data)

    def _create_template_config(self, template_dir: Path, repo_data: Dict[str, Any], stack_name: str):
        """
        Create template configuration file.

        Args:
            template_dir: Template directory
            repo_data: Repository data
            stack_name: Stack name
        """
        config = {
            "name": repo_data.get("name", ""),
            "description": repo_data.get("description", ""),
            "stack": stack_name,
            "upstream_url": repo_data.get("html_url", ""),
            "stars": repo_data.get("stargazers_count", 0),
            "forks": repo_data.get("forks_count", 0),
            "language": repo_data.get("language", ""),
            "topics": repo_data.get("topics", []),
            "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else None,
            "created_at": datetime.now().isoformat(),
            "template_version": "1.0.0"
        }

        config_file = template_dir / "template.json"
        config_file.write_text(json.dumps(config, indent=2))

    async def _create_template_documentation(self, candidate: Dict[str, Any], template_dir: Path, stack_name: str):
        """
        Create comprehensive template documentation.

        Args:
            candidate: Template candidate data
            template_dir: Template directory
            stack_name: Stack name
        """
        repo_data = candidate["repository"]

        # Update main README with template information
        readme_path = template_dir / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            # Add additional documentation sections
            enhanced_content = self._enhance_template_readme(content, candidate, stack_name)
            readme_path.write_text(enhanced_content)

    def _store_template_metadata(self, candidate: Dict[str, Any], stack_name: str, template_name: str):
        """
        Store template metadata in cache.

        Args:
            candidate: Template candidate data
            stack_name: Stack name
            template_name: Template name
        """
        repo_data = candidate["repository"]

        metadata = {
            "name": template_name,
            "stack": stack_name,
            "source_url": repo_data.get("html_url", ""),
            "local_path": f"stacks/{stack_name}/{template_name}",
            "github_id": repo_data.get("id"),
            "stars": repo_data.get("stargazers_count", 0),
            "forks": repo_data.get("forks_count", 0),
            "growth_rate": 0.0,  # Would be calculated from metrics
            "quality_score": candidate.get("quality_score", 0.0),
            "validation_status": "approved"
        }

        self.cache.store_template_metadata(metadata)

    async def _update_stack_documentation(self, stack_name: str, templates: List[Dict[str, Any]]):
        """
        Update stack branch documentation with new templates.

        Args:
            stack_name: Stack name
            templates: List of added templates
        """
        # This would update the stack README with the new templates
        # For now, just log the information
        logger.info(f"Stack {stack_name} documentation would be updated with {len(templates)} templates")

        # In a full implementation, this would:
        # 1. Read the current stack README
        # 2. Update the template list
        # 3. Update statistics
        # 4. Commit the changes

    def _generate_template_name(self, repo_name: str, repo_full_name: str) -> str:
        """
        Generate a clean template name from repository information.

        Args:
            repo_name: Repository name
            repo_full_name: Full repository name

        Returns:
            Clean template name
        """
        # Use repo name, clean it up
        name = repo_name.lower().replace("_", "-").replace(" ", "-")

        # Remove common prefixes/suffixes
        prefixes_to_remove = ["template", "starter", "boilerplate", "example"]
        for prefix in prefixes_to_remove:
            if name.startswith(prefix + "-"):
                name = name[len(prefix) + 1:]
            elif name.endswith("-" + prefix):
                name = name[:-len(prefix) - 1]

        return name or repo_name.lower()

    def _is_template_already_added(self, repo_full_name: str, stack_name: str) -> bool:
        """
        Check if a template from this repository is already added to the stack.

        Args:
            repo_full_name: Full repository name
            stack_name: Stack name

        Returns:
            True if template already exists
        """
        # Check cache for existing templates from this repo
        templates = self.cache.find_template_candidates(stack_name, limit=100)
        for template in templates:
            if template.get("repository", {}).get("full_name") == repo_full_name:
                return True
        return False

    def _is_frontend_stack(self, stack_name: str) -> bool:
        """Check if stack is frontend-related."""
        return stack_name in ["frontend", "fullstack"]

    # Template file generation methods
    def _generate_template_readme(self, repo_data: Dict[str, Any], stack_name: str) -> str:
        """Generate template README content."""
        name = repo_data.get("name", "")
        description = repo_data.get("description", "")
        url = repo_data.get("html_url", "")

        return f"""# {name}

{description or "A template for " + stack_name + " development."}

## 🚀 Features

- Based on [{repo_data.get('full_name', '')}]({url})
- {repo_data.get('stargazers_count', 0)} stars on GitHub
- {repo_data.get('language', 'Multiple languages')} primary language
- {", ".join(repo_data.get('topics', [])[:5])}

## 📋 Prerequisites

- See original repository for requirements

## 🛠️ Quick Start

```bash
# Copy this template
cp -r . ../my-project
cd ../my-project

# Install dependencies and run
# (See original repository documentation)
```

## 🔗 Upstream Source

- **Repository**: [{repo_data.get('full_name', '')}]({url})
- **License**: {repo_data.get('license', {}).get('name', 'Unknown') if repo_data.get('license') else 'Unknown'}

---
*This template is automatically generated from the upstream repository.*
"""

    def _generate_gitignore(self, stack_name: str) -> str:
        """Generate .gitignore content based on stack."""
        base_gitignore = """
# Dependencies
node_modules/
__pycache__/
*.pyc
venv/
env/

# Build outputs
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment
.env
.env.local
"""

        if stack_name == "frontend":
            base_gitignore += """
# Frontend specific
.next/
.nuxt/
.cache/
.parcel-cache/
"""
        elif stack_name == "backend":
            base_gitignore += """
# Backend specific
*.sqlite3
*.db
media/
staticfiles/
"""
        elif stack_name == "ai-ml":
            base_gitignore += """
# ML specific
models/
checkpoints/
*.h5
*.pb
*.onnx
"""

        return base_gitignore.strip()

    def _generate_requirements(self, repo_data: Dict[str, Any], stack_name: str) -> str:
        """Generate requirements.txt content."""
        language = repo_data.get("language", "").lower()

        requirements = [
            "# Template Heaven - Generated Requirements",
            "# Based on upstream repository analysis",
            "",
        ]

        if language == "python":
            requirements.extend([
                "requests>=2.25.0",
                "python-dotenv>=0.19.0",
                "click>=8.0.0",
            ])
        elif stack_name == "frontend":
            requirements.extend([
                "# Frontend dependencies would be in package.json",
                "# This file is for Python tooling if any",
            ])
        else:
            requirements.extend([
                "# Add stack-specific dependencies here",
            ])

        return "\n".join(requirements)

    def _generate_package_json(self, repo_data: Dict[str, Any], stack_name: str) -> str:
        """Generate package.json for frontend templates."""
        name = repo_data.get("name", "").lower().replace(" ", "-")
        description = repo_data.get("description", "")

        package_json = {
            "name": f"templateheaven-{stack_name}-{name}",
            "version": "1.0.0",
            "description": description or f"A {stack_name} template based on {repo_data.get('full_name', '')}",
            "main": "src/index.js",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview",
                "test": "vitest",
                "lint": "eslint src --ext js,jsx,ts,tsx",
                "format": "prettier --write src/**/*.{js,jsx,ts,tsx,json,css,md}"
            },
            "keywords": repo_data.get("topics", []),
            "author": repo_data.get("owner", {}).get("login", ""),
            "license": repo_data.get("license", {}).get("name", "MIT") if repo_data.get("license") else "MIT",
            "repository": {
                "type": "git",
                "url": repo_data.get("html_url", "")
            },
            "dependencies": {},
            "devDependencies": {
                "vite": "^4.0.0",
                "vitest": "^0.30.0",
                "eslint": "^8.0.0",
                "prettier": "^2.8.0"
            }
        }

        return json.dumps(package_json, indent=2)

    def _create_frontend_template_files(self, src_dir: Path, repo_data: Dict[str, Any]):
        """Create frontend-specific template files."""
        # Create basic React component
        app_content = """import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Template Heaven Frontend Template</h1>
        <p>Based on {repo_data.get('full_name', '')}</p>
      </header>
    </div>
  );
}

export default App;
"""
        (src_dir / "App.jsx").write_text(app_content)

        # Create basic CSS
        css_content = """/* Template Heaven - Basic Styles */
.App {
  text-align: center;
}

.App-header {
  background-color: #282c34;
  padding: 20px;
  color: white;
}

.App-link {
  color: #61dafb;
}
"""
        (src_dir / "App.css").write_text(css_content)

    def _create_backend_template_files(self, src_dir: Path, repo_data: Dict[str, Any]):
        """Create backend-specific template files."""
        # Create basic FastAPI app
        app_content = '''"""
Template Heaven Backend Template
Based on {repo_data.get('full_name', '')}
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Template Heaven Backend",
    description="A backend template",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Template Heaven Backend Template"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
'''
        (src_dir / "main.py").write_text(app_content)

    def _create_ai_ml_template_files(self, src_dir: Path, repo_data: Dict[str, Any]):
        """Create AI/ML-specific template files."""
        # Create basic ML script
        ml_content = '''"""
Template Heaven AI/ML Template
Based on {repo_data.get('full_name', '')}
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    """Basic ML pipeline example."""
    print("Template Heaven AI/ML Template")
    print("Based on {repo_data.get('full_name', '')}")

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
'''
        (src_dir / "main.py").write_text(ml_content)

    def _create_devops_template_files(self, src_dir: Path, repo_data: Dict[str, Any]):
        """Create DevOps-specific template files."""
        # Create Dockerfile
        dockerfile_content = """# Template Heaven DevOps Template
# Based on {repo_data.get('full_name', '')}

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
"""
        (src_dir.parent / "Dockerfile").write_text(dockerfile_content)

        # Create docker-compose.yml
        compose_content = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
"""
        (src_dir.parent / "docker-compose.yml").write_text(compose_content)

    def _enhance_template_readme(self, content: str, candidate: Dict[str, Any], stack_name: str) -> str:
        """Enhance template README with additional information."""
        # Add quality metrics and validation info
        quality_score = candidate.get("quality_score", 0.0)
        stack_validation = candidate.get("stack_validation", {})

        enhancement = f"""

## 📊 Quality Metrics

- **Quality Score**: {quality_score:.2f}/1.0
- **Stack Validation**: {'✅ Passed' if stack_validation.get('valid', False) else '❌ Failed'}
- **Template Potential**: {candidate.get('template_potential', 0.0):.2f}/1.0

## 🔍 Validation Results

### Passed Checks
{chr(10).join(f"- {reason}" for reason in candidate.get('reasons', []))}

### Issues
{chr(10).join(f"- {issue}" for issue in stack_validation.get('issues', []))}

### Warnings
{chr(10).join(f"- {warning}" for warning in stack_validation.get('warnings', []))}
"""

        return content + enhancement

    async def get_population_status(self) -> Dict[str, Any]:
        """
        Get the current population status across all stacks.

        Returns:
            Status information
        """
        status = {
            "stacks": {},
            "total_templates": 0,
            "last_updated": None
        }

        # Get all stacks
        all_stacks = self.stack_config.get_all_stacks()

        for stack_name in all_stacks:
            stack_status = {
                "templates_count": 0,
                "last_updated": None,
                "quality_average": 0.0
            }

            # Count templates in stack directory
            stack_dir = Path("stacks") / stack_name
            if stack_dir.exists():
                templates = [d for d in stack_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                stack_status["templates_count"] = len(templates)

            status["stacks"][stack_name] = stack_status
            status["total_templates"] += stack_status["templates_count"]

        return status
