"""
Interactive Wizard for Template Heaven.

This module provides an interactive wizard for project initialization
with beautiful terminal output using Rich and Questionary.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm
import asyncio
import tempfile
import subprocess

from ..core.template_manager import TemplateManager
from ..core.models import Template, ProjectConfig, StackCategory
from ..core.customizer import Customizer
from ..core.architecture_questionnaire import (
    ArchitectureQuestionnaire, ArchitectureAnswers,
    ArchitecturePattern, DeploymentModel, ScalabilityRequirement,
    ArchitectureQuestion
)
from ..config.settings import Config
from ..utils.logger import get_logger
from ..utils.helpers import validate_project_name, sanitize_project_name

logger = get_logger(__name__)


class Wizard:
    """
    Interactive wizard for project initialization.
    
    Guides users through:
    1. Stack category selection
    2. Template selection
    3. Project configuration
    4. Basic customization
    5. Project creation
    
    Uses Rich for beautiful terminal output and Questionary for interactive prompts.
    """
    
    def __init__(self, template_manager: TemplateManager, config: Config):
        """
        Initialize the wizard.
        
        Args:
            template_manager: Template manager instance
            config: Configuration instance
        """
        self.template_manager = template_manager
        self.config = config
        self.console = Console()
        self.customizer = Customizer()
        self.architecture_questionnaire = ArchitectureQuestionnaire()
        
        logger.debug("Wizard initialized")
    
    def run(self, output_dir: Path = Path('.')) -> None:
        """
        Run the complete wizard flow.
        
        Args:
            output_dir: Output directory for the project
        """
        try:
            # Display welcome message
            self._display_welcome()
            
            # Step 1: Select stack category
            stack = self._select_stack()
            
            # Step 2: Select template
            template = self._select_template(stack)
            
            # Step 3: Configure project
            project_config = self._configure_project(template, output_dir)
            
            # Step 4: Architecture questionnaire (mandatory)
            architecture_answers = self._collect_architecture_answers(project_config)
            project_config.architecture_answers = architecture_answers
            
            # Step 5: Confirm and create
            if self._confirm_creation(project_config):
                self._create_project(project_config)
            else:
                self.console.print("[yellow]Project creation cancelled[/yellow]")
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Wizard cancelled by user[/yellow]")
        except Exception as e:
            logger.error(f"Wizard failed: {e}")
            self.console.print(f"[red]Wizard failed: {e}[/red]")
            raise
    
    def _display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = """
Welcome to Template Heaven! 🎉

This wizard will help you create a new project from one of our templates.
We'll guide you through selecting a template and configuring your project.

Let's get started!
"""
        
        panel = Panel(
            Text(welcome_text.strip(), style="white"),
            title="Template Heaven Wizard",
            border_style="blue"
        )
        
        self.console.print(panel)
        self.console.print()
    
    def _select_stack(self) -> StackCategory:
        """
        Interactive stack selection.
        
        Returns:
            Selected stack category
        """
        self.console.print("[bold cyan]Step 1: Select Technology Stack[/bold cyan]")
        self.console.print()
        
        # Get available stacks with template counts
        stacks = self.template_manager.get_stacks()
        stack_choices = []
        
        for stack in stacks:
            stack_info = self.template_manager.get_stack_info(stack)
            choice_text = f"{stack_info['name']} ({stack_info['template_count']} templates)"
            stack_choices.append((choice_text, stack))
        
        # Add search option
        stack_choices.append(("🔍 Search all templates", "search"))
        
        # Display stack selection
        selected = questionary.select(
            "Choose a technology stack:",
            choices=stack_choices,
            style=questionary.Style([
                ('qmark', 'fg:#673ab7 bold'),
                ('question', 'bold'),
                ('answer', 'fg:#f44336 bold'),
                ('pointer', 'fg:#673ab7 bold'),
                ('highlighted', 'fg:#673ab7 bold'),
                ('selected', 'fg:#cc5454'),
                ('separator', 'fg:#cc5454'),
                ('instruction', ''),
                ('text', ''),
                ('disabled', 'fg:#858585 italic')
            ])
        ).ask()
        
        if selected == "search":
            return self._search_all_templates()
        
        self.console.print(f"[green]Selected: {selected}[/green]")
        self.console.print()
        
        return selected
    
    def _search_all_templates(self) -> StackCategory:
        """
        Search across all templates.
        
        Returns:
            Stack category of selected template
        """
        query = questionary.text("Enter search query:").ask()

        if not query:
            return self._select_stack()  # Fallback to stack selection

        # Ask user where to search: local vs GitHub
        # Default to GitHub if configured to prefer GitHub
        default_choice = 'github' if self.template_manager.prefer_github else 'local'
        search_source = questionary.select(
            "Where should we search?",
            choices=[('Local templates (bundled)', 'local'), ('GitHub repositories (live search)', 'github')],
            default=default_choice
        ).ask()

        results = []
        if search_source == 'github':
            try:
                if not self.template_manager.github_search.github_available:
                    self.console.print('[yellow]GitHub integration not available or dependencies missing.[/yellow]')
                    results = self.template_manager.search_templates(query, limit=10)
                else:
                    results = asyncio.run(self.template_manager.github_search.search_github_templates(query, limit=10))
            except Exception as e:
                self.console.print(f"[red]GitHub search failed: {e}[/red]")
                results = self.template_manager.search_templates(query, limit=10)
        else:
            # Local template search
            results = self.template_manager.search_templates(query, limit=10)

        if not results:
            self.console.print("[yellow]No templates found matching your query.[/yellow]")
            return self._select_stack()  # Fallback to stack selection
        
        # Display search results
        self._display_search_results(results)
        
        # Select from results
        choices = []
        for result in results:
            if hasattr(result, 'metadata') and result.metadata and 'github_data' in result.metadata:
                repo = result.metadata['github_data']
                choice_text = f"{repo.get('full_name')} - {repo.get('description', '')[:50]}..."
            else:
                choice_text = f"{result.template.name} ({result.template.stack.value}) - {result.template.description[:50]}..."
            choices.append((choice_text, result))
        
        selected_template = questionary.select(
            "Choose a template:",
            choices=choices
        ).ask()

        # If the selected result is a GitHub candidate, handle cloning and scaffolding
        if hasattr(selected_template, 'metadata') and selected_template.metadata and 'github_data' in selected_template.metadata:
            repo = selected_template.metadata['github_data']
            repo_url = repo.get('html_url') or repo.get('clone_url') or repo.get('ssh_url')
            clone_confirm = questionary.confirm(f"Scaffold from GitHub repo {repo.get('full_name')}? ").ask()
            if not clone_confirm:
                return self._select_stack()

            # Ask for project name
            pname = questionary.text("Project name to scaffold into:", default='my-app').ask()
            project_name = pname or 'my-app'

            # Clone and scaffold into the selected output
            tmp_dir = Path(tempfile.mkdtemp(prefix='th-git-'))
            try:
                subprocess.run(['git', 'clone', '--depth', '1', repo_url, str(tmp_dir)], check=True)

                # Build a minimal ProjectConfig and call customizer.customize_from_repo_dir
                project_config = ProjectConfig(
                    name=project_name,
                    directory=str(Path.cwd()),
                    template=selected_template.template,
                    author=self.config.get('default_author'),
                    license=self.config.get('default_license'),
                    package_manager=self.config.get('package_managers', {}).get('python', 'pip')
                )

                success = self.customizer.customize_from_repo_dir(tmp_dir, project_config, Path('.'))
                if success:
                    self.console.print(f"[green]Scaffolded project {project_name} from GitHub repo {repo_url}[/green]")
                    return project_config.template.stack
                else:
                    self.console.print(f"[red]Failed to scaffold from GitHub repo: {repo_url}[/red]")
                    return self._select_stack()
            except Exception as e:
                self.console.print(f"[red]Failed to clone or scaffold repository: {e}[/red]")
                return self._select_stack()
        else:
            return selected_template.template.stack
    
    def _select_template(self, stack: StackCategory) -> Template:
        """
        Interactive template selection for stack.
        
        Args:
            stack: Selected stack category
            
        Returns:
            Selected template
        """
        self.console.print("[bold cyan]Step 2: Select Template[/bold cyan]")
        self.console.print()
        
        # Get templates for the stack
        # Use GitHub discovery by default if configured
        use_github = self.template_manager.prefer_github
        templates = self.template_manager.list_templates(stack=stack.value, use_github=use_github)
        
        if not templates:
            self.console.print(f"[red]No templates found for stack: {stack.value}[/red]")
            raise ValueError(f"No templates found for stack: {stack.value}")
        
        # Display templates
        self._display_templates(templates)
        
        # Create choices
        choices = []
        for template in templates:
            # Create choice text with template info
            choice_text = f"{template.name} - {template.description}"
            if template.version:
                choice_text += f" (v{template.version})"
            
            choices.append((choice_text, template))
        
        # Add back option
        choices.append(("← Back to stack selection", None))
        
        # Select template
        selected = questionary.select(
            f"Choose a template from {stack.value}:",
            choices=choices
        ).ask()
        
        if selected is None:
            return self._select_template(self._select_stack())
        
        self.console.print(f"[green]Selected: {selected.name}[/green]")
        self.console.print()
        
        return selected
    
    def _configure_project(self, template: Template, output_dir: Path) -> ProjectConfig:
        """
        Configure project settings.
        
        Args:
            template: Selected template
            output_dir: Output directory
            
        Returns:
            Project configuration
        """
        self.console.print("[bold cyan]Step 3: Configure Project[/bold cyan]")
        self.console.print()
        
        # Project name
        project_name = self._get_project_name()
        
        # Author
        author = self._get_author()
        
        # License
        license_type = self._get_license()
        
        # Package manager
        package_manager = self._get_package_manager(template)
        
        # Description
        description = self._get_description(template)
        
        # Create project configuration
        project_config = ProjectConfig(
            name=project_name,
            directory=str(output_dir),
            template=template,
            author=author,
            license=license_type,
            package_manager=package_manager,
            description=description
        )
        
        return project_config
    
    def _get_project_name(self) -> str:
        """Get project name from user."""
        while True:
            name = questionary.text(
                "Project name:",
                default="my-project"
            ).ask()
            
            if not name:
                self.console.print("[red]Project name cannot be empty[/red]")
                continue
            
            try:
                validate_project_name(name)
                return name
            except ValueError as e:
                self.console.print(f"[red]{e}[/red]")
                
                # Suggest sanitized name
                sanitized = sanitize_project_name(name)
                if sanitized != name:
                    if Confirm.ask(f"Use '{sanitized}' instead?"):
                        return sanitized
    
    def _get_author(self) -> str:
        """Get project author from user."""
        default_author = self.config.get('default_author', 'Template Heaven User')
        
        author = questionary.text(
            "Author:",
            default=default_author
        ).ask()
        
        return author or default_author
    
    def _get_license(self) -> str:
        """Get project license from user."""
        licenses = [
            "MIT",
            "Apache-2.0",
            "GPL-3.0",
            "BSD-3-Clause",
            "ISC",
            "Unlicense"
        ]
        
        default_license = self.config.get('default_license', 'MIT')
        
        license_type = questionary.select(
            "License:",
            choices=licenses,
            default=default_license
        ).ask()
        
        return license_type
    
    def _get_package_manager(self, template: Template) -> str:
        """Get package manager based on template."""
        # Determine appropriate package managers based on template
        if any(tag in template.tags for tag in ['python', 'fastapi', 'django', 'pytorch']):
            managers = ['pip', 'poetry']
            default = self.config.get('package_managers.python', 'pip')
        elif any(tag in template.tags for tag in ['nodejs', 'react', 'vue', 'typescript', 'nextjs']):
            managers = ['npm', 'yarn', 'pnpm']
            default = self.config.get('package_managers.node', 'npm')
        elif any(tag in template.tags for tag in ['rust']):
            managers = ['cargo']
            default = 'cargo'
        elif any(tag in template.tags for tag in ['go']):
            managers = ['go']
            default = 'go'
        else:
            managers = ['npm', 'pip']
            default = 'npm'
        
        if len(managers) == 1:
            return managers[0]
        
        package_manager = questionary.select(
            "Package manager:",
            choices=managers,
            default=default
        ).ask()
        
        return package_manager
    
    def _get_description(self, template: Template) -> str:
        """Get project description from user."""
        default_desc = f"A {template.stack.value} project created with {template.name}"
        
        description = questionary.text(
            "Project description:",
            default=default_desc
        ).ask()
        
        return description or default_desc
    
    def _collect_architecture_answers(self, project_config: ProjectConfig) -> ArchitectureAnswers:
        """
        Collect architecture questionnaire answers.
        
        Args:
            project_config: Project configuration
            
        Returns:
            ArchitectureAnswers object
        """
        self.console.print("[bold cyan]Step 4: Architecture & System Design Questionnaire[/bold cyan]")
        self.console.print()
        
        panel_text = """
This questionnaire is MANDATORY to prevent architectural drift and ensure
your project has proper system design documentation.

The questions cover:
- Architecture patterns and deployment models
- Performance, security, and compliance requirements
- Infrastructure and data architecture
- API design and observability
- Feature prioritization and roadmap
- Risk assessment and mitigation
"""

        panel = Panel(
            Text(panel_text.strip(), style="white"),
            title="Architecture Questionnaire",
            border_style="yellow"
        )
        self.console.print(panel)
        self.console.print()

        # Ask user to choose between quick mode and comprehensive mode
        mode_choice = questionary.select(
            "Choose questionnaire mode:",
            choices=[
                questionary.Choice(
                    title="Quick Mode (10 essential questions - recommended for getting started)",
                    value="quick"
                ),
                questionary.Choice(
                    title="Comprehensive Mode (47 detailed questions - thorough architecture planning)",
                    value="comprehensive"
                )
            ],
            default="quick"
        ).ask()

        # Reinitialize questionnaire with selected mode
        quick_mode = (mode_choice == "quick")
        self.architecture_questionnaire = ArchitectureQuestionnaire(quick_mode=quick_mode)

        if quick_mode:
            self.console.print("[green]Quick mode selected: 10 essential questions[/green]")
        else:
            self.console.print("[yellow]Comprehensive mode selected: 47 detailed questions[/yellow]")
        self.console.print()

        # Ask if user wants to use AI assistance
        use_ai = questionary.confirm(
            "Would you like to use AI/LLM assistance to fill out the questionnaire?",
            default=False
        ).ask()
        
        if use_ai:
            return self._collect_answers_with_ai(project_config)
        else:
            return self._collect_answers_manually(project_config)
    
    async def _collect_answers_with_ai(self, project_config: ProjectConfig) -> ArchitectureAnswers:
        """Collect answers using AI/LLM assistance with multi-turn conversation."""
        from ..core.llm import get_llm_provider, ConversationManager, SystemDesignAgent, SystemDesignContext
        from ..core.architecture_questionnaire import ArchitectureQuestionnaire
        
        self.console.print("[bold cyan]🤖 Starting AI-Powered System Design Consultation[/bold cyan]")
        self.console.print()
        
        # Check for LLM configuration
        llm_provider_name = self.config.get('llm.provider', 'openai')
        llm_api_key = self.config.get('llm.api_key') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        if not llm_api_key:
            self.console.print("[yellow]⚠️  No LLM API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.[/yellow]")
            self.console.print("[yellow]Falling back to manual entry...[/yellow]")
            self.console.print()
            return self._collect_answers_manually(project_config)
        
        try:
            # Initialize LLM provider
            llm_config = {
                "api_key": llm_api_key,
                "model": self.config.get('llm.model'),
                "temperature": self.config.get('llm.temperature', 0.7)
            }
            llm_provider = get_llm_provider(llm_provider_name, llm_config)
            
            # Create conversation manager and agent
            conversation_manager = ConversationManager()
            agent = SystemDesignAgent(llm_provider, conversation_manager, self.architecture_questionnaire)
            
            # Create context
            context = SystemDesignContext(
                project_name=project_config.name,
                project_description=project_config.description,
                template_stack=project_config.stack if hasattr(project_config, 'stack') else None,
                current_answers={}
            )
            
            # Start conversation
            state = await agent.start_conversation(context)
            session_id = state.session_id
            
            self.console.print(f"[green]✅ Conversation started (Session: {session_id[:8]}...)[/green]")
            self.console.print()
            
            # Display greeting
            if state.messages:
                greeting = state.messages[-1]["content"]
                self.console.print(f"[cyan]Assistant:[/cyan] {greeting}")
                self.console.print()
            
            # Interactive conversation loop
            answers = ArchitectureAnswers()
            questions_answered = set()
            
            # Get questions by category
            questions_by_category = self.architecture_questionnaire.get_questions_by_category()
            
            for category, questions in questions_by_category.items():
                if not questions:
                    continue
                
                self.console.print(f"[bold yellow]📋 {category}[/bold yellow]")
                self.console.print()
                
                for question in questions:
                    if question.id in questions_answered:
                        continue
                    
                    # Ask question via LLM
                    question_text = await agent.ask_question(session_id, question, {
                        "project_name": project_config.name,
                        "category": category
                    })
                    
                    self.console.print(f"[cyan]🤖 {question_text}[/cyan]")
                    self.console.print()
                    
                    # Get user response
                    user_response = questionary.text(
                        "Your answer:",
                        default=""
                    ).ask()
                    
                    if not user_response:
                        continue
                    
                    # Continue conversation with user response
                    llm_response = await agent.continue_conversation(session_id, user_response, stream=False)
                    
                    # Display LLM response if it provides guidance
                    if llm_response and len(llm_response) > 50:
                        self.console.print(f"[green]💡 {llm_response[:200]}...[/green]")
                        self.console.print()
                    
                    # Map answer to ArchitectureAnswers (reuse existing mapping logic)
                    self._map_answer_to_answers(answers, question, user_response)
                    questions_answered.add(question.id)
                    
                    # Ask if user wants to search for open-source solutions
                    if question.category in ["Integration", "Infrastructure", "API Design"]:
                        search_decision = questionary.confirm(
                            "Would you like me to search for existing open-source solutions?",
                            default=False
                        ).ask()
                        
                        if search_decision:
                            # Trigger repository search
                            recommendation = await agent.suggest_repository_search(
                                session_id,
                                user_response,
                                answers.technology_stack if hasattr(answers, 'technology_stack') else None
                            )
                            
                            self.console.print(f"[blue]🔍 Searching for solutions...[/blue]")
                            self.console.print()
                            
                            # Note: Actual repo analysis would happen via API endpoint
                            # For now, just show the suggestion
                            if "search_strategy" in recommendation:
                                self.console.print(f"[green]💡 {recommendation['search_strategy'][:200]}...[/green]")
                                self.console.print()
            
            # Get final architecture recommendations
            requirements = {
                "project_name": project_config.name,
                "architecture_pattern": answers.architecture_pattern.value if answers.architecture_pattern else None,
                "scalability": answers.scalability_requirement.value if answers.scalability_requirement else None
            }
            
            # Get architecture recommendation
            recommendation = await agent.suggest_architecture_pattern(session_id, requirements)
            
            self.console.print()
            self.console.print("[bold green]✅ Architecture consultation complete![/bold green]")
            self.console.print()
            
            # Export conversation for documentation
            conversation_export = conversation_manager.export_conversation(session_id)
            answers.reference_architectures = [f"Conversation: {session_id}"]
            answers.additional_notes = f"AI consultation session: {session_id}\n\n{conversation_export.get('messages', [])[-1].get('content', '') if conversation_export.get('messages') else ''}"
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in AI conversation: {e}")
            self.console.print(f"[red]❌ Error in AI conversation: {e}[/red]")
            self.console.print("[yellow]Falling back to manual entry...[/yellow]")
            self.console.print()
            return self._collect_answers_manually(project_config)
    
    def _collect_answers_manually(self, project_config: ProjectConfig) -> ArchitectureAnswers:
        """Collect answers manually through interactive prompts."""
        answers = ArchitectureAnswers()
        
        questions_by_category = self.architecture_questionnaire.get_questions_by_category()
        
        for category, questions in questions_by_category.items():
            self.console.print(f"[bold cyan]{category}[/bold cyan]")
            self.console.print()
            
            for question in questions:
                answer = self._ask_question(question, project_config)
                
                # Map answer to ArchitectureAnswers object
                if question.id == "project_vision":
                    answers.project_vision = answer
                elif question.id == "target_users":
                    answers.target_users = answer
                elif question.id == "business_objectives":
                    answers.business_objectives = [obj.strip() for obj in answer.split(",") if obj.strip()]
                elif question.id == "success_metrics":
                    answers.success_metrics = [metric.strip() for metric in answer.split(",") if metric.strip()]
                elif question.id == "architecture_pattern":
                    try:
                        answers.architecture_pattern = ArchitecturePattern(answer)
                    except ValueError:
                        pass
                elif question.id == "deployment_model":
                    try:
                        answers.deployment_model = DeploymentModel(answer)
                    except ValueError:
                        pass
                elif question.id == "scalability_requirement":
                    try:
                        answers.scalability_requirement = ScalabilityRequirement(answer)
                    except ValueError:
                        pass
                elif question.id == "performance_requirements":
                    try:
                        import json
                        answers.performance_requirements = json.loads(answer) if answer else {}
                    except:
                        answers.performance_requirements = {}
                elif question.id == "security_requirements":
                    answers.security_requirements = [req.strip() for req in answer.split(",") if req.strip()]
                elif question.id == "compliance_requirements":
                    answers.compliance_requirements = [req.strip() for req in answer.split(",") if req.strip()]
                elif question.id == "integration_requirements":
                    answers.integration_requirements = [req.strip() for req in answer.split(",") if req.strip()]
                elif question.id == "cloud_provider":
                    answers.cloud_provider = answer if answer != "none" else None
                elif question.id == "containerization":
                    answers.containerization = answer
                elif question.id == "orchestration_platform":
                    answers.orchestration_platform = answer if answer != "none" else None
                elif question.id == "database_requirements":
                    answers.database_requirements = [db.strip() for db in answer.split(",") if db.strip()]
                elif question.id == "caching_strategy":
                    answers.caching_strategy = answer if answer != "none" else None
                elif question.id == "cdn_required":
                    answers.cdn_required = answer
                elif question.id == "data_volume":
                    answers.data_volume = answer
                elif question.id == "data_velocity":
                    answers.data_velocity = answer
                elif question.id == "data_variety":
                    answers.data_variety = answer
                elif question.id == "data_retention_policy":
                    answers.data_retention_policy = answer
                elif question.id == "backup_strategy":
                    answers.backup_strategy = answer
                elif question.id == "api_style":
                    answers.api_style = answer
                elif question.id == "api_versioning_strategy":
                    answers.api_versioning_strategy = answer
                elif question.id == "api_security_model":
                    answers.api_security_model = answer
                elif question.id == "api_rate_limiting":
                    answers.api_rate_limiting = answer
                elif question.id == "logging_strategy":
                    answers.logging_strategy = answer
                elif question.id == "monitoring_strategy":
                    answers.monitoring_strategy = answer
                elif question.id == "tracing_strategy":
                    answers.tracing_strategy = answer if answer != "none" else None
                elif question.id == "alerting_strategy":
                    answers.alerting_strategy = answer
                elif question.id == "ci_cd_strategy":
                    answers.ci_cd_strategy = answer
                elif question.id == "testing_strategy":
                    answers.testing_strategy = [test.strip() for test in answer.split(",") if test.strip()]
                elif question.id == "code_review_process":
                    answers.code_review_process = answer
                elif question.id == "deployment_frequency":
                    answers.deployment_frequency = answer
                elif question.id == "must_have_features":
                    answers.must_have_features = [f.strip() for f in answer.split(",") if f.strip()]
                elif question.id == "nice_to_have_features":
                    answers.nice_to_have_features = [f.strip() for f in answer.split(",") if f.strip()]
                elif question.id == "future_features":
                    answers.future_features = [f.strip() for f in answer.split(",") if f.strip()]
                elif question.id == "feature_flags_required":
                    answers.feature_flags_required = answer
                elif question.id == "technical_constraints":
                    answers.technical_constraints = [c.strip() for c in answer.split(",") if c.strip()]
                elif question.id == "business_constraints":
                    answers.business_constraints = [c.strip() for c in answer.split(",") if c.strip()]
                elif question.id == "risk_factors":
                    answers.risk_factors = [r.strip() for r in answer.split(",") if r.strip()]
                elif question.id == "mitigation_strategies":
                    answers.mitigation_strategies = [m.strip() for m in answer.split(",") if m.strip()]
                elif question.id == "team_size":
                    answers.team_size = int(answer) if answer else None
                elif question.id == "timeline":
                    answers.timeline = answer
                elif question.id == "budget_constraints":
                    answers.budget_constraints = answer if answer != "none" else None
                elif question.id == "reference_architectures":
                    answers.reference_architectures = [ref.strip() for ref in answer.split(",") if ref.strip()]
                elif question.id == "additional_notes":
                    answers.additional_notes = answer
            
            self.console.print()
        
        return answers
    
    def _ask_question(self, question: ArchitectureQuestion, project_config: ProjectConfig) -> Any:
        """Ask a single architecture question."""
        question_text = question.question
        if question.help_text:
            question_text += f"\n  [dim]{question.help_text}[/dim]"
        
        if question.question_type == "text":
            default = question.default or ""
            answer = questionary.text(
                question_text,
                default=default
            ).ask()
            return answer or default
        
        elif question.question_type == "select":
            default = question.default or (question.options[0] if question.options else None)
            answer = questionary.select(
                question_text,
                choices=question.options or [],
                default=default
            ).ask()
            return answer
        
        elif question.question_type == "boolean":
            default = question.default if question.default is not None else False
            answer = questionary.confirm(
                question_text,
                default=default
            ).ask()
            return answer
        
        elif question.question_type == "number":
            default = str(question.default) if question.default is not None else ""
            answer = questionary.text(
                question_text,
                default=default
            ).ask()
            try:
                return int(answer) if answer else None
            except ValueError:
                return None
        
        else:
            return questionary.text(question_text).ask()
    
    def _confirm_creation(self, project_config: ProjectConfig) -> bool:
        """
        Confirm project creation with preview.
        
        Args:
            project_config: Project configuration
            
        Returns:
            True if user confirms creation
        """
        self.console.print("[bold cyan]Step 4: Confirm Creation[/bold cyan]")
        self.console.print()
        
        # Display project preview
        self._display_project_preview(project_config)
        
        return Confirm.ask("Create this project?")
    
    def _create_project(self, project_config: ProjectConfig) -> None:
        """
        Create the project.
        
        Args:
            project_config: Project configuration
        """
        self.console.print("[bold cyan]Creating Project...[/bold cyan]")
        self.console.print()
        
        try:
            # Create project
            success = self.customizer.customize(
                project_config.template,
                project_config,
                Path(project_config.directory)
            )
            
            if success:
                project_path = Path(project_config.directory) / project_config.name
                
                # Display success message
                success_text = f"""
Project created successfully! 🎉

Location: {project_path}
Template: {project_config.template.name}
Stack: {project_config.template.stack.value}
"""
                
                panel = Panel(
                    Text(success_text.strip(), style="white"),
                    title="Success",
                    border_style="green"
                )
                
                self.console.print(panel)
                
                # Display next steps
                self._display_next_steps(project_path, project_config)
                
            else:
                self.console.print("[red]Project creation failed[/red]")
                
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            self.console.print(f"[red]Project creation failed: {e}[/red]")
            raise
    
    def _display_templates(self, templates: List[Template]) -> None:
        """
        Display templates in a table.
        
        Args:
            templates: List of templates to display
        """
        table = Table(title="Available Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Tags", style="blue")
        table.add_column("Version", style="yellow")
        
        for template in templates:
            # Truncate description
            description = template.description
            if len(description) > 50:
                description = description[:47] + "..."
            
            # Format tags
            tags = ", ".join(template.tags[:3])
            if len(template.tags) > 3:
                tags += f" (+{len(template.tags) - 3})"
            
            table.add_row(
                template.name,
                description,
                tags,
                template.version or "N/A"
            )
        
        self.console.print(table)
        self.console.print()
    
    def _display_search_results(self, results: List) -> None:
        """
        Display search results.
        
        Args:
            results: List of search results
        """
        table = Table(title="Search Results")
        table.add_column("Template", style="cyan")
        table.add_column("Stack", style="green")
        table.add_column("Description", style="white")
        table.add_column("Score", style="yellow")
        
        for result in results:
            description = result.template.description
            if len(description) > 40:
                description = description[:37] + "..."
            
            table.add_row(
                result.template.name,
                result.template.stack.value,
                description,
                f"{result.score:.2f}"
            )
        
        self.console.print(table)
        self.console.print()
    
    def _display_project_preview(self, project_config: ProjectConfig) -> None:
        """
        Display project preview.
        
        Args:
            project_config: Project configuration
        """
        table = Table(title="Project Preview")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Name", project_config.name)
        table.add_row("Template", project_config.template.name)
        table.add_row("Stack", project_config.template.stack.value)
        table.add_row("Author", project_config.author or "N/A")
        table.add_row("License", project_config.license or "N/A")
        table.add_row("Package Manager", project_config.package_manager)
        table.add_row("Description", project_config.description or "N/A")
        table.add_row("Location", str(Path(project_config.directory) / project_config.name))
        
        self.console.print(table)
        self.console.print()
    
    def _display_next_steps(self, project_path: Path, project_config: ProjectConfig) -> None:
        """
        Display next steps for the user.
        
        Args:
            project_path: Path to the created project
            project_config: Project configuration
        """
        steps = [
            f"cd {project_path.name}",
        ]
        
        # Add installation step
        if project_config.package_manager in ['npm', 'yarn', 'pnpm']:
            steps.append(f"{project_config.package_manager} install")
        elif project_config.package_manager == 'pip':
            steps.append("pip install -r requirements.txt")
        elif project_config.package_manager == 'poetry':
            steps.append("poetry install")
        elif project_config.package_manager == 'cargo':
            steps.append("cargo build")
        elif project_config.package_manager == 'go':
            steps.append("go mod tidy")
        
        # Add development step
        template = project_config.template
        if any(tag in template.tags for tag in ['react', 'vue', 'nextjs', 'typescript']):
            steps.append("npm run dev")
        elif any(tag in template.tags for tag in ['python', 'fastapi', 'django']):
            steps.append("python app.py  # or python main.py")
        elif any(tag in template.tags for tag in ['rust']):
            steps.append("cargo run")
        elif any(tag in template.tags for tag in ['go']):
            steps.append("go run main.go")
        
        steps_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(steps))
        
        panel = Panel(
            Text(steps_text, style="white"),
            title="Next Steps",
            border_style="green"
        )
        
        self.console.print(panel)
