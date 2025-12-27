"""
Architecture questionnaire API routes for Template Heaven.

Provides endpoints for AI/LLM integration to auto-fill architecture questionnaires.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.models import APIResponse, ProjectConfig
from ...core.architecture_questionnaire import (
    ArchitectureQuestionnaire, ArchitectureAnswers,
    ArchitecturePattern, DeploymentModel, ScalabilityRequirement
)
from ..dependencies import get_settings, get_optional_user, get_request_id
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class QuestionnaireFillRequest(BaseModel):
    """Request model for AI/LLM questionnaire filling."""
    project_name: str = Field(..., min_length=1, description="Project name (required, cannot be empty)")
    project_description: Optional[str] = None
    template_stack: Optional[str] = None
    context: Optional[Dict[str, Any]] = None  # Additional context for LLM
    llm_provider: Optional[str] = None  # e.g., "openai", "anthropic", "custom"
    llm_model: Optional[str] = None  # e.g., "gpt-4", "claude-3-opus"
    llm_api_key: Optional[str] = None  # API key for LLM provider


class QuestionnaireFillResponse(BaseModel):
    """Response model for questionnaire filling."""
    success: bool
    answers: Dict[str, Any]
    confidence_scores: Optional[Dict[str, float]] = None
    warnings: Optional[List[str]] = None
    message: str


@router.post("/architecture/questionnaire/fill", response_model=APIResponse)
async def fill_questionnaire_with_ai(
    request: QuestionnaireFillRequest = Body(...),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """
    Fill architecture questionnaire using AI/LLM.
    
    This endpoint accepts project context and uses an LLM to intelligently
    fill out the architecture questionnaire based on best practices and
    the solution architecture patterns repository.
    
    Args:
        request: Questionnaire fill request with project context
        current_user: Optional authenticated user
        request_id: Request ID for tracking
        
    Returns:
        APIResponse with filled questionnaire answers
    """
    try:
        questionnaire = ArchitectureQuestionnaire()
        
        # Build context for LLM
        llm_context = {
            "project_name": request.project_name,
            "project_description": request.project_description or "",
            "template_stack": request.template_stack or "",
            "additional_context": request.context or {},
        }
        
        # If LLM provider is specified, call LLM
        if request.llm_provider:
            answers = await _call_llm_for_answers(
                questionnaire,
                llm_context,
                request.llm_provider,
                request.llm_model,
                request.llm_api_key
            )
        else:
            # Use rule-based intelligent defaults
            answers = _generate_intelligent_defaults(
                questionnaire,
                llm_context
            )
        
        # Validate answers
        is_valid, errors = questionnaire.validate_answers(answers.to_dict())
        
        if not is_valid:
            logger.warning(f"Generated answers have validation errors: {errors}")
        
        return APIResponse(
            success=True,
            message="Questionnaire filled successfully",
            data={
                "answers": answers.to_dict(),
                "validation_errors": errors if not is_valid else [],
                "warnings": [
                    "Review all answers carefully before proceeding",
                    "Some answers may need manual adjustment based on your specific requirements"
                ] if not is_valid else []
            },
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error filling questionnaire: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fill questionnaire: {str(e)}"
        )


@router.get("/architecture/questionnaire/structure", response_model=APIResponse)
async def get_questionnaire_structure(
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Get complete questionnaire structure."""
    try:
        questionnaire = ArchitectureQuestionnaire()
        questions = questionnaire.get_all_questions()
        
        questions_data = [
            {
                "key": q.key,
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "question_type": q.question_type,
                "required": q.required,
                "options": q.options,
                "choices": q.choices,
                "help_text": q.help_text,
                "default": q.default
            }
            for q in questions
        ]
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(questions_data)} questions",
            data={
                "questions": questions_data,
                "categories": list(set(q.category for q in questions))
            },
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Error retrieving questionnaire structure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve questionnaire structure: {str(e)}"
        )


@router.get("/architecture/questionnaire/category/{category}", response_model=APIResponse)
async def get_questionnaire_by_category(
    category: str,
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Get questions by category."""
    try:
        questionnaire = ArchitectureQuestionnaire()
        questions = questionnaire.get_questions_by_category(category)
        
        questions_data = [
            {
                "key": q.key,
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "question_type": q.question_type,
                "required": q.required,
                "options": q.options,
                "choices": q.choices,
                "help_text": q.help_text,
                "default": q.default
            }
            for q in questions
        ]
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(questions_data)} questions for category {category}",
            data={"questions": questions_data},
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Error retrieving questions by category: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve questions: {str(e)}"
        )


@router.get("/architecture/questionnaire/question/{key}", response_model=APIResponse)
async def get_question_by_key(
    key: str,
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Get a specific question by key."""
    try:
        questionnaire = ArchitectureQuestionnaire()
        question = questionnaire.get_question_by_key(key)
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question with key '{key}' not found"
            )
        
        question_data = {
            "key": question.key,
            "id": question.id,
            "category": question.category,
            "question": question.question,
            "question_type": question.question_type,
            "required": question.required,
            "options": question.options,
            "choices": question.choices,
            "help_text": question.help_text,
            "default": question.default
        }
        
        return APIResponse(
            success=True,
            message="Question retrieved successfully",
            data={"question": question_data},
            request_id=request_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve question: {str(e)}"
        )


@router.post("/architecture/questionnaire/validate", response_model=APIResponse)
async def validate_questionnaire_answers(
    answers: Dict[str, Any] = Body(...),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Validate questionnaire answers."""
    try:
        questionnaire = ArchitectureQuestionnaire()
        
        # Convert answers dict to ArchitectureAnswers object for validation
        try:
            arch_answers = ArchitectureAnswers.from_dict(answers.get("answers", answers))
            is_valid = arch_answers.validate()
            errors = [] if is_valid else ["Missing required fields"]
        except Exception as e:
            is_valid = False
            errors = [str(e)]
        
        # Also validate using questionnaire's validation
        is_valid_dict, validation_errors = questionnaire.validate_answers(answers.get("answers", answers))
        is_valid = is_valid and is_valid_dict
        errors.extend(validation_errors)
        
        return APIResponse(
            success=True,
            message="Validation completed",
            data={
                "valid": is_valid,
                "errors": errors if not is_valid else []
            },
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Error validating answers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate answers: {str(e)}"
        )


async def _call_llm_for_answers(
    questionnaire: ArchitectureQuestionnaire,
    context: Dict[str, Any],
    provider: str,
    model: Optional[str],
    api_key: Optional[str]
) -> ArchitectureAnswers:
    """
    Call LLM to generate questionnaire answers.
    
    Args:
        questionnaire: Questionnaire instance
        context: Project context
        provider: LLM provider name
        model: LLM model name
        api_key: API key for LLM
        
    Returns:
        ArchitectureAnswers object
    """
    # This is a placeholder - implement actual LLM integration
    # For now, fall back to intelligent defaults
    logger.info(f"LLM integration not yet implemented for provider: {provider}")
    return _generate_intelligent_defaults(questionnaire, context)


def _generate_intelligent_defaults(
    questionnaire: ArchitectureQuestionnaire,
    context: Dict[str, Any]
) -> ArchitectureAnswers:
    """
    Generate intelligent default answers based on context.
    
    Uses heuristics and best practices to fill questionnaire.
    
    Args:
        questionnaire: Questionnaire instance
        context: Project context
        
    Returns:
        ArchitectureAnswers with intelligent defaults
    """
    answers = ArchitectureAnswers()
    
    project_name = context.get("project_name", "")
    project_description = context.get("project_description", "")
    template_stack = context.get("template_stack", "")
    
    # Set basic defaults
    answers.project_vision = project_description or f"Build {project_name}"
    answers.target_users = "End users and API consumers"
    
    # Infer architecture pattern from stack
    if "microservices" in template_stack.lower():
        answers.architecture_pattern = ArchitecturePattern.MICROSERVICES
    elif "serverless" in template_stack.lower():
        answers.architecture_pattern = ArchitecturePattern.SERVERLESS
    elif "event" in template_stack.lower():
        answers.architecture_pattern = ArchitecturePattern.EVENT_DRIVEN
    else:
        answers.architecture_pattern = ArchitecturePattern.MONOLITH
    
    # Default deployment model
    answers.deployment_model = DeploymentModel.SINGLE_REGION
    answers.scalability_requirement = ScalabilityRequirement.MEDIUM
    
    # Default infrastructure
    answers.containerization = True
    answers.orchestration_platform = "kubernetes"
    answers.cloud_provider = "aws"
    
    # Default API design
    answers.api_style = "REST"
    answers.api_versioning_strategy = "url-path"
    answers.api_security_model = "JWT"
    answers.api_rate_limiting = True
    
    # Default observability
    answers.logging_strategy = "centralized"
    answers.monitoring_strategy = "prometheus"
    answers.alerting_strategy = "slack"
    
    # Default DevOps
    answers.ci_cd_strategy = "github-actions"
    answers.testing_strategy = ["unit", "integration"]
    answers.code_review_process = "mandatory-pr"
    answers.deployment_frequency = "daily"
    
    # Default features
    answers.feature_flags_required = True
    
    # Default data
    answers.data_volume = "TBD"
    answers.data_velocity = "real-time"
    answers.data_variety = "structured"
    answers.data_retention_policy = "7 years"
    answers.backup_strategy = "daily-snapshots"
    
    # Default security
    answers.security_requirements = ["authentication", "authorization", "encryption"]
    
    # Default team
    answers.team_size = 5
    answers.timeline = "3 months MVP"
    
    return answers


# ============================================================================
# Conversation Management Endpoints
# ============================================================================

class ConversationStartRequest(BaseModel):
    """Request to start a conversation."""
    project_name: str
    project_description: Optional[str] = None
    template_stack: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    llm_provider: Optional[str] = "openai"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None


class ConversationMessageRequest(BaseModel):
    """Request to send a message in conversation."""
    session_id: str
    message: str
    stream: bool = False


class ConversationAnalyzeReposRequest(BaseModel):
    """Request to analyze repositories."""
    session_id: str
    requirement: str
    technology_stack: Optional[List[str]] = None
    max_repos: int = 5


@router.post("/architecture/conversation/start", response_model=APIResponse)
async def start_conversation(
    request: ConversationStartRequest = Body(...),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Start a new system design conversation."""
    try:
        from ...core.llm import get_llm_provider, ConversationManager, SystemDesignAgent, SystemDesignContext
        
        # Get LLM provider
        llm_config = {
            "api_key": request.llm_api_key,
            "model": request.llm_model
        }
        llm_provider = get_llm_provider(request.llm_provider or "openai", llm_config)
        
        # Create conversation manager and agent
        conversation_manager = ConversationManager()
        agent = SystemDesignAgent(llm_provider, conversation_manager)
        
        # Create context
        context = SystemDesignContext(
            project_name=request.project_name,
            project_description=request.project_description,
            template_stack=request.template_stack,
            additional_context=request.context or {}
        )
        
        # Start conversation
        state = await agent.start_conversation(context)
        
        return APIResponse(
            success=True,
            message="Conversation started successfully",
            data={
                "session_id": state.session_id,
                "greeting": state.messages[-1]["content"] if state.messages else "",
                "conversation_state": state.to_dict()
            },
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start conversation: {str(e)}"
        )


@router.post("/architecture/conversation/message", response_model=APIResponse)
async def send_conversation_message(
    request: ConversationMessageRequest = Body(...),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Send a message in an active conversation."""
    try:
        from ...core.llm import ConversationManager, SystemDesignAgent
        from ...core.llm.providers import get_llm_provider
        
        # Get conversation manager
        conversation_manager = ConversationManager()
        state = conversation_manager.get_session(request.session_id)
        
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session {request.session_id} not found"
            )
        
        # Get LLM provider from context or use default
        llm_provider = get_llm_provider("openai", {})
        agent = SystemDesignAgent(llm_provider, conversation_manager)
        
        # Send message
        response = await agent.continue_conversation(
            request.session_id,
            request.message,
            stream=request.stream
        )
        
        if request.stream:
            # For streaming, we'd need WebSocket - return instruction for now
            return APIResponse(
                success=True,
                message="Use WebSocket endpoint for streaming",
                data={"session_id": request.session_id},
                request_id=request_id
            )
        else:
            return APIResponse(
                success=True,
                message="Message processed",
                data={
                    "session_id": request.session_id,
                    "response": response,
                    "conversation_state": state.to_dict()
                },
                request_id=request_id
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@router.get("/architecture/conversation/{session_id}", response_model=APIResponse)
async def get_conversation(
    session_id: str,
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Get conversation state and history."""
    try:
        from ...core.llm import ConversationManager
        
        conversation_manager = ConversationManager()
        state = conversation_manager.get_session(session_id)
        
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session {session_id} not found"
            )
        
        return APIResponse(
            success=True,
            message="Conversation retrieved",
            data=state.to_dict(),
            request_id=request_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@router.post("/architecture/conversation/analyze-repos", response_model=APIResponse)
async def analyze_repositories(
    request: ConversationAnalyzeReposRequest = Body(...),
    current_user: Optional[Any] = Depends(get_optional_user),
    request_id: str = Depends(get_request_id)
):
    """Analyze repositories based on requirements and provide integration recommendations."""
    try:
        from ...core.llm import ConversationManager, SystemDesignAgent
        from ...core.llm.providers import get_llm_provider
        from ...core.github_client import GitHubClient
        from ...core.repo_analyzer import RepositoryAnalyzer, IntegrationRecommender
        
        # Get conversation state
        conversation_manager = ConversationManager()
        state = conversation_manager.get_session(request.session_id)
        
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation session {request.session_id} not found"
            )
        
        # Search for repositories and analyze
        github_client = GitHubClient()
        analyzer = RepositoryAnalyzer()
        recommender = IntegrationRecommender()
        
        requirements = {
            "use_case": request.requirement,
            "technology_stack": request.technology_stack or []
        }
        
        recommendations = []
        async with github_client:
            # Search repositories
            repos = await github_client.search_repositories(
                query=request.requirement,
                language=request.technology_stack[0] if request.technology_stack else None,
                min_stars=50,
                per_page=request.max_repos
            )
            
            # Analyze repositories (keep session open for fetching code if needed)
            for repo in repos[:request.max_repos]:
                try:
                    owner = repo["owner"]["login"]
                    repo_name = repo["name"]
                    
                    # Fetch repository contents for analysis
                    code_contents = await github_client.get_repository_contents(
                        owner, repo_name, max_files=20
                    )
                    
                    # Perform deep analysis
                    analysis = await analyzer.analyze_repository(
                        repo_url=repo["html_url"],
                        repo_data=repo,
                        code_contents=code_contents if code_contents else None
                    )
                
                # Generate recommendation
                recommendation = recommender.recommend_integration(
                    analysis,
                    requirements,
                    repo
                )
                
                recommendations.append(recommendation.to_dict())
            except Exception as e:
                logger.warning(f"Failed to analyze repository {repo.get('full_name')}: {e}")
                continue
        
        # Add recommendations to conversation
        for rec in recommendations:
            conversation_manager.add_recommendation(
                request.session_id,
                "repository",
                rec
            )
        
        return APIResponse(
            success=True,
            message=f"Analyzed {len(recommendations)} repositories",
            data={
                "session_id": request.session_id,
                "recommendations": recommendations,
                "count": len(recommendations)
            },
            request_id=request_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing repositories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze repositories: {str(e)}"
        )

