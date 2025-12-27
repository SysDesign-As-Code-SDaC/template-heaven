"""
System Design Agent for Template Heaven.

Specialized LLM agent for system design consultation, architecture pattern
recommendations, and integration with architecture questionnaires.
"""

from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass

from .providers import LLMProvider
from .conversation import ConversationManager, ConversationState
from ..architecture_questionnaire import ArchitectureQuestionnaire, ArchitectureQuestion
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemDesignContext:
    """Context for system design consultation."""
    project_name: str
    project_description: Optional[str] = None
    template_stack: Optional[str] = None
    current_answers: Dict[str, Any] = None
    additional_context: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.current_answers is None:
            self.current_answers = {}
        if self.additional_context is None:
            self.additional_context = {}


class SystemDesignAgent:
    """Specialized agent for system design consultation."""
    
    SYSTEM_PROMPT = """You are an expert system design consultant helping users design robust, scalable software architectures. Your role is to:

1. Ask thoughtful, probing questions to understand the user's requirements
2. Suggest appropriate architecture patterns based on their needs
3. Recommend existing open-source solutions when applicable
4. Guide users through the system design process in a conversational manner
5. Explain the reasoning behind your recommendations

You should be:
- Conversational and approachable
- Thorough in understanding requirements
- Practical in your recommendations
- Educational in explaining concepts
- Proactive in suggesting alternatives

When you identify requirements that might have existing open-source solutions, you should:
- Suggest searching for relevant repositories
- Analyze potential integrations
- Provide pros/cons of different options
- Help users make informed decisions

Focus on understanding the user's specific needs before making recommendations. Ask follow-up questions when answers are ambiguous or incomplete."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        conversation_manager: ConversationManager,
        questionnaire: Optional[ArchitectureQuestionnaire] = None
    ):
        """
        Initialize system design agent.
        
        Args:
            llm_provider: LLM provider instance
            conversation_manager: Conversation manager instance
            questionnaire: Optional architecture questionnaire for structure
        """
        self.llm_provider = llm_provider
        self.conversation_manager = conversation_manager
        self.questionnaire = questionnaire or ArchitectureQuestionnaire()
        logger.info("SystemDesignAgent initialized")
    
    async def start_conversation(
        self,
        context: SystemDesignContext
    ) -> ConversationState:
        """
        Start a new system design conversation.
        
        Args:
            context: System design context
            
        Returns:
            ConversationState for the new session
        """
        # Create conversation session
        initial_context = {
            "project_name": context.project_name,
            "project_description": context.project_description,
            "template_stack": context.template_stack,
            "current_answers": context.current_answers,
            **context.additional_context
        }
        
        state = self.conversation_manager.create_session(initial_context)
        
        # Add system prompt
        self.conversation_manager.add_message(
            state.session_id,
            "system",
            self.SYSTEM_PROMPT
        )
        
        # Generate initial greeting and first question
        greeting = await self._generate_greeting(context)
        self.conversation_manager.add_message(
            state.session_id,
            "assistant",
            greeting
        )
        
        logger.info(f"Started system design conversation: {state.session_id}")
        return state
    
    async def _generate_greeting(self, context: SystemDesignContext) -> str:
        """Generate initial greeting message."""
        project_info = f"**{context.project_name}**"
        if context.project_description:
            project_info += f": {context.project_description}"
        
        greeting = f"""Hello! I'm here to help you design the architecture for {project_info}.

I'll guide you through a series of questions to understand your requirements, and then suggest the best architecture patterns and potentially existing open-source solutions that could help.

Let's start with understanding your project's core purpose and goals."""
        
        return greeting
    
    async def continue_conversation(
        self,
        session_id: str,
        user_message: str,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Continue conversation with user message.
        
        Args:
            session_id: Conversation session ID
            user_message: User's message
            stream: Whether to stream the response
            
        Returns:
            Assistant response or async iterator for streaming
        """
        state = self.conversation_manager.get_session(session_id)
        if not state:
            raise ValueError(f"Conversation session {session_id} not found")
        
        # Add user message
        self.conversation_manager.add_message(session_id, "user", user_message)
        
        # Get conversation history
        messages = self.conversation_manager.get_conversation_history(session_id)
        
        # Generate response
        response = await self.llm_provider.chat_completion(
            messages=messages,
            stream=stream
        )
        
        if stream:
            # For streaming, we need to collect chunks and add to conversation
            async def stream_with_save():
                full_response = ""
                async for chunk in response:
                    full_response += chunk
                    yield chunk
                
                # Save complete response to conversation
                self.conversation_manager.add_message(
                    session_id,
                    "assistant",
                    full_response
                )
            
            return stream_with_save()
        else:
            # Add assistant response to conversation
            response_text = response if isinstance(response, str) else response.get("content", "")
            self.conversation_manager.add_message(session_id, "assistant", response_text)
            return response_text
    
    async def ask_question(
        self,
        session_id: str,
        question: ArchitectureQuestion,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ask a specific architecture question.
        
        Args:
            session_id: Conversation session ID
            question: Architecture question to ask
            context: Optional additional context
            
        Returns:
            Formatted question message
        """
        state = self.conversation_manager.get_session(session_id)
        if not state:
            raise ValueError(f"Conversation session {session_id} not found")
        
        # Build question message
        question_text = question.question
        if question.help_text:
            question_text += f"\n\n💡 {question.help_text}"
        
        if question.options:
            question_text += f"\n\nOptions: {', '.join(question.options)}"
        
        # Add context if provided
        if context:
            context_str = ", ".join(f"{k}: {v}" for k, v in context.items())
            question_text += f"\n\nContext: {context_str}"
        
        # Ask question via LLM for natural phrasing
        messages = self.conversation_manager.get_conversation_history(session_id)
        messages.append({
            "role": "user",
            "content": f"Ask the user this question in a conversational way: {question.question}"
        })
        
        response = await self.llm_provider.chat_completion(messages=messages)
        natural_question = response if isinstance(response, str) else response.get("content", question_text)
        
        # Add to conversation
        self.conversation_manager.add_message(session_id, "assistant", natural_question)
        self.conversation_manager.update_session(session_id, {
            "current_question_category": question.category
        })
        
        return natural_question
    
    async def suggest_architecture_pattern(
        self,
        session_id: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest architecture patterns based on requirements.
        
        Args:
            session_id: Conversation session ID
            requirements: Project requirements
            
        Returns:
            Dictionary with architecture recommendations
        """
        state = self.conversation_manager.get_session(session_id)
        if not state:
            raise ValueError(f"Conversation session {session_id} not found")
        
        # Build prompt for architecture suggestion
        requirements_text = "\n".join(f"- {k}: {v}" for k, v in requirements.items())
        
        prompt = f"""Based on these requirements, suggest appropriate architecture patterns:

{requirements_text}

Consider:
1. Scalability needs
2. Team size and experience
3. Deployment constraints
4. Performance requirements
5. Maintenance complexity

Provide:
- Recommended architecture pattern(s)
- Reasoning for the recommendation
- Trade-offs and alternatives
- When to use each pattern"""
        
        messages = self.conversation_manager.get_conversation_history(session_id)
        messages.append({"role": "user", "content": prompt})
        
        response = await self.llm_provider.chat_completion(messages=messages)
        suggestion_text = response if isinstance(response, str) else response.get("content", "")
        
        # Parse and structure recommendation
        recommendation = {
            "suggestion": suggestion_text,
            "requirements": requirements,
            "timestamp": state.updated_at.isoformat()
        }
        
        # Add to conversation
        self.conversation_manager.add_recommendation(
            session_id,
            "architecture",
            recommendation
        )
        
        # Add to conversation messages
        self.conversation_manager.add_message(
            session_id,
            "assistant",
            f"Based on your requirements, I recommend:\n\n{suggestion_text}"
        )
        
        return recommendation
    
    async def suggest_repository_search(
        self,
        session_id: str,
        requirement: str,
        technology_stack: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Suggest searching for open-source repositories.
        
        Args:
            session_id: Conversation session ID
            requirement: Requirement that might have existing solutions
            technology_stack: Optional technology stack filters
            
        Returns:
            Dictionary with search suggestions
        """
        state = self.conversation_manager.get_session(session_id)
        if not state:
            raise ValueError(f"Conversation session {session_id} not found")
        
        stack_text = ""
        if technology_stack:
            stack_text = f" Technology stack: {', '.join(technology_stack)}"
        
        prompt = f"""The user needs: {requirement}{stack_text}

Suggest:
1. What to search for on GitHub
2. Key terms and technologies
3. What to look for in repositories (features, architecture, etc.)
4. Integration considerations

Provide a structured search strategy."""
        
        messages = self.conversation_manager.get_conversation_history(session_id)
        messages.append({"role": "user", "content": prompt})
        
        response = await self.llm_provider.chat_completion(messages=messages)
        suggestion_text = response if isinstance(response, str) else response.get("content", "")
        
        recommendation = {
            "requirement": requirement,
            "search_strategy": suggestion_text,
            "technology_stack": technology_stack,
            "timestamp": state.updated_at.isoformat()
        }
        
        # Add to conversation
        self.conversation_manager.add_recommendation(
            session_id,
            "repository",
            recommendation
        )
        
        # Add to conversation messages
        self.conversation_manager.add_message(
            session_id,
            "assistant",
            f"For {requirement}, I suggest searching for existing solutions:\n\n{suggestion_text}"
        )
        
        return recommendation
    
    def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary of conversation.
        
        Args:
            session_id: Conversation session ID
            
        Returns:
            Summary dictionary or None if session not found
        """
        state = self.conversation_manager.get_session(session_id)
        if not state:
            return None
        
        return {
            "session_id": session_id,
            "total_messages": len(state.messages),
            "answered_questions": len(state.answered_questions),
            "architecture_recommendations": len(state.architecture_recommendations),
            "repo_recommendations": len(state.repo_recommendations),
            "status": state.status.value,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat()
        }

