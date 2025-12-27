"""
Tests for LLM integration and async conversation flow.

Tests the multi-turn LLM conversation system, repository analysis,
and integration recommendations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Optional, Union, AsyncIterator, Any

from templateheaven.core.llm import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    get_llm_provider,
    ConversationManager,
    ConversationState,
    SystemDesignAgent,
    SystemDesignContext,
)
from templateheaven.core.architecture_questionnaire import (
    ArchitectureQuestionnaire,
    ArchitectureAnswers,
)
from templateheaven.core.repo_analyzer import (
    RepositoryAnalyzer,
    IntegrationRecommender,
)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.responses = []
        self.call_count = 0
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Return mock response."""
        self.call_count += 1
        if self.responses:
            response = self.responses.pop(0)
        else:
            response = f"Mock response {self.call_count}"
        
        if stream:
            async def stream_response():
                for char in response:
                    yield char
            return stream_response()
        return response
    
    def supports_streaming(self) -> bool:
        return True
    
    def supports_tools(self) -> bool:
        return True


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider(api_key="test-key", model="test-model")


@pytest.fixture
def conversation_manager():
    """Create a conversation manager."""
    return ConversationManager()


@pytest.fixture
def architecture_questionnaire():
    """Create an architecture questionnaire."""
    return ArchitectureQuestionnaire(quick_mode=True)


@pytest.fixture
def system_design_agent(mock_llm_provider, conversation_manager, architecture_questionnaire):
    """Create a system design agent."""
    return SystemDesignAgent(
        mock_llm_provider,
        conversation_manager,
        architecture_questionnaire
    )


@pytest.mark.asyncio
async def test_conversation_manager_create_session():
    """Test creating a conversation session."""
    manager = ConversationManager()
    
    context = {"project_name": "test-project"}
    state = manager.create_session(context)
    
    assert state is not None
    assert state.session_id is not None
    assert state.context == context
    assert state.status.value == "active"


@pytest.mark.asyncio
async def test_conversation_manager_add_message():
    """Test adding messages to conversation."""
    manager = ConversationManager()
    state = manager.create_session()
    
    manager.add_message(state.session_id, "user", "Hello")
    manager.add_message(state.session_id, "assistant", "Hi there!")
    
    history = manager.get_conversation_history(state.session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there!"


@pytest.mark.asyncio
async def test_system_design_agent_start_conversation(mock_llm_provider, conversation_manager, architecture_questionnaire):
    """Test starting a conversation with system design agent."""
    # Set up mock response
    mock_llm_provider.responses = ["Hello! I'm here to help you design the architecture for **test-project**."]
    
    agent = SystemDesignAgent(mock_llm_provider, conversation_manager, architecture_questionnaire)
    
    context = SystemDesignContext(
        project_name="test-project",
        project_description="A test project"
    )
    
    state = await agent.start_conversation(context)
    
    assert state is not None
    assert state.session_id is not None
    assert len(state.messages) >= 2  # System prompt + greeting
    assert mock_llm_provider.call_count > 0


@pytest.mark.asyncio
async def test_system_design_agent_continue_conversation(system_design_agent):
    """Test continuing a conversation."""
    # Set up initial conversation
    context = SystemDesignContext(project_name="test-project")
    state = await system_design_agent.start_conversation(context)
    
    # Set up mock response
    system_design_agent.llm_provider.responses = ["That's a great question! Let me help you with that."]
    
    # Continue conversation
    response = await system_design_agent.continue_conversation(
        state.session_id,
        "What architecture pattern should I use?",
        stream=False
    )
    
    assert response is not None
    assert len(response) > 0
    
    # Check conversation history
    updated_state = system_design_agent.conversation_manager.get_session(state.session_id)
    assert len(updated_state.messages) >= 4  # System, greeting, user, assistant


@pytest.mark.asyncio
async def test_system_design_agent_ask_question(system_design_agent, architecture_questionnaire):
    """Test asking a specific architecture question."""
    # Set up initial conversation
    context = SystemDesignContext(project_name="test-project")
    state = await system_design_agent.start_conversation(context)
    
    # Get a question
    questions = architecture_questionnaire.get_all_questions()
    if questions:
        question = questions[0]
        
        # Set up mock response
        system_design_agent.llm_provider.responses = [
            f"Let me ask you about {question.question} in a conversational way."
        ]
        
        # Ask question
        question_text = await system_design_agent.ask_question(
            state.session_id,
            question,
            {"project_name": "test-project"}
        )
        
        assert question_text is not None
        assert len(question_text) > 0


@pytest.mark.asyncio
async def test_repository_analyzer_analyze_repository():
    """Test repository analysis."""
    analyzer = RepositoryAnalyzer()
    
    repo_data = {
        "name": "test-repo",
        "stargazers_count": 100,
        "forks_count": 20,
        "language": "Python",
        "description": "A test repository",
        "topics": ["python", "api"],
        "license": {"name": "MIT"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T00:00:00Z",
    }
    
    code_contents = {
        "package.json": '{"name": "test", "dependencies": {"express": "^4.0"}}',
        "README.md": "# Test Repository",
        "src/index.js": "console.log('Hello');",
    }
    
    analysis = await analyzer.analyze_repository(
        repo_url="https://github.com/test/test-repo",
        repo_data=repo_data,
        code_contents=code_contents
    )
    
    assert analysis.repository_name == "test/test-repo"
    assert analysis.technology_stack is not None
    # Check for JavaScript/TypeScript or just JavaScript
    assert any("JavaScript" in tech or "TypeScript" in tech for tech in analysis.technology_stack)
    assert len(analysis.dependencies) > 0
    assert "package.json" in analysis.dependencies


@pytest.mark.asyncio
async def test_integration_recommender_recommend_integration():
    """Test integration recommendation."""
    from templateheaven.core.repo_analyzer.code_analyzer import RepositoryAnalysis
    
    recommender = IntegrationRecommender()
    
    # Create mock analysis
    analysis = RepositoryAnalysis(
        repository_url="https://github.com/test/express-rate-limit",
        repository_name="test/express-rate-limit",
        technology_stack=["JavaScript/TypeScript", "Express.js"],
        dependencies={"package.json": ["express@^4.0", "express-rate-limit@^6.0"]},
        metadata={
            "stars": 2500,
            "license": "MIT",
            "description": "Rate limiting middleware for Express"
        }
    )
    
    requirements = {
        "use_case": "API rate limiting",
        "technology_stack": ["JavaScript/TypeScript"]
    }
    
    recommendation = recommender.recommend_integration(
        analysis,
        requirements,
        {"name": "express-rate-limit", "html_url": "https://github.com/test/express-rate-limit"}
    )
    
    assert recommendation is not None
    assert recommendation.relevance_score > 0
    assert recommendation.compatibility > 0
    assert recommendation.integration_approach is not None
    assert len(recommendation.integration_steps) > 0


@pytest.mark.asyncio
async def test_async_flow_integration(mock_llm_provider, conversation_manager, architecture_questionnaire):
    """Test complete async flow from conversation start to recommendation."""
    # Set up mock responses
    mock_llm_provider.responses = [
        "Hello! I'm here to help you design the architecture.",
        "That's a great requirement. Let me suggest some solutions.",
        "Based on your needs, I recommend microservices architecture."
    ]
    
    agent = SystemDesignAgent(mock_llm_provider, conversation_manager, architecture_questionnaire)
    
    # Start conversation
    context = SystemDesignContext(
        project_name="test-project",
        project_description="A scalable API service"
    )
    state = await agent.start_conversation(context)
    
    # Continue conversation
    response1 = await agent.continue_conversation(
        state.session_id,
        "I need to handle high traffic",
        stream=False
    )
    assert response1 is not None
    
    # Suggest architecture pattern
    requirements = {
        "high_scale": True,
        "large_team": False
    }
    recommendation = await agent.suggest_architecture_pattern(state.session_id, requirements)
    assert recommendation is not None
    assert "suggestion" in recommendation
    
    # Verify conversation state
    final_state = conversation_manager.get_session(state.session_id)
    assert len(final_state.messages) >= 4
    assert len(final_state.architecture_recommendations) > 0


@pytest.mark.asyncio
async def test_llm_provider_factory():
    """Test LLM provider factory function."""
    # Test OpenAI provider creation
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        provider = get_llm_provider("openai", {"model": "gpt-4"})
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4"
    
    # Test Anthropic provider creation
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
        provider = get_llm_provider("anthropic", {"model": "claude-3-opus"})
        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-opus"


def test_conversation_state_serialization():
    """Test conversation state serialization."""
    state = ConversationState(session_id="test-session")
    state.add_message("user", "Hello")
    state.add_recommendation("architecture", {"pattern": "microservices"})
    
    # Convert to dict
    state_dict = state.to_dict()
    assert state_dict["session_id"] == "test-session"
    assert len(state_dict["messages"]) == 1
    assert len(state_dict["architecture_recommendations"]) == 1
    
    # Recreate from dict
    new_state = ConversationState.from_dict(state_dict)
    assert new_state.session_id == "test-session"
    assert len(new_state.messages) == 1
    assert len(new_state.architecture_recommendations) == 1

