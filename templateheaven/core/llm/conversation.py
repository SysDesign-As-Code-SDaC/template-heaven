"""
Conversation management for multi-turn LLM dialogues.

Manages conversation state, history, and context for system design consultations.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum

from ...utils.logger import get_logger

logger = get_logger(__name__)


class ConversationStatus(Enum):
    """Conversation status states."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ConversationState:
    """State container for a conversation session."""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_question_category: Optional[str] = None
    answered_questions: Set[str] = field(default_factory=set)
    pending_questions: List[str] = field(default_factory=list)
    architecture_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    repo_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: ConversationStatus = ConversationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.updated_at = datetime.now()
        logger.debug(f"Added {role} message to conversation {self.session_id}")
    
    def mark_question_answered(self, question_id: str) -> None:
        """Mark a question as answered."""
        self.answered_questions.add(question_id)
        if question_id in self.pending_questions:
            self.pending_questions.remove(question_id)
    
    def add_recommendation(self, recommendation_type: str, recommendation: Dict[str, Any]) -> None:
        """Add an architecture or repository recommendation."""
        recommendation["type"] = recommendation_type
        recommendation["timestamp"] = datetime.now().isoformat()
        
        if recommendation_type == "architecture":
            self.architecture_recommendations.append(recommendation)
        elif recommendation_type == "repository":
            self.repo_recommendations.append(recommendation)
        
        logger.debug(f"Added {recommendation_type} recommendation to conversation {self.session_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation state to dictionary."""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "current_question_category": self.current_question_category,
            "answered_questions": list(self.answered_questions),
            "pending_questions": self.pending_questions,
            "architecture_recommendations": self.architecture_recommendations,
            "repo_recommendations": self.repo_recommendations,
            "context": self.context,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create ConversationState from dictionary."""
        state = cls(
            session_id=data["session_id"],
            messages=data.get("messages", []),
            current_question_category=data.get("current_question_category"),
            answered_questions=set(data.get("answered_questions", [])),
            pending_questions=data.get("pending_questions", []),
            architecture_recommendations=data.get("architecture_recommendations", []),
            repo_recommendations=data.get("repo_recommendations", []),
            context=data.get("context", {}),
            status=ConversationStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        )
        return state


class ConversationManager:
    """Manages conversation sessions and state."""
    
    def __init__(self):
        """Initialize conversation manager."""
        self.sessions: Dict[str, ConversationState] = {}
        logger.info("ConversationManager initialized")
    
    def create_session(
        self,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ConversationState:
        """
        Create a new conversation session.
        
        Args:
            initial_context: Optional initial context for the conversation
            
        Returns:
            New ConversationState instance
        """
        session_id = str(uuid.uuid4())
        state = ConversationState(
            session_id=session_id,
            context=initial_context or {}
        )
        
        self.sessions[session_id] = state
        logger.info(f"Created conversation session: {session_id}")
        
        return state
    
    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """
        Get conversation session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationState or None if not found
        """
        return self.sessions.get(session_id)
    
    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ConversationState]:
        """
        Update conversation session.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
            
        Returns:
            Updated ConversationState or None if not found
        """
        state = self.get_session(session_id)
        if not state:
            return None
        
        # Update context
        if "context" in updates:
            state.context.update(updates["context"])
        
        # Update status
        if "status" in updates:
            state.status = ConversationStatus(updates["status"])
        
        # Update current category
        if "current_question_category" in updates:
            state.current_question_category = updates["current_question_category"]
        
        state.updated_at = datetime.now()
        logger.debug(f"Updated conversation session: {session_id}")
        
        return state
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> Optional[ConversationState]:
        """
        Add message to conversation.
        
        Args:
            session_id: Session identifier
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            
        Returns:
            Updated ConversationState or None if not found
        """
        state = self.get_session(session_id)
        if not state:
            return None
        
        state.add_message(role, content)
        return state
    
    def mark_question_answered(
        self,
        session_id: str,
        question_id: str
    ) -> Optional[ConversationState]:
        """
        Mark a question as answered.
        
        Args:
            session_id: Session identifier
            question_id: Question identifier
            
        Returns:
            Updated ConversationState or None if not found
        """
        state = self.get_session(session_id)
        if not state:
            return None
        
        state.mark_question_answered(question_id)
        return state
    
    def add_recommendation(
        self,
        session_id: str,
        recommendation_type: str,
        recommendation: Dict[str, Any]
    ) -> Optional[ConversationState]:
        """
        Add recommendation to conversation.
        
        Args:
            session_id: Session identifier
            recommendation_type: Type of recommendation ('architecture' or 'repository')
            recommendation: Recommendation data
            
        Returns:
            Updated ConversationState or None if not found
        """
        state = self.get_session(session_id)
        if not state:
            return None
        
        state.add_recommendation(recommendation_type, recommendation)
        return state
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> Optional[List[Dict[str, str]]]:
        """
        Get conversation message history.
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages to return
            
        Returns:
            List of messages or None if session not found
        """
        state = self.get_session(session_id)
        if not state:
            return None
        
        messages = state.messages
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def export_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export full conversation state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with full conversation state or None if not found
        """
        state = self.get_session(session_id)
        if not state:
            return None
        
        return state.to_dict()
    
    def close_session(self, session_id: str) -> bool:
        """
        Close a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was closed, False if not found
        """
        state = self.get_session(session_id)
        if not state:
            return False
        
        state.status = ConversationStatus.COMPLETED
        state.updated_at = datetime.now()
        logger.info(f"Closed conversation session: {session_id}")
        
        return True
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old conversation sessions.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned = 0
        
        for session_id, state in list(self.sessions.items()):
            if state.updated_at < cutoff_time:
                del self.sessions[session_id]
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old conversation sessions")
        
        return cleaned

