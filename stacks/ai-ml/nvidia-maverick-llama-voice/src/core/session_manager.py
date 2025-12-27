"""
NVIDIA Maverick Session Manager

This module provides conversation session management following NVIDIA's Maverick
architecture patterns for maintaining context in voice-enabled LLM interactions.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib

from .inference_engine import VoiceInferenceContext, InferenceMode
from .voice_processor import VoiceProcessingResult


@dataclass
class ConversationMessage:
    """Represents a message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SessionContext:
    """Context information for a conversation session."""
    session_id: str
    user_id: Optional[str] = None
    language: str = "en-US"
    personality: str = "neutral"
    conversation_style: str = "casual"
    memory_enabled: bool = True
    max_history_length: int = 50
    session_timeout: int = 3600  # 1 hour
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionState:
    """Current state of a conversation session."""
    context: SessionContext
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    voice_context: VoiceInferenceContext = field(default_factory=VoiceInferenceContext)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary."""
        return {
            "context": {
                "session_id": self.context.session_id,
                "user_id": self.context.user_id,
                "language": self.context.language,
                "personality": self.context.personality,
                "conversation_style": self.context.conversation_style,
                "memory_enabled": self.context.memory_enabled,
                "max_history_length": self.context.max_history_length,
                "session_timeout": self.context.session_timeout,
                "created_at": self.context.created_at.isoformat(),
                "last_activity": self.context.last_activity.isoformat()
            },
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "voice_context": {
                "conversation_history": self.voice_context.conversation_history,
                "emotion_state": self.voice_context.emotion_state,
                "language": self.voice_context.language,
                "streaming_enabled": self.voice_context.streaming_enabled,
                "voice_response_required": self.voice_context.voice_response_required
            },
            "preferences": self.preferences,
            "metadata": self.metadata
        }


class SessionManager:
    """
    Manages conversation sessions following NVIDIA Maverick patterns.

    This class provides:
    - Session lifecycle management
    - Conversation history tracking
    - Context preservation across interactions
    - Memory management and optimization
    - Session persistence and recovery
    """

    def __init__(self, storage_path: Optional[str] = None, enable_persistence: bool = True):
        self.storage_path = storage_path or "./sessions"
        self.enable_persistence = enable_persistence
        self.sessions: Dict[str, SessionState] = {}
        self.session_lock = asyncio.Lock()

        # Session cleanup
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300  # 5 minutes

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize storage
        if self.enable_persistence:
            self._initialize_storage()

        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._session_cleanup_worker())

    def _initialize_storage(self):
        """Initialize session storage."""
        import os
        os.makedirs(self.storage_path, exist_ok=True)

    async def create_session(self, user_id: Optional[str] = None,
                           language: str = "en-US",
                           personality: str = "neutral") -> str:
        """
        Create a new conversation session.

        Args:
            user_id: Optional user identifier
            language: Session language
            personality: AI personality for the session

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())

        context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            language=language,
            personality=personality
        )

        session_state = SessionState(context=context)

        # Initialize with system prompt based on personality
        system_prompt = self._get_system_prompt(personality)
        session_state.conversation_history.append(
            ConversationMessage(role="system", content=system_prompt)
        )

        async with self.session_lock:
            self.sessions[session_id] = session_state

        # Persist if enabled
        if self.enable_persistence:
            await self._persist_session(session_state)

        self.logger.info(f"Created session: {session_id} for user: {user_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session state if found, None otherwise
        """
        async with self.session_lock:
            session = self.sessions.get(session_id)

            if session:
                # Update last activity
                session.context.last_activity = datetime.utcnow()
                return session

            # Try to load from storage
            if self.enable_persistence:
                session = await self._load_session(session_id)
                if session:
                    self.sessions[session_id] = session
                    return session

        return None

    async def update_session(self, session_id: str, updates: Dict[str, Any]):
        """
        Update session properties.

        Args:
            session_id: Session identifier
            updates: Properties to update
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Update context
        context_updates = updates.get("context", {})
        for key, value in context_updates.items():
            if hasattr(session.context, key):
                setattr(session.context, key, value)

        # Update preferences
        preferences_updates = updates.get("preferences", {})
        session.preferences.update(preferences_updates)

        # Update metadata
        metadata_updates = updates.get("metadata", {})
        session.metadata.update(metadata_updates)

        # Persist changes
        if self.enable_persistence:
            await self._persist_session(session)

    async def add_message(self, session_id: str, role: str, content: str,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to the conversation history.

        Args:
            session_id: Session identifier
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Optional message metadata
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )

        session.conversation_history.append(message)

        # Trim history if needed
        if len(session.conversation_history) > session.context.max_history_length:
            # Keep system message and recent messages
            system_messages = [msg for msg in session.conversation_history if msg.role == "system"]
            recent_messages = session.conversation_history[-session.context.max_history_length + len(system_messages):]

            if system_messages:
                session.conversation_history = system_messages + recent_messages
            else:
                session.conversation_history = recent_messages

        # Update voice context
        session.voice_context.conversation_history.append({
            "role": role,
            "content": content
        })

        # Persist changes
        if self.enable_persistence:
            await self._persist_session(session)

    async def process_message(self, session_id: str, user_input: str,
                            voice_result: Optional[VoiceProcessingResult] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Process a user message in the context of a session.

        Args:
            session_id: Session identifier
            user_input: User input text
            voice_result: Optional voice processing results
            **kwargs: Additional processing parameters

        Returns:
            Processing results including response and metadata
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Add user message
        await self.add_message(session_id, "user", user_input)

        # Update voice context if voice data available
        if voice_result:
            session.voice_context.emotion_state = voice_result.emotion.value
            session.voice_context.language = voice_result.language
            if voice_result.voice_features:
                session.voice_context.voice_features = voice_result.voice_features

        # Prepare inference context
        inference_context = self._prepare_inference_context(session, **kwargs)

        # Generate response (this would call the inference engine)
        response = await self._generate_response(inference_context)

        # Add assistant response
        await self.add_message(session_id, "assistant", response["text"])

        # Update session metadata
        session.metadata["last_interaction"] = datetime.utcnow().isoformat()
        session.metadata["total_messages"] = len(session.conversation_history)

        return {
            "response": response["text"],
            "voice_response": response.get("voice_audio"),
            "emotion": response.get("emotion", "neutral"),
            "confidence": response.get("confidence", 0.0),
            "session_id": session_id,
            "processing_time": response.get("processing_time", 0.0)
        }

    async def end_session(self, session_id: str):
        """
        End a conversation session.

        Args:
            session_id: Session identifier
        """
        session = await self.get_session(session_id)
        if session:
            # Add session summary to metadata
            session.metadata["ended_at"] = datetime.utcnow().isoformat()
            session.metadata["total_messages"] = len(session.conversation_history)
            session.metadata["duration"] = (
                datetime.utcnow() - session.context.created_at
            ).total_seconds()

            # Final persistence
            if self.enable_persistence:
                await self._persist_session(session)

            # Remove from memory
            async with self.session_lock:
                del self.sessions[session_id]

            self.logger.info(f"Ended session: {session_id}")

    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List active sessions.

        Args:
            user_id: Optional user ID filter

        Returns:
            List of session information
        """
        sessions_info = []

        async with self.session_lock:
            for session_id, session in self.sessions.items():
                if user_id and session.context.user_id != user_id:
                    continue

                sessions_info.append({
                    "session_id": session_id,
                    "user_id": session.context.user_id,
                    "created_at": session.context.created_at.isoformat(),
                    "last_activity": session.context.last_activity.isoformat(),
                    "message_count": len(session.conversation_history),
                    "language": session.context.language,
                    "personality": session.context.personality
                })

        return sessions_info

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary if found
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        # Calculate conversation statistics
        user_messages = len([msg for msg in session.conversation_history if msg.role == "user"])
        assistant_messages = len([msg for msg in session.conversation_history if msg.role == "assistant"])

        return {
            "session_id": session_id,
            "user_id": session.context.user_id,
            "created_at": session.context.created_at.isoformat(),
            "duration": (datetime.utcnow() - session.context.created_at).total_seconds(),
            "total_messages": len(session.conversation_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "language": session.context.language,
            "personality": session.context.personality,
            "conversation_style": session.context.conversation_style,
            "last_activity": session.context.last_activity.isoformat()
        }

    def _get_system_prompt(self, personality: str) -> str:
        """Get system prompt based on personality."""
        prompts = {
            "neutral": "You are a helpful AI assistant. Provide accurate and balanced responses.",
            "friendly": "You are a friendly and approachable AI assistant. Be warm and engaging in your responses.",
            "professional": "You are a professional AI assistant. Provide clear, concise, and business-appropriate responses.",
            "creative": "You are a creative AI assistant. Provide imaginative and innovative responses.",
            "technical": "You are a technical AI assistant. Focus on accuracy, detail, and technical precision.",
            "educational": "You are an educational AI assistant. Provide informative responses with learning focus."
        }

        return prompts.get(personality, prompts["neutral"])

    def _prepare_inference_context(self, session: SessionState, **kwargs) -> Dict[str, Any]:
        """Prepare context for inference."""
        # Extract recent conversation history
        recent_history = session.conversation_history[-10:]  # Last 10 messages

        context = {
            "conversation_history": [msg.to_dict() for msg in recent_history],
            "session_context": {
                "personality": session.context.personality,
                "language": session.context.language,
                "conversation_style": session.context.conversation_style
            },
            "voice_context": session.voice_context,
            "preferences": session.preferences,
            "inference_mode": kwargs.get("inference_mode", InferenceMode.SYNCHRONOUS),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512)
        }

        return context

    async def _generate_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using inference engine (mock implementation)."""
        # This would integrate with the actual inference engine
        # For now, return a mock response

        await asyncio.sleep(0.1)  # Simulate processing time

        return {
            "text": "This is a mock response from the NVIDIA Maverick Llama Voice model.",
            "confidence": 0.85,
            "processing_time": 0.1,
            "emotion": "neutral"
        }

    async def _persist_session(self, session: SessionState):
        """Persist session to storage."""
        if not self.enable_persistence:
            return

        try:
            session_file = f"{self.storage_path}/{session.context.session_id}.json"

            with open(session_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to persist session {session.context.session_id}: {str(e)}")

    async def _load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session from storage."""
        if not self.enable_persistence:
            return None

        try:
            session_file = f"{self.storage_path}/{session_id}.json"

            if not os.path.exists(session_file):
                return None

            with open(session_file, 'r') as f:
                session_data = json.load(f)

            # Reconstruct session state
            context_data = session_data["context"]
            context = SessionContext(
                session_id=context_data["session_id"],
                user_id=context_data["user_id"],
                language=context_data["language"],
                personality=context_data["personality"],
                conversation_style=context_data["conversation_style"],
                memory_enabled=context_data["memory_enabled"],
                max_history_length=context_data["max_history_length"],
                session_timeout=context_data["session_timeout"],
                created_at=datetime.fromisoformat(context_data["created_at"]),
                last_activity=datetime.fromisoformat(context_data["last_activity"])
            )

            # Reconstruct conversation history
            history = []
            for msg_data in session_data["conversation_history"]:
                msg = ConversationMessage(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    metadata=msg_data["metadata"]
                )
                history.append(msg)

            session = SessionState(
                context=context,
                conversation_history=history,
                preferences=session_data["preferences"],
                metadata=session_data["metadata"]
            )

            return session

        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {str(e)}")
            return None

    async def _session_cleanup_worker(self):
        """Background worker for cleaning up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                current_time = datetime.utcnow()
                expired_sessions = []

                async with self.session_lock:
                    for session_id, session in self.sessions.items():
                        timeout_duration = timedelta(seconds=session.context.session_timeout)
                        if current_time - session.context.last_activity > timeout_duration:
                            expired_sessions.append(session_id)

                    # Remove expired sessions
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        self.logger.info(f"Cleaned up expired session: {session_id}")

            except Exception as e:
                self.logger.error(f"Session cleanup error: {str(e)}")

    async def cleanup(self):
        """Cleanup resources."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass


# Convenience functions
async def create_voice_session(user_id: Optional[str] = None,
                             language: str = "en-US") -> str:
    """Create a new voice-enabled conversation session."""
    manager = SessionManager()
    return await manager.create_session(user_id=user_id, language=language)


async def process_voice_message(session_id: str, audio_data: bytes,
                              voice_processor) -> Dict[str, Any]:
    """Process a voice message in a session."""
    manager = SessionManager()

    # Convert speech to text
    voice_result = await voice_processor.speech_to_text(audio_data)

    if not voice_result.success:
        return {"error": "Speech recognition failed"}

    # Process message
    response = await manager.process_message(
        session_id,
        voice_result.text,
        voice_result
    )

    return response
