"""
LLM Provider abstraction layer for Template Heaven.

Supports multiple LLM providers (OpenAI, Anthropic) with unified interface
for chat completions, tool calling, and streaming responses.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, AsyncIterator, Any
from enum import Enum

from ...utils.logger import get_logger

logger = get_logger(__name__)


class LLMProviderType(Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize LLM provider.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions for function calling
            stream: Whether to stream responses
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response string or async iterator of response chunks
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    def __init__(self, **kwargs):
        """Initialize OpenAI provider."""
        api_key = kwargs.pop('api_key', None) or os.getenv('OPENAI_API_KEY')
        model = kwargs.pop('model', None) or os.getenv('OPENAI_MODEL', 'gpt-4')
        
        super().__init__(api_key=api_key, model=model, **kwargs)
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            self.client = None
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate chat completion using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")
        
        # Merge kwargs with instance settings
        params = {
            "model": kwargs.get('model', self.model),
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            **kwargs
        }
        
        # Prepare messages
        openai_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        # Add tools if provided
        if tools and self.supports_tools():
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")
        
        if stream:
            async def stream_response():
                async for chunk in await self.client.chat.completions.create(
                    messages=openai_messages,
                    stream=True,
                    **params
                ):
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            return stream_response()
        else:
            response = await self.client.chat.completions.create(
                messages=openai_messages,
                **params
            )
            
            # Handle tool calls if present
            if response.choices[0].message.tool_calls:
                return {
                    "content": response.choices[0].message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response.choices[0].message.tool_calls
                    ]
                }
            
            return response.choices[0].message.content or ""
    
    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True
    
    def supports_tools(self) -> bool:
        """OpenAI supports function calling."""
        return True


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(self, **kwargs):
        """Initialize Anthropic provider."""
        api_key = kwargs.pop('api_key', None) or os.getenv('ANTHROPIC_API_KEY')
        model = kwargs.pop('model', None) or os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
        
        super().__init__(api_key=api_key, model=model, **kwargs)
        
        if not self.api_key:
            logger.warning("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.error("anthropic package not installed. Install with: pip install anthropic")
            self.client = None
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate chat completion using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized. Provide API key.")
        
        # Merge kwargs with instance settings
        params = {
            "model": kwargs.get('model', self.model),
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            **kwargs
        }
        
        # Prepare messages (Anthropic uses different format)
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        if system_message:
            params["system"] = system_message
        
        # Add tools if provided (Anthropic uses 'tools' parameter)
        if tools and self.supports_tools():
            params["tools"] = tools
        
        if stream:
            async def stream_response():
                async with self.client.messages.stream(
                    messages=anthropic_messages,
                    **params
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
            
            return stream_response()
        else:
            response = await self.client.messages.create(
                messages=anthropic_messages,
                **params
            )
            
            # Handle tool use if present
            if response.content and isinstance(response.content, list):
                text_content = ""
                tool_uses = []
                
                for content_block in response.content:
                    if content_block.type == "text":
                        text_content += content_block.text
                    elif content_block.type == "tool_use":
                        tool_uses.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })
                
                if tool_uses:
                    return {
                        "content": text_content,
                        "tool_uses": tool_uses
                    }
                
                return text_content
            
            return ""
    
    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
        return True
    
    def supports_tools(self) -> bool:
        """Anthropic supports tool use."""
        return True


def get_llm_provider(
    provider_type: Union[str, LLMProviderType],
    config: Optional[Dict[str, Any]] = None
) -> LLMProvider:
    """
    Factory function to get LLM provider instance.
    
    Args:
        provider_type: Provider type ('openai', 'anthropic', or LLMProviderType enum)
        config: Optional configuration dictionary
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider type is not supported
    """
    if isinstance(provider_type, str):
        try:
            provider_type = LLMProviderType(provider_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    config = config or {}
    
    if provider_type == LLMProviderType.OPENAI:
        return OpenAIProvider(**config)
    elif provider_type == LLMProviderType.ANTHROPIC:
        return AnthropicProvider(**config)
    else:
        raise ValueError(f"Provider type {provider_type} not yet implemented")

