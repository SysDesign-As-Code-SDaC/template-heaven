"""
MCP Protocol Implementation

Complete implementation of the Model Context Protocol (MCP) specification.
This module defines all MCP message types, data structures, and protocol handling.
"""

import json
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict


class MCPErrorCode(Enum):
    """MCP error codes as defined in the specification."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000


@dataclass
class MCPError:
    """MCP error object."""
    code: MCPErrorCode
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"code": self.code.value, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result

    def to_response(self, request_id: Optional[Union[str, int]]) -> 'MCPResponse':
        """Convert to error response."""
        return MCPResponse(
            id=request_id,
            error=self.to_dict(),
            result=None
        )


@dataclass
class MCPContent:
    """MCP content object."""
    type: str
    text: Optional[str] = None
    data: Optional[Any] = None
    mime_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"type": self.type}
        if self.text is not None:
            result["text"] = self.text
        if self.data is not None:
            result["data"] = self.data
        if self.mime_type is not None:
            result["mimeType"] = self.mime_type
        return result


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: Optional[str] = None
    inputSchema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name}
        if self.description is not None:
            result["description"] = self.description
        if self.inputSchema is not None:
            result["inputSchema"] = self.inputSchema
        return result


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"uri": self.uri, "name": self.name}
        if self.description is not None:
            result["description"] = self.description
        if self.mimeType is not None:
            result["mimeType"] = self.mimeType
        return result


@dataclass
class MCPPrompt:
    """MCP prompt definition."""
    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name}
        if self.description is not None:
            result["description"] = self.description
        if self.arguments is not None:
            result["arguments"] = self.arguments
        return result


# Base message classes
@dataclass
class MCPMessage:
    """Base MCP message class."""
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary."""
        if "method" in data:
            if "id" in data:
                return MCPRequest.from_dict(data)
            else:
                return MCPNotification.from_dict(data)
        else:
            return MCPResponse.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))


@dataclass
class MCPRequest(MCPMessage):
    """MCP request message."""
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        """Create request from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data["method"],
            params=data.get("params"),
            id=data.get("id")
        )


@dataclass
class MCPNotification(MCPMessage):
    """MCP notification message."""
    method: str
    params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPNotification':
        """Create notification from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data["method"],
            params=data.get("params")
        )


@dataclass
class MCPResponse(MCPMessage):
    """MCP response message."""
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        """Create response from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error")
        )


# Template Heaven specific data structures
@dataclass
class TemplateInfo:
    """Template information."""
    name: str
    description: str
    stack: str
    category: str
    version: str
    tags: List[str]
    features: List[str]
    dependencies: Dict[str, str]
    upstream_url: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class StackInfo:
    """Stack information."""
    name: str
    description: str
    templates_count: int
    categories: List[str]
    last_updated: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProjectOptions:
    """Project generation options."""
    name: str
    description: Optional[str] = None
    author: Optional[str] = None
    version: str = "1.0.0"
    license: str = "MIT"
    python_version: str = "3.9"
    include_tests: bool = True
    include_docs: bool = True
    include_docker: bool = True
    include_ci: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ValidationResult:
    """Template validation result."""
    valid: bool
    score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# MCP protocol utilities
def create_tool_result(content: Union[str, List[MCPContent]], is_error: bool = False) -> Dict[str, Any]:
    """Create a tool result response."""
    if isinstance(content, str):
        content = [MCPContent(type="text", text=content)]

    return {
        "content": [item.to_dict() for item in content],
        "isError": is_error
    }


def create_resource_result(contents: List[MCPContent]) -> Dict[str, Any]:
    """Create a resource read result."""
    return {
        "contents": [content.to_dict() for content in contents]
    }


def create_prompt_result(description: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a prompt get result."""
    return {
        "description": description,
        "messages": messages
    }


# Validation functions
def validate_mcp_message(data: Dict[str, Any]) -> bool:
    """Validate that data is a valid MCP message."""
    if not isinstance(data, dict):
        return False

    # Check JSON-RPC version
    if data.get("jsonrpc") != "2.0":
        return False

    # Check message type
    if "method" in data:
        # Request or notification
        if not isinstance(data["method"], str):
            return False
        # Request must have id
        if "id" in data and not isinstance(data["id"], (str, int)):
            return False
        # Params must be dict if present
        if "params" in data and not isinstance(data["params"], dict):
            return False
    else:
        # Response must have id
        if "id" not in data or not isinstance(data["id"], (str, int, type(None))):
            return False
        # Must have either result or error
        has_result = "result" in data
        has_error = "error" in data
        if not (has_result or has_error):
            return False
        if has_result and has_error:
            return False

    return True


def create_error_response(error_code: MCPErrorCode, message: str, request_id: Optional[Union[str, int]] = None) -> MCPResponse:
    """Create an error response."""
    error = MCPError(code=error_code, message=message)
    return error.to_response(request_id)
