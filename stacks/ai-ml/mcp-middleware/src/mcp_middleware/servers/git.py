"""
Git MCP Server.

This server provides Git repository operations through the MCP protocol,
allowing AI assistants to interact with Git repositories.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class GitServer(BaseMCPServer):
    """MCP server for Git operations."""

    def __init__(self, name: str, config: Dict[str, Any], auth: Dict[str, Any]):
        super().__init__(name, config, auth)
        self.allowed_paths = config.get("allowed_paths", [])
        self.max_commits = config.get("max_commits", 100)

    @classmethod
    def validate_config(cls, config: Dict[str, Any]):
        """Validate Git server configuration."""
        allowed_paths = config.get("allowed_paths", [])
        for path in allowed_paths:
            if not Path(path).is_absolute():
                raise ValueError(f"Git path must be absolute: {path}")

    async def initialize(self):
        """Initialize the Git server."""
        self.initialized = True
        logger.info(f"Git server {self.name} initialized")

    async def shutdown(self):
        """Shutdown the Git server."""
        self.initialized = False
        logger.info(f"Git server {self.name} shutdown")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": True,
            "timestamp": datetime.utcnow().isoformat(),
            "server": self.name
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available Git tools."""
        return [
            {
                "name": "get_commit_history",
                "description": "Get commit history for a repository",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to Git repository"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of commits",
                            "default": 10
                        }
                    },
                    "required": ["repo_path"]
                }
            },
            {
                "name": "get_file_diff",
                "description": "Get diff for a specific file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Path to Git repository"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to file within repository"
                        },
                        "commit_sha": {
                            "type": "string",
                            "description": "Specific commit SHA"
                        }
                    },
                    "required": ["repo_path", "file_path"]
                }
            }
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a Git tool."""
        try:
            if tool_name == "get_commit_history":
                return await self._get_commit_history(arguments)
            elif tool_name == "get_file_diff":
                return await self._get_file_diff(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            return self._handle_error(f"call_tool_{tool_name}", e)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List Git resources."""
        return []

    async def read_resource(self, resource_uri: str) -> Dict[str, Any]:
        """Read a Git resource."""
        raise NotImplementedError("Git resources not implemented")

    def _validate_repo_path(self, repo_path: str) -> bool:
        """Validate repository path."""
        if not self.allowed_paths:
            return True

        repo_path = Path(repo_path).resolve()
        for allowed_path in self.allowed_paths:
            if str(repo_path).startswith(allowed_path):
                return True
        return False

    async def _get_commit_history(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get commit history."""
        repo_path = arguments["repo_path"]
        limit = min(arguments.get("limit", 10), self.max_commits)

        if not self._validate_repo_path(repo_path):
            raise ValueError(f"Repository path not allowed: {repo_path}")

        try:
            import git

            repo = git.Repo(repo_path)
            commits = []

            for commit in repo.iter_commits(max_count=limit):
                commits.append({
                    "sha": commit.hexsha,
                    "author": commit.author.name,
                    "email": commit.author.email,
                    "date": commit.authored_datetime.isoformat(),
                    "message": commit.message.strip(),
                    "parents": [p.hexsha for p in commit.parents]
                })

            return {
                "repository": repo_path,
                "commits": commits,
                "count": len(commits)
            }

        except ImportError:
            raise ValueError("GitPython not available")
        except Exception as e:
            raise ValueError(f"Failed to get commit history: {str(e)}")

    async def _get_file_diff(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get file diff."""
        repo_path = arguments["repo_path"]
        file_path = arguments["file_path"]
        commit_sha = arguments.get("commit_sha")

        if not self._validate_repo_path(repo_path):
            raise ValueError(f"Repository path not allowed: {repo_path}")

        try:
            import git

            repo = git.Repo(repo_path)

            if commit_sha:
                commit = repo.commit(commit_sha)
                # Get diff for specific commit
                if len(commit.parents) > 0:
                    diff = commit.parents[0].diff(commit, paths=[file_path])
                else:
                    # Initial commit
                    diff = commit.diff(git.NULL_TREE, paths=[file_path])
            else:
                # Get unstaged changes
                diff = repo.index.diff(None, paths=[file_path])

            diff_text = ""
            for d in diff:
                diff_text += str(d) + "\n"

            return {
                "repository": repo_path,
                "file_path": file_path,
                "commit_sha": commit_sha,
                "diff": diff_text
            }

        except ImportError:
            raise ValueError("GitPython not available")
        except Exception as e:
            raise ValueError(f"Failed to get file diff: {str(e)}")
