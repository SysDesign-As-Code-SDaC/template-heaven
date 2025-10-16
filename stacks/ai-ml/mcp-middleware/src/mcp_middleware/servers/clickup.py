"""
ClickUp MCP Server.

This server provides AI assistants with the ability to interact with ClickUp workspaces,
including managing tasks, lists, spaces, and team collaboration features.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
import aiohttp
from datetime import datetime, timedelta
import re

from .base import BaseMCPServer

logger = logging.getLogger(__name__)

class ClickUpServer(BaseMCPServer):
    """
    MCP server for ClickUp integration.

    Provides tools for AI assistants to interact with ClickUp workspaces,
    manage tasks, lists, and collaborate with team members.
    """

    BASE_URL = "https://api.clickup.com/api/v2"

    def __init__(self, name: str, config: Dict[str, Any], auth: Optional[Dict[str, Any]] = None):
        super().__init__(name, config, auth)
        self.session: Optional[aiohttp.ClientSession] = None
        self.team_id = None
        self.workspaces: Dict[str, Any] = {}
        self.lists: Dict[str, Any] = {}
        self.tasks_cache: Dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize the ClickUp server."""
        try:
            # Validate configuration
            self._validate_config()

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Validate authentication and get team info
            await self._validate_authentication()

            # Cache workspace and list information
            await self._load_workspaces_and_lists()

            logger.info(f"ClickUp server '{self.name}' initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ClickUp server '{self.name}': {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the ClickUp server."""
        try:
            if self.session:
                await self.session.close()
                self.session = None

            self.workspaces.clear()
            self.lists.clear()
            self.tasks_cache.clear()

            logger.info(f"ClickUp server '{self.name}' shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error shutting down ClickUp server '{self.name}': {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            # Test API connectivity
            async with self.session.get(f"{self.BASE_URL}/user") as response:
                if response.status == 200:
                    user_data = await response.json()

                    health_status = {
                        "status": "healthy",
                        "user_id": user_data.get("id"),
                        "username": user_data.get("username"),
                        "team_id": self.team_id,
                        "workspaces_count": len(self.workspaces),
                        "lists_count": len(self.lists),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    health_status = {
                        "status": "unhealthy",
                        "error": f"API returned status {response.status}",
                        "timestamp": datetime.utcnow().isoformat()
                    }

            return health_status

        except Exception as e:
            logger.error(f"Health check failed for ClickUp server '{self.name}': {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available ClickUp tools."""
        return [
            {
                "name": "get_workspaces",
                "description": "Get all accessible ClickUp workspaces (teams)",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_workspace_details",
                "description": "Get detailed information about a specific workspace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the workspace to retrieve"
                        }
                    },
                    "required": ["workspace_id"]
                }
            },
            {
                "name": "get_spaces",
                "description": "Get all spaces in a workspace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the workspace"
                        }
                    },
                    "required": ["workspace_id"]
                }
            },
            {
                "name": "get_lists",
                "description": "Get all lists in a space or workspace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "space_id": {
                            "type": "string",
                            "description": "ID of the space (optional, gets all lists if not provided)"
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the workspace (required if space_id not provided)"
                        },
                        "archived": {
                            "type": "boolean",
                            "description": "Include archived lists",
                            "default": false
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "create_task",
                "description": "Create a new task in a list",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "list_id": {
                            "type": "string",
                            "description": "ID of the list to create task in"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name/title of the task"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the task"
                        },
                        "assignees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of user IDs to assign the task to"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tags to add to the task"
                        },
                        "priority": {
                            "type": "integer",
                            "enum": [1, 2, 3, 4],
                            "description": "Priority level (1=Urgent, 2=High, 3=Normal, 4=Low)"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date in ISO format (YYYY-MM-DDTHH:MM:SSZ)"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in ISO format (YYYY-MM-DDTHH:MM:SSZ)"
                        }
                    },
                    "required": ["list_id", "name"]
                }
            },
            {
                "name": "get_tasks",
                "description": "Get tasks from a list or search for tasks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "list_id": {
                            "type": "string",
                            "description": "ID of the list to get tasks from"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query to find tasks"
                        },
                        "assignees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by assignee user IDs"
                        },
                        "statuses": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by task statuses"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags"
                        },
                        "due_date_from": {
                            "type": "string",
                            "description": "Filter tasks due after this date (YYYY-MM-DD)"
                        },
                        "due_date_to": {
                            "type": "string",
                            "description": "Filter tasks due before this date (YYYY-MM-DD)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tasks to return",
                            "default": 50,
                            "maximum": 100
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "update_task",
                "description": "Update an existing task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to update"
                        },
                        "name": {
                            "type": "string",
                            "description": "New name/title for the task"
                        },
                        "description": {
                            "type": "string",
                            "description": "New description for the task"
                        },
                        "status": {
                            "type": "string",
                            "description": "New status for the task"
                        },
                        "priority": {
                            "type": "integer",
                            "enum": [1, 2, 3, 4],
                            "description": "New priority level (1=Urgent, 2=High, 3=Normal, 4=Low)"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "New due date in ISO format (YYYY-MM-DDTHH:MM:SSZ)"
                        },
                        "assignees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New list of assignee user IDs"
                        },
                        "add_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to add to the task"
                        },
                        "remove_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to remove from the task"
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "delete_task",
                "description": "Delete a task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to delete"
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "add_task_comment",
                "description": "Add a comment to a task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to comment on"
                        },
                        "comment_text": {
                            "type": "string",
                            "description": "Comment text to add"
                        },
                        "assignee": {
                            "type": "string",
                            "description": "User ID to assign the comment to (optional)"
                        },
                        "notify_all": {
                            "type": "boolean",
                            "description": "Whether to notify all task members",
                            "default": false
                        }
                    },
                    "required": ["task_id", "comment_text"]
                }
            },
            {
                "name": "get_task_comments",
                "description": "Get comments for a task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to get comments for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of comments to return",
                            "default": 50
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "create_list",
                "description": "Create a new list in a space",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "space_id": {
                            "type": "string",
                            "description": "ID of the space to create list in"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name of the new list"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description for the list"
                        },
                        "assignee": {
                            "type": "string",
                            "description": "Default assignee for tasks in this list"
                        }
                    },
                    "required": ["space_id", "name"]
                }
            },
            {
                "name": "get_team_members",
                "description": "Get all members of a team/workspace",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "ID of the workspace/team"
                        }
                    },
                    "required": ["workspace_id"]
                }
            },
            {
                "name": "search_tasks",
                "description": "Advanced task search across workspaces",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace to search in"
                        },
                        "assignee": {
                            "type": "string",
                            "description": "Filter by assignee user ID"
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by task status"
                        },
                        "date_created_gt": {
                            "type": "string",
                            "description": "Filter tasks created after this date (YYYY-MM-DD)"
                        },
                        "date_updated_gt": {
                            "type": "string",
                            "description": "Filter tasks updated after this date (YYYY-MM-DD)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 25,
                            "maximum": 100
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a ClickUp tool."""
        try:
            if tool_name == "get_workspaces":
                return await self._get_workspaces()
            elif tool_name == "get_workspace_details":
                return await self._get_workspace_details(arguments)
            elif tool_name == "get_spaces":
                return await self._get_spaces(arguments)
            elif tool_name == "get_lists":
                return await self._get_lists(arguments)
            elif tool_name == "create_task":
                return await self._create_task(arguments)
            elif tool_name == "get_tasks":
                return await self._get_tasks(arguments)
            elif tool_name == "update_task":
                return await self._update_task(arguments)
            elif tool_name == "delete_task":
                return await self._delete_task(arguments)
            elif tool_name == "add_task_comment":
                return await self._add_task_comment(arguments)
            elif tool_name == "get_task_comments":
                return await self._get_task_comments(arguments)
            elif tool_name == "create_list":
                return await self._create_list(arguments)
            elif tool_name == "get_team_members":
                return await self._get_team_members(arguments)
            elif tool_name == "search_tasks":
                return await self._search_tasks(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing ClickUp tool '{tool_name}': {e}")
            return [{
                "type": "text",
                "text": f"Error executing {tool_name}: {str(e)}"
            }]

    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available ClickUp resources."""
        resources = []

        # Add workspaces as resources
        for workspace_id, workspace in self.workspaces.items():
            resources.append({
                "uri": f"clickup://{self.name}/workspace/{workspace_id}",
                "mimeType": "application/json",
                "description": f"Workspace: {workspace.get('name', workspace_id)}"
            })

        # Add lists as resources
        for list_id, list_info in self.lists.items():
            resources.append({
                "uri": f"clickup://{self.name}/list/{list_id}",
                "mimeType": "application/json",
                "description": f"List: {list_info.get('name', list_id)}"
            })

        # Add team information
        if self.team_id:
            resources.append({
                "uri": f"clickup://{self.name}/team/{self.team_id}",
                "mimeType": "application/json",
                "description": "Team information and members"
            })

        return resources

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a ClickUp resource."""
        try:
            if not uri.startswith("clickup://"):
                raise ValueError(f"Invalid ClickUp resource URI: {uri}")

            # Parse URI: clickup://server_name/resource_type/resource_id
            parts = uri[10:].split("/", 2)  # Remove "clickup://" prefix
            if len(parts) < 2:
                raise ValueError(f"Invalid ClickUp resource URI format: {uri}")

            server_name = parts[0]
            resource_type = parts[1]

            if server_name != self.name:
                raise ValueError(f"Resource belongs to different server: {server_name}")

            if resource_type == "workspace" and len(parts) > 2:
                workspace_id = parts[2]
                details = await self._get_workspace_details({"workspace_id": workspace_id})
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(details[0]["text"] if details else {"error": "Workspace not found"}, indent=2)
                    }]
                }

            elif resource_type == "list" and len(parts) > 2:
                list_id = parts[2]
                # Get list details
                list_details = self.lists.get(list_id, {"error": "List not found"})
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(list_details, indent=2)
                    }]
                }

            elif resource_type == "team" and len(parts) > 2:
                team_id = parts[2]
                members = await self._get_team_members({"workspace_id": team_id})
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(members[0]["text"] if members else {"error": "Team not found"}, indent=2)
                    }]
                }

            else:
                raise ValueError(f"Unknown resource type: {resource_type}")

        except Exception as e:
            logger.error(f"Error reading ClickUp resource '{uri}': {e}")
            raise

    def _validate_config(self):
        """Validate ClickUp server configuration."""
        required_keys = []
        if not all(key in self.config for key in required_keys):
            missing = [key for key in required_keys if key not in self.config]
            raise ValueError(f"Missing required configuration keys: {missing}")

        # Validate timeout
        if "timeout" in self.config:
            if not isinstance(self.config["timeout"], (int, float)) or self.config["timeout"] <= 0:
                raise ValueError("Timeout must be a positive number")

    async def _validate_authentication(self):
        """Validate ClickUp API authentication."""
        if not self.auth or "api_token" not in self.auth:
            raise ValueError("ClickUp API token is required in auth configuration")

        # Test authentication by getting user info
        async with self.session.get(f"{self.BASE_URL}/user") as response:
            if response.status == 200:
                user_data = await response.json()
                self.team_id = user_data.get("default_team")
                if not self.team_id:
                    # Get first available team
                    teams_response = await self.session.get(f"{self.BASE_URL}/team")
                    if teams_response.status == 200:
                        teams_data = await teams_response.json()
                        if teams_data.get("teams"):
                            self.team_id = teams_data["teams"][0]["id"]
            else:
                raise ValueError(f"ClickUp API authentication failed: {response.status}")

    async def _load_workspaces_and_lists(self):
        """Load and cache workspace and list information."""
        try:
            # Get workspaces (teams)
            async with self.session.get(f"{self.BASE_URL}/team") as response:
                if response.status == 200:
                    teams_data = await response.json()
                    for team in teams_data.get("teams", []):
                        self.workspaces[team["id"]] = team

            # Get spaces and lists for each workspace
            for workspace_id in self.workspaces.keys():
                # Get spaces
                async with self.session.get(f"{self.BASE_URL}/team/{workspace_id}/space") as response:
                    if response.status == 200:
                        spaces_data = await response.json()
                        for space in spaces_data.get("spaces", []):
                            # Get lists for each space
                            async with self.session.get(f"{self.BASE_URL}/space/{space['id']}/list") as list_response:
                                if list_response.status == 200:
                                    lists_data = await list_response.json()
                                    for list_item in lists_data.get("lists", []):
                                        self.lists[list_item["id"]] = {
                                            **list_item,
                                            "space_id": space["id"],
                                            "workspace_id": workspace_id
                                        }

        except Exception as e:
            logger.warning(f"Failed to load workspaces and lists: {e}")

    async def _get_workspaces(self) -> List[Dict[str, Any]]:
        """Get all accessible workspaces."""
        workspaces_info = []
        for workspace_id, workspace in self.workspaces.items():
            workspaces_info.append({
                "id": workspace_id,
                "name": workspace.get("name", "Unnamed Workspace"),
                "color": workspace.get("color"),
                "members_count": len(workspace.get("members", []))
            })

        return [{
            "type": "text",
            "text": f"Found {len(workspaces_info)} workspaces:\n\n" +
                   "\n".join([f"• **{w['name']}** (ID: {w['id']}) - {w['members_count']} members"
                             for w in workspaces_info])
        }]

    async def _get_workspace_details(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get detailed workspace information."""
        workspace_id = args["workspace_id"]

        if workspace_id not in self.workspaces:
            return [{
                "type": "text",
                "text": f"Workspace {workspace_id} not found or not accessible."
            }]

        workspace = self.workspaces[workspace_id]

        # Get additional details
        async with self.session.get(f"{self.BASE_URL}/team/{workspace_id}") as response:
            if response.status == 200:
                details = await response.json()
                workspace_details = {
                    "id": workspace_id,
                    "name": details.get("name", workspace.get("name")),
                    "color": details.get("color"),
                    "avatar": details.get("avatar"),
                    "members": details.get("members", []),
                    "spaces_count": len([l for l in self.lists.values() if l.get("workspace_id") == workspace_id])
                }

                return [{
                    "type": "text",
                    "text": f"Workspace Details:\n\n**Name:** {workspace_details['name']}\n**ID:** {workspace_details['id']}\n**Color:** {workspace_details['color']}\n**Members:** {len(workspace_details['members'])}\n**Spaces:** {workspace_details['spaces_count']}"
                }]

        return [{
            "type": "text",
            "text": f"Could not retrieve details for workspace {workspace_id}."
        }]

    async def _get_spaces(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get spaces in a workspace."""
        workspace_id = args["workspace_id"]

        async with self.session.get(f"{self.BASE_URL}/team/{workspace_id}/space") as response:
            if response.status == 200:
                spaces_data = await response.json()
                spaces = spaces_data.get("spaces", [])

                spaces_info = []
                for space in spaces:
                    space_info = {
                        "id": space["id"],
                        "name": space["name"],
                        "color": space.get("color"),
                        "private": space.get("private", False),
                        "statuses": space.get("statuses", []),
                        "lists_count": len([l for l in self.lists.values() if l.get("space_id") == space["id"]])
                    }
                    spaces_info.append(space_info)

                return [{
                    "type": "text",
                    "text": f"Spaces in workspace {workspace_id}:\n\n" +
                           "\n".join([f"• **{s['name']}** (ID: {s['id']}) - {s['lists_count']} lists"
                                     for s in spaces_info])
                }]

        return [{
            "type": "text",
            "text": f"Could not retrieve spaces for workspace {workspace_id}."
        }]

    async def _get_lists(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get lists in a space or workspace."""
        space_id = args.get("space_id")
        workspace_id = args.get("workspace_id")
        archived = args.get("archived", False)

        if space_id:
            # Get lists for specific space
            async with self.session.get(f"{self.BASE_URL}/space/{space_id}/list?archived={str(archived).lower()}") as response:
                if response.status == 200:
                    lists_data = await response.json()
                    lists = lists_data.get("lists", [])
                else:
                    lists = []
        else:
            # Get all lists for workspace (from cache)
            lists = [l for l in self.lists.values() if l.get("workspace_id") == workspace_id]

        lists_info = []
        for list_item in lists:
            list_info = {
                "id": list_item["id"],
                "name": list_item["name"],
                "content": list_item.get("content", ""),
                "status": list_item.get("status"),
                "priority": list_item.get("priority"),
                "assignee": list_item.get("assignee"),
                "space_id": list_item.get("space_id"),
                "workspace_id": list_item.get("workspace_id")
            }
            lists_info.append(list_info)

        return [{
            "type": "text",
            "text": f"Found {len(lists_info)} lists:\n\n" +
                   "\n".join([f"• **{l['name']}** (ID: {l['id']}) - Status: {l.get('status', 'No status')}"
                             for l in lists_info])
        }]

    async def _create_task(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a new task."""
        list_id = args["list_id"]
        name = args["name"]
        description = args.get("description", "")
        assignees = args.get("assignees", [])
        tags = args.get("tags", [])
        priority = args.get("priority")
        due_date = args.get("due_date")
        start_date = args.get("start_date")

        # Prepare task data
        task_data = {
            "name": name,
            "description": description,
            "assignees": assignees,
            "tags": tags
        }

        if priority:
            task_data["priority"] = priority

        if due_date:
            # Convert to Unix timestamp
            due_datetime = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            task_data["due_date"] = int(due_datetime.timestamp() * 1000)

        if start_date:
            start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            task_data["start_date"] = int(start_datetime.timestamp() * 1000)

        async with self.session.post(f"{self.BASE_URL}/list/{list_id}/task", json=task_data) as response:
            if response.status == 200:
                task_result = await response.json()
                return [{
                    "type": "text",
                    "text": f"Task created successfully!\n\n**Name:** {task_result['name']}\n**ID:** {task_result['id']}\n**URL:** {task_result['url']}\n**Status:** {task_result.get('status', {}).get('status', 'to do')}"
                }]
            else:
                error_data = await response.text()
                return [{
                    "type": "text",
                    "text": f"Failed to create task. Status: {response.status}\nError: {error_data}"
                }]

    async def _get_tasks(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get tasks from a list or search."""
        list_id = args.get("list_id")
        query = args.get("query")
        assignees = args.get("assignees", [])
        statuses = args.get("statuses", [])
        tags = args.get("tags", [])
        due_date_from = args.get("due_date_from")
        due_date_to = args.get("due_date_to")
        limit = min(args.get("limit", 50), 100)

        if list_id:
            # Get tasks from specific list
            url = f"{self.BASE_URL}/list/{list_id}/task"
            params = {"limit": limit}
        else:
            # Search across workspace
            url = f"{self.BASE_URL}/team/{self.team_id}/task"
            params = {"limit": limit}

            if query:
                params["search"] = query
            if assignees:
                params["assignees"] = ",".join(assignees)
            if statuses:
                params["statuses"] = ",".join(statuses)
            if tags:
                params["tags"] = ",".join(tags)
            if due_date_from:
                params["due_date_gt"] = int(datetime.fromisoformat(due_date_from).timestamp() * 1000)
            if due_date_to:
                params["due_date_lt"] = int(datetime.fromisoformat(due_date_to).timestamp() * 1000)

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                tasks_data = await response.json()
                tasks = tasks_data.get("tasks", [])

                tasks_info = []
                for task in tasks[:limit]:
                    task_info = {
                        "id": task["id"],
                        "name": task["name"],
                        "status": task.get("status", {}).get("status", "to do"),
                        "priority": task.get("priority"),
                        "due_date": task.get("due_date"),
                        "assignees": [a["username"] for a in task.get("assignees", [])],
                        "tags": [t["name"] for t in task.get("tags", [])],
                        "url": task.get("url")
                    }
                    tasks_info.append(task_info)

                return [{
                    "type": "text",
                    "text": f"Found {len(tasks_info)} tasks:\n\n" +
                           "\n".join([f"• **{t['name']}** (ID: {t['id']})\n  Status: {t['status']}, Priority: {t.get('priority', 'None')}, Assignees: {', '.join(t['assignees']) or 'None'}"
                                     for t in tasks_info])
                }]

        return [{
            "type": "text",
            "text": f"Could not retrieve tasks. Status: {response.status}"
        }]

    async def _update_task(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update an existing task."""
        task_id = args["task_id"]
        update_data = {}

        # Map arguments to ClickUp API fields
        if "name" in args:
            update_data["name"] = args["name"]
        if "description" in args:
            update_data["description"] = args["description"]
        if "status" in args:
            update_data["status"] = args["status"]
        if "priority" in args:
            update_data["priority"] = args["priority"]
        if "due_date" in args:
            due_datetime = datetime.fromisoformat(args["due_date"].replace('Z', '+00:00'))
            update_data["due_date"] = int(due_datetime.timestamp() * 1000)
        if "assignees" in args:
            update_data["assignees"] = args["assignees"]

        # Handle tags separately
        if "add_tags" in args or "remove_tags" in args:
            await self._update_task_tags(task_id, args.get("add_tags", []), args.get("remove_tags", []))

        if update_data:
            async with self.session.put(f"{self.BASE_URL}/task/{task_id}", json=update_data) as response:
                if response.status == 200:
                    task_result = await response.json()
                    return [{
                        "type": "text",
                        "text": f"Task updated successfully!\n\n**Name:** {task_result['name']}\n**ID:** {task_result['id']}\n**Status:** {task_result.get('status', {}).get('status', 'to do')}"
                    }]
                else:
                    error_data = await response.text()
                    return [{
                        "type": "text",
                        "text": f"Failed to update task. Status: {response.status}\nError: {error_data}"
                    }]

        return [{
            "type": "text",
            "text": "No updates specified for the task."
        }]

    async def _delete_task(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Delete a task."""
        task_id = args["task_id"]

        async with self.session.delete(f"{self.BASE_URL}/task/{task_id}") as response:
            if response.status == 204:
                return [{
                    "type": "text",
                    "text": f"Task {task_id} deleted successfully."
                }]
            else:
                error_data = await response.text()
                return [{
                    "type": "text",
                    "text": f"Failed to delete task. Status: {response.status}\nError: {error_data}"
                }]

    async def _add_task_comment(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add a comment to a task."""
        task_id = args["task_id"]
        comment_text = args["comment_text"]
        assignee = args.get("assignee")
        notify_all = args.get("notify_all", False)

        comment_data = {
            "comment_text": comment_text,
            "notify_all": notify_all
        }

        if assignee:
            comment_data["assignee"] = assignee

        async with self.session.post(f"{self.BASE_URL}/task/{task_id}/comment", json=comment_data) as response:
            if response.status == 200:
                comment_result = await response.json()
                return [{
                    "type": "text",
                    "text": f"Comment added successfully to task {task_id}.\n\n**Comment ID:** {comment_result['id']}\n**Author:** {comment_result.get('user', {}).get('username', 'Unknown')}"
                }]
            else:
                error_data = await response.text()
                return [{
                    "type": "text",
                    "text": f"Failed to add comment. Status: {response.status}\nError: {error_data}"
                }]

    async def _get_task_comments(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get comments for a task."""
        task_id = args["task_id"]
        limit = min(args.get("limit", 50), 100)

        async with self.session.get(f"{self.BASE_URL}/task/{task_id}/comment") as response:
            if response.status == 200:
                comments_data = await response.json()
                comments = comments_data.get("comments", [])[:limit]

                comments_info = []
                for comment in comments:
                    comment_info = {
                        "id": comment["id"],
                        "comment_text": comment["comment_text"],
                        "author": comment.get("user", {}).get("username", "Unknown"),
                        "date": comment.get("date"),
                        "resolved": comment.get("resolved", False)
                    }
                    comments_info.append(comment_info)

                return [{
                    "type": "text",
                    "text": f"Found {len(comments_info)} comments for task {task_id}:\n\n" +
                           "\n".join([f"• **{c['author']}** ({c['date'][:19] if c['date'] else 'Unknown'}): {c['comment_text'][:100]}{'...' if len(c['comment_text']) > 100 else ''}"
                                     for c in comments_info])
                }]

        return [{
            "type": "text",
            "text": f"Could not retrieve comments for task {task_id}."
        }]

    async def _create_list(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a new list in a space."""
        space_id = args["space_id"]
        name = args["name"]
        description = args.get("description", "")
        assignee = args.get("assignee")

        list_data = {
            "name": name,
            "content": description
        }

        if assignee:
            list_data["assignee"] = assignee

        async with self.session.post(f"{self.BASE_URL}/space/{space_id}/list", json=list_data) as response:
            if response.status == 200:
                list_result = await response.json()
                # Update local cache
                self.lists[list_result["id"]] = {
                    **list_result,
                    "space_id": space_id,
                    "workspace_id": list_result.get("team_id")
                }

                return [{
                    "type": "text",
                    "text": f"List created successfully!\n\n**Name:** {list_result['name']}\n**ID:** {list_result['id']}\n**Space ID:** {space_id}"
                }]
            else:
                error_data = await response.text()
                return [{
                    "type": "text",
                    "text": f"Failed to create list. Status: {response.status}\nError: {error_data}"
                }]

    async def _get_team_members(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get team members."""
        workspace_id = args["workspace_id"]

        async with self.session.get(f"{self.BASE_URL}/team/{workspace_id}") as response:
            if response.status == 200:
                team_data = await response.json()
                members = team_data.get("members", [])

                members_info = []
                for member in members:
                    member_info = {
                        "id": member["id"],
                        "username": member["username"],
                        "email": member["email"],
                        "role": member.get("role"),
                        "timezone": member.get("timezone")
                    }
                    members_info.append(member_info)

                return [{
                    "type": "text",
                    "text": f"Team {workspace_id} has {len(members_info)} members:\n\n" +
                           "\n".join([f"• **{m['username']}** ({m['email']}) - Role: {m.get('role', 'Member')}"
                                     for m in members_info])
                }]

        return [{
            "type": "text",
            "text": f"Could not retrieve team members for workspace {workspace_id}."
        }]

    async def _search_tasks(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced task search."""
        query = args["query"]
        workspace_id = args.get("workspace_id", self.team_id)
        assignee = args.get("assignee")
        status = args.get("status")
        date_created_gt = args.get("date_created_gt")
        date_updated_gt = args.get("date_updated_gt")
        limit = min(args.get("limit", 25), 100)

        params = {
            "search": query,
            "limit": limit
        }

        if assignee:
            params["assignees"] = assignee
        if status:
            params["statuses"] = status
        if date_created_gt:
            params["date_created_gt"] = int(datetime.fromisoformat(date_created_gt).timestamp() * 1000)
        if date_updated_gt:
            params["date_updated_gt"] = int(datetime.fromisoformat(date_updated_gt).timestamp() * 1000)

        async with self.session.get(f"{self.BASE_URL}/team/{workspace_id}/task", params=params) as response:
            if response.status == 200:
                search_data = await response.json()
                tasks = search_data.get("tasks", [])

                search_results = []
                for task in tasks:
                    result = {
                        "id": task["id"],
                        "name": task["name"],
                        "status": task.get("status", {}).get("status", "to do"),
                        "url": task.get("url"),
                        "list": task.get("list", {}).get("name", "Unknown"),
                        "assignees": [a["username"] for a in task.get("assignees", [])]
                    }
                    search_results.append(result)

                return [{
                    "type": "text",
                    "text": f"Search results for '{query}':\n\n" +
                           "\n".join([f"• **{r['name']}** (ID: {r['id']})\n  Status: {r['status']}, List: {r['list']}, Assignees: {', '.join(r['assignees']) or 'None'}"
                                     for r in search_results])
                }]

        return [{
            "type": "text",
            "text": f"Search failed. Status: {response.status}"
        }]

    async def _update_task_tags(self, task_id: str, add_tags: List[str], remove_tags: List[str]):
        """Update task tags."""
        # Add tags
        for tag in add_tags:
            tag_data = {"name": tag}
            async with self.session.post(f"{self.BASE_URL}/task/{task_id}/tag", json=tag_data) as response:
                if response.status not in [200, 201]:
                    logger.warning(f"Failed to add tag '{tag}' to task {task_id}")

        # Remove tags (would need tag IDs, simplified implementation)
        # This would require getting current tags first
        pass
