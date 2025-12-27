#!/usr/bin/env python3
"""
Template Service - Microservice for Template Management

This microservice provides REST API endpoints for managing templates in the
Template Heaven ecosystem. It handles template discovery, metadata management,
and template operations.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Template Service",
    description="Microservice for template management in Template Heaven",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Template storage (in production, this would be a database)
TEMPLATE_STORAGE = Path("/app/templates")
TEMPLATE_DATA_FILE = TEMPLATE_STORAGE / "templates.json"

# Initialize template storage
TEMPLATE_STORAGE.mkdir(exist_ok=True)
if not TEMPLATE_DATA_FILE.exists():
    with open(TEMPLATE_DATA_FILE, 'w') as f:
        json.dump([], f)


def load_templates() -> List[Dict[str, Any]]:
    """Load templates from storage."""
    try:
        with open(TEMPLATE_DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load templates: {e}")
        return []


def save_templates(templates: List[Dict[str, Any]]):
    """Save templates to storage."""
    try:
        with open(TEMPLATE_DATA_FILE, 'w') as f:
            json.dump(templates, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save templates: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "template-service"}


@app.get("/api/v1/templates")
async def list_templates(
    stack: Optional[str] = Query(None, description="Filter by stack"),
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search query")
) -> Dict[str, Any]:
    """List available templates with optional filtering."""
    templates = load_templates()

    # Apply filters
    if stack:
        templates = [t for t in templates if t.get("stack") == stack]

    if category:
        templates = [t for t in templates if category in t.get("category", [])]

    if search:
        search_lower = search.lower()
        templates = [
            t for t in templates
            if search_lower in t.get("name", "").lower() or
               search_lower in t.get("description", "").lower() or
               any(search_lower in tag.lower() for tag in t.get("tags", []))
        ]

    return {"templates": templates, "total": len(templates)}


@app.get("/api/v1/templates/{template_name}")
async def get_template(template_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific template."""
    templates = load_templates()

    for template in templates:
        if template.get("name") == template_name:
            return template

    raise HTTPException(status_code=404, detail="Template not found")


@app.post("/api/v1/templates")
async def create_template(template: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new template."""
    templates = load_templates()

    # Check if template already exists
    for existing in templates:
        if existing.get("name") == template.get("name"):
            raise HTTPException(status_code=409, detail="Template already exists")

    # Add template
    templates.append(template)
    save_templates(templates)

    logger.info(f"Created template: {template.get('name')}")
    return {"message": "Template created", "template": template}


@app.put("/api/v1/templates/{template_name}")
async def update_template(template_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing template."""
    templates = load_templates()

    for i, template in enumerate(templates):
        if template.get("name") == template_name:
            # Update template
            templates[i].update(updates)
            save_templates(templates)

            logger.info(f"Updated template: {template_name}")
            return {"message": "Template updated", "template": templates[i]}

    raise HTTPException(status_code=404, detail="Template not found")


@app.delete("/api/v1/templates/{template_name}")
async def delete_template(template_name: str) -> Dict[str, Any]:
    """Delete a template."""
    templates = load_templates()

    for i, template in enumerate(templates):
        if template.get("name") == template_name:
            # Remove template
            deleted_template = templates.pop(i)
            save_templates(templates)

            logger.info(f"Deleted template: {template_name}")
            return {"message": "Template deleted", "template": deleted_template}

    raise HTTPException(status_code=404, detail="Template not found")


@app.get("/api/v1/stacks")
async def list_stacks() -> Dict[str, Any]:
    """List available template stacks."""
    templates = load_templates()

    # Group templates by stack
    stacks = {}
    for template in templates:
        stack_name = template.get("stack", "unknown")
        if stack_name not in stacks:
            stacks[stack_name] = {
                "name": stack_name,
                "templates_count": 0,
                "categories": set(),
                "last_updated": template.get("updated_at", template.get("created_at"))
            }

        stacks[stack_name]["templates_count"] += 1
        stacks[stack_name]["categories"].update(template.get("category", []))

    # Convert to list format
    stack_list = []
    for stack_data in stacks.values():
        stack_data["categories"] = list(stack_data["categories"])
        stack_list.append(stack_data)

    return {"stacks": stack_list}


@app.get("/api/v1/tools")
async def get_tools() -> Dict[str, Any]:
    """Get available MCP tools."""
    tools = [
        {
            "name": "template-inspect",
            "description": "Inspect a template's structure and metadata",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "template_name": {
                        "type": "string",
                        "description": "Name of the template to inspect"
                    }
                },
                "required": ["template_name"]
            }
        },
        {
            "name": "template-search",
            "description": "Search for templates by keywords",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "stack": {
                        "type": "string",
                        "description": "Filter by stack"
                    }
                },
                "required": ["query"]
            }
        }
    ]

    return {"tools": tools}


@app.post("/api/v1/tools/call")
async def call_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an MCP tool."""
    tool_name = request.get("tool")
    arguments = request.get("arguments", {})

    if tool_name == "template-inspect":
        template_name = arguments.get("template_name")
        if not template_name:
            raise HTTPException(status_code=400, detail="template_name required")

        template = await get_template(template_name)
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Template: {template['name']}\nDescription: {template.get('description', 'N/A')}\nStack: {template.get('stack', 'N/A')}\nTags: {', '.join(template.get('tags', []))}"
                }
            ]
        }

    elif tool_name == "template-search":
        query = arguments.get("query")
        stack = arguments.get("stack")

        if not query:
            raise HTTPException(status_code=400, detail="query required")

        # Perform search
        templates = await list_templates(stack=stack, search=query)

        results = []
        for template in templates["templates"][:5]:  # Limit results
            results.append(f"- {template['name']}: {template.get('description', 'N/A')}")

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Search results for '{query}':\n" + "\n".join(results)
                }
            ]
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")


@app.get("/api/v1/resources")
async def get_resources() -> Dict[str, Any]:
    """Get available MCP resources."""
    resources = [
        {
            "uri": "template://catalog",
            "name": "Template Catalog",
            "description": "Complete catalog of available templates",
            "mimeType": "application/json"
        },
        {
            "uri": "template://stacks",
            "name": "Template Stacks",
            "description": "Available template stacks and categories",
            "mimeType": "application/json"
        }
    ]

    return {"resources": resources}


@app.post("/api/v1/resources/read")
async def read_resource(request: Dict[str, Any]) -> Dict[str, Any]:
    """Read an MCP resource."""
    uri = request.get("uri")

    if uri == "template://catalog":
        templates = await list_templates()
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(templates, indent=2)
                }
            ]
        }

    elif uri == "template://stacks":
        stacks = await list_stacks()
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(stacks, indent=2)
                }
            ]
        }

    else:
        raise HTTPException(status_code=404, detail="Resource not found")


@app.get("/api/v1/prompts")
async def get_prompts() -> Dict[str, Any]:
    """Get available MCP prompts."""
    prompts = [
        {
            "name": "template-recommendation",
            "description": "Get template recommendations based on project requirements",
            "arguments": [
                {
                    "name": "project_type",
                    "description": "Type of project (web, api, ml, etc.)",
                    "required": True
                },
                {
                    "name": "technologies",
                    "description": "Technologies to include",
                    "required": False
                }
            ]
        }
    ]

    return {"prompts": prompts}


@app.post("/api/v1/prompts/get")
async def get_prompt(request: Dict[str, Any]) -> Dict[str, Any]:
    """Get an MCP prompt."""
    prompt_name = request.get("name")
    arguments = request.get("arguments", {})

    if prompt_name == "template-recommendation":
        project_type = arguments.get("project_type")
        technologies = arguments.get("technologies", [])

        # Generate recommendations based on input
        recommendations = []

        if project_type == "web":
            recommendations.extend(["react-vite", "vue-template", "angular-app"])
        elif project_type == "api":
            recommendations.extend(["fastapi-postgresql", "express-mongodb"])
        elif project_type == "ml":
            recommendations.extend(["quantum-computing-starter", "ml-pipeline"])

        response_text = f"Based on your project type '{project_type}', here are recommended templates:\n"
        response_text += "\n".join(f"- {template}" for template in recommendations[:3])

        return {
            "description": f"Template recommendations for {project_type} project",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"I need to create a {project_type} project with {', '.join(technologies) if technologies else 'modern technologies'}"
                    }
                },
                {
                    "role": "assistant",
                    "content": {
                        "type": "text",
                        "text": response_text
                    }
                }
            ]
        }

    else:
        raise HTTPException(status_code=404, detail="Prompt not found")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
