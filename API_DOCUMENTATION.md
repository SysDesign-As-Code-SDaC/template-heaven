# Template Heaven API Documentation

## Overview

Template Heaven provides a RESTful API for managing and discovering project templates across various technology stacks. The API is built with FastAPI and provides comprehensive functionality for template management.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for basic operations. Authentication will be implemented in future versions.

## Endpoints

### Health Check

#### GET /api/v1/health

Returns the current health status of the API and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 123.45,
  "timestamp": "2025-10-14T14:32:59.838181",
  "dependencies": {
    "database": "healthy",
    "cache": "healthy",
    "github_api": "degraded",
    "filesystem": "healthy"
  },
  "metrics": {
    "cpu_percent": 20.0,
    "memory_percent": 89.3,
    "memory_available_gb": 0.625,
    "disk_percent": 53.7,
    "disk_free_gb": 95.4,
    "process_count": 289
  }
}
```

### Templates

#### GET /api/v1/templates

List all available templates with optional filtering and pagination.

**Query Parameters:**
- `stack` (optional): Filter by technology stack
- `tags` (optional): Filter by tags (comma-separated)
- `limit` (optional): Maximum number of results (default: 100, max: 1000)
- `offset` (optional): Number of results to skip (default: 0)
- `sort_by` (optional): Field to sort by (default: "quality_score")
- `sort_order` (optional): Sort order - "asc" or "desc" (default: "desc")

**Example:**
```bash
curl "http://localhost:8000/api/v1/templates?stack=frontend&limit=10"
```

**Response:**
```json
{
  "success": true,
  "message": "Found 1 templates",
  "data": {
    "templates": [
      {
        "name": "test-template",
        "description": "A test template",
        "stack": "frontend",
        "path": "/templates/default",
        "upstream_url": null,
        "version": "0.1.0",
        "author": "Unknown",
        "license": "MIT",
        "tags": ["test"],
        "technologies": ["react", "typescript"],
        "features": [],
        "dependencies": {},
        "min_python_version": null,
        "min_node_version": null,
        "stars": 50,
        "forks": 10,
        "growth_rate": 0.0,
        "quality_score": 0.8,
        "created_at": "2025-10-14T14:32:59.838181",
        "updated_at": "2025-10-14T14:32:59.838181"
      }
    ],
    "total_count": 1,
    "limit": 100,
    "offset": 0
  }
}
```

#### GET /api/v1/templates/{template_id}

Get details of a specific template by ID or name.

**Path Parameters:**
- `template_id`: Template ID or name

**Query Parameters:**
- `stack` (optional): Stack name (if template_id is name)

**Example:**
```bash
curl "http://localhost:8000/api/v1/templates/test-template?stack=frontend"
```

**Response:**
```json
{
  "success": true,
  "message": "Template retrieved successfully",
  "data": {
    "template": {
      "name": "test-template",
      "description": "A test template",
      "stack": "frontend",
      "path": "/templates/default",
      "upstream_url": null,
      "version": "0.1.0",
      "author": "Unknown",
      "license": "MIT",
      "tags": ["test"],
      "technologies": ["react", "typescript"],
      "features": [],
      "dependencies": {},
      "min_python_version": null,
      "min_node_version": null,
      "stars": 50,
      "forks": 10,
      "growth_rate": 0.0,
      "quality_score": 0.8,
      "created_at": "2025-10-14T14:32:59.838181",
      "updated_at": "2025-10-14T14:32:59.838181"
    }
  }
}
```

#### POST /api/v1/templates

Create a new template.

**Request Body:**
```json
{
  "name": "my-new-template",
  "stack": "frontend",
  "description": "A new template",
  "technologies": ["react", "typescript"],
  "tags": ["frontend", "react"],
  "stars": 100,
  "forks": 20,
  "quality_score": 0.9
}
```

**Response:**
```json
{
  "success": true,
  "message": "Template created successfully",
  "data": {
    "template": {
      "name": "my-new-template",
      "description": "A new template",
      "stack": "frontend",
      "path": "/templates/default",
      "upstream_url": null,
      "version": "0.1.0",
      "author": "Unknown",
      "license": "MIT",
      "tags": ["frontend", "react"],
      "technologies": ["react", "typescript"],
      "features": [],
      "dependencies": {},
      "min_python_version": null,
      "min_node_version": null,
      "stars": 100,
      "forks": 20,
      "growth_rate": 0.0,
      "quality_score": 0.9,
      "created_at": "2025-10-14T14:32:59.838181",
      "updated_at": "2025-10-14T14:32:59.838181"
    }
  }
}
```

#### PUT /api/v1/templates/{template_id}

Update an existing template.

**Path Parameters:**
- `template_id`: Template ID

**Request Body:**
```json
{
  "description": "Updated description",
  "quality_score": 0.95
}
```

#### DELETE /api/v1/templates/{template_id}

Delete a template (soft delete).

**Path Parameters:**
- `template_id`: Template ID

**Response:**
- Status: 204 No Content

### Search

#### POST /api/v1/search

Search templates by query string.

**Query Parameters:**
- `query` (required): Search query
- `stack` (optional): Filter by stack
- `tags` (optional): Filter by tags (comma-separated)
- `limit` (optional): Maximum results (default: 20, max: 100)

**Example:**
```bash
curl "http://localhost:8000/api/v1/search?query=react&stack=frontend"
```

**Response:**
```json
{
  "success": true,
  "message": "Found 1 templates for 'react'",
  "data": {
    "query": "react",
    "templates": [
      {
        "name": "test-template",
        "description": "A test template",
        "stack": "frontend",
        "path": "/templates/default",
        "upstream_url": null,
        "version": "0.1.0",
        "author": "Unknown",
        "license": "MIT",
        "tags": ["test"],
        "technologies": ["react", "typescript"],
        "features": [],
        "dependencies": {},
        "min_python_version": null,
        "min_node_version": null,
        "stars": 50,
        "forks": 10,
        "growth_rate": 0.0,
        "quality_score": 0.8,
        "created_at": "2025-10-14T14:32:59.838181",
        "updated_at": "2025-10-14T14:32:59.838181"
      }
    ],
    "total_count": 1
  }
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "message": "Error description",
  "data": {
    "error": "Detailed error information"
  }
}
```

### Common HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `204 No Content`: Request successful, no content returned
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Limit**: 100 requests per minute per IP address
- **Headers**: Rate limit information is included in response headers
  - `x-ratelimit-limit`: Maximum requests per window
  - `x-ratelimit-remaining`: Remaining requests in current window
  - `x-ratelimit-reset`: Time when the rate limit resets

## Interactive Documentation

The API provides interactive documentation through Swagger UI:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Examples

### Python Client

```python
import requests

# List templates
response = requests.get("http://localhost:8000/api/v1/templates")
templates = response.json()

# Search templates
response = requests.post("http://localhost:8000/api/v1/search?query=react")
results = response.json()

# Create a template
template_data = {
    "name": "my-template",
    "stack": "frontend",
    "description": "My custom template",
    "technologies": ["react", "typescript"]
}
response = requests.post("http://localhost:8000/api/v1/templates", json=template_data)
new_template = response.json()
```

### JavaScript Client

```javascript
// List templates
const templates = await fetch('http://localhost:8000/api/v1/templates')
  .then(response => response.json());

// Search templates
const results = await fetch('http://localhost:8000/api/v1/search?query=react')
  .then(response => response.json());

// Create a template
const templateData = {
  name: 'my-template',
  stack: 'frontend',
  description: 'My custom template',
  technologies: ['react', 'typescript']
};

const newTemplate = await fetch('http://localhost:8000/api/v1/templates', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(templateData)
}).then(response => response.json());
```

## Status

**Current Status**: 70% Production Ready

**Working Features**:
- ✅ Template CRUD operations
- ✅ Search functionality
- ✅ Health monitoring
- ✅ Rate limiting
- ✅ API documentation

**Pending Features**:
- ⏳ Authentication and authorization
- ⏳ File system integration
- ⏳ GitHub API integration
- ⏳ Advanced filtering and sorting
