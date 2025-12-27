# Template Heaven System Architecture & How It Works

## 🏗️ System Overview

Template Heaven is a **template management and discovery service** built with FastAPI, SQLAlchemy, and async Python. It provides APIs and services for managing code templates, technology stacks, and user authentication.

---

## 📐 Architecture Components

### 1. **Database Layer** (`templateheaven/database/`)

**Purpose**: Manages data persistence using SQLAlchemy with async support.

**Key Components**:
- **`connection.py`**: Database connection manager
  - Creates async SQLAlchemy engine
  - Manages database sessions
  - Initializes database schema (`init_database()`)
  - Uses SQLite by default (`templateheaven.db`)

- **`models.py`**: SQLAlchemy ORM models
  - `User`: User accounts and authentication
  - `Template`: Template metadata and information
  - `Stack`: Technology stack categories
  - `Role`, `UserRole`: Role-based access control
  - `APIKey`: API key management
  - `TemplateDownload`: Download tracking

**How It Works**:
```python
# Database initialization happens automatically on startup
await init_database()  # Creates all tables

# Sessions are managed via context managers
async with db_manager.get_session() as session:
    # Database operations here
```

**Database File**: `templateheaven.db` (SQLite)

---

### 2. **Service Layer** (`templateheaven/services/`)

#### **AuthService** (`auth_service.py`)

**Purpose**: Handles authentication, authorization, and user management.

**Key Features**:
- **Password Hashing**: Uses `passlib` with PBKDF2-SHA256
- **JWT Tokens**: Creates and validates JWT access tokens
- **User Management**: CRUD operations for users
- **Role Management**: Assigns and checks user roles
- **Session Management**: Tracks user sessions
- **API Keys**: Generates and manages API keys

**How It Works**:
```python
auth_service = AuthService()

# Hash password
hashed = auth_service.get_password_hash("password123")

# Verify password
is_valid = auth_service.verify_password("password123", hashed)

# Create JWT token
token = auth_service.create_access_token({"sub": "username", "user_id": "123"})

# Authenticate user
user = await auth_service.authenticate_user("username", "password")
```

**Authentication Flow**:
1. User provides username/password
2. Service hashes password and compares with stored hash
3. If valid, creates JWT token
4. Token is returned to client
5. Client includes token in `Authorization: Bearer <token>` header

#### **TemplateService** (`template_service.py`)

**Purpose**: Manages templates, stacks, and template operations.

**Key Features**:
- **Template CRUD**: Create, read, update, delete templates
- **Template Search**: Search templates by name, description, tags
- **Stack Management**: Organize templates by technology stack
- **Template Downloads**: Track template downloads
- **Template Validation**: Validates template metadata

**How It Works**:
```python
template_service = TemplateService()

# List templates (with pagination)
templates, total = await template_service.list_templates(
    limit=10, 
    offset=0,
    stack="frontend"
)

# Get specific template
template = await template_service.get_template("template-id")

# Create template
new_template = await template_service.create_template({
    "name": "my-template",
    "stack": "frontend",
    "description": "A great template"
})

# Search templates
results = await template_service.search_templates("react")
```

**Template Model Structure**:
- `id`: UUID identifier
- `name`: Template name
- `stack`: Technology stack (e.g., "frontend", "backend")
- `description`: Template description
- `path`: Local file path
- `upstream_url`: Source repository URL
- `version`: Template version
- `quality_score`: Quality rating
- `tags`: List of tags
- `technologies`: List of technologies used

---

### 3. **API Layer** (`templateheaven/api/`)

#### **Main Application** (`main.py`)

**Purpose**: FastAPI application entry point with middleware and route registration.

**Key Features**:
- **Lifespan Management**: Initializes database on startup, closes on shutdown
- **Middleware Stack**:
  - CORS middleware
  - Security middleware
  - Logging middleware
  - Rate limiting middleware
- **Route Registration**: Includes all API routes
- **OpenAPI Documentation**: Auto-generated API docs at `/docs`

**How It Works**:
```python
# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await init_database()
    
    yield
    
    # Shutdown: Close database connections
    await close_database()

# Create app
app = create_app()  # Configures FastAPI with all routes and middleware
```

**Startup Sequence**:
1. Load settings from environment variables
2. Initialize database (create tables if needed)
3. Register middleware (CORS, security, logging, rate limiting)
4. Register API routes
5. Start uvicorn server

#### **Routes** (`routes/`)

**Available Endpoints**:

1. **Health** (`health.py`):
   - `GET /api/v1/health` - Health check
   - `GET /api/v1/health/live` - Liveness probe
   - `GET /api/v1/health/ready` - Readiness probe

2. **Authentication** (`auth.py`):
   - `POST /api/v1/auth/login` - User login
   - `POST /api/v1/auth/register` - User registration
   - `GET /api/v1/auth/me` - Get current user info
   - `POST /api/v1/auth/logout` - Logout
   - `PUT /api/v1/auth/password` - Update password
   - `POST /api/v1/auth/password-reset` - Request password reset
   - `POST /api/v1/auth/password-reset/verify` - Verify reset token

3. **Templates** (`templates.py`):
   - `GET /api/v1/templates` - List templates
   - `GET /api/v1/templates/{id}` - Get template
   - `POST /api/v1/templates` - Create template
   - `PUT /api/v1/templates/{id}` - Update template
   - `DELETE /api/v1/templates/{id}` - Delete template
   - `GET /api/v1/templates/{id}/download` - Download template

4. **Stacks** (`stacks.py`):
   - `GET /api/v1/stacks` - List stacks
   - `GET /api/v1/stacks/{name}` - Get stack details

5. **Search** (`search.py`):
   - `GET /api/v1/search` - Search templates

6. **Population** (`populate.py`):
   - `POST /api/v1/populate` - Populate database with templates

#### **Dependencies** (`dependencies.py`)

**Purpose**: Dependency injection for FastAPI routes.

**Key Dependencies**:
- `get_settings()`: Returns application settings
- `get_current_user()`: Extracts user from JWT token (optional)
- `require_auth()`: Requires authentication (currently optional)
- `require_admin()`: Requires admin role
- `get_request_id()`: Generates unique request ID

**How It Works**:
```python
# Optional authentication (current behavior)
@router.get("/templates")
async def list_templates(
    current_user: Optional[User] = Depends(require_auth)
):
    # current_user is None if no auth provided
    # Can still access endpoint without authentication
    pass
```

**Authentication Flow**:
1. Client sends request with `Authorization: Bearer <token>` header
2. `get_current_user()` extracts token from header
3. `AuthService.verify_token()` validates token
4. User is loaded from database
5. User object is injected into route handler

---

### 4. **Core Models** (`templateheaven/core/`)

**Purpose**: Pydantic models for data validation and serialization.

**Key Models**:
- `Template`: Template data model
- `Stack`: Stack data model
- `APIResponse`: Standard API response format
- `HealthCheck`: Health check response

**How It Works**:
```python
# Pydantic models provide validation
template = Template(
    name="my-template",
    stack="frontend",
    description="A template"
)

# Models can be serialized to JSON
json_data = template.json()

# Models can be created from dictionaries
template = Template(**data_dict)
```

---

### 5. **Configuration** (`templateheaven/api/dependencies.py`)

**Settings Class**: Centralized configuration management.

**Key Settings**:
- `app_name`: Application name
- `host`, `port`: Server host and port
- `secret_key`: JWT secret key
- `database_url`: Database connection string
- `cors_origins`: CORS allowed origins
- `rate_limit_requests`: Rate limit configuration

**How It Works**:
```python
# Settings loaded from environment variables
settings = get_settings()

# Can be overridden with .env file or environment variables
# Example: DATABASE_URL=postgresql://... python app.py
```

---

## 🔄 Request Flow

### Example: Listing Templates

1. **Client Request**:
   ```
   GET /api/v1/templates?limit=10&stack=frontend
   ```

2. **FastAPI Routing**:
   - Request hits `templates.py` router
   - Middleware processes request (logging, rate limiting)

3. **Dependency Injection**:
   - `require_auth()` extracts user (optional)
   - `get_request_id()` generates request ID

4. **Service Layer**:
   - Route calls `template_service.list_templates()`
   - Service queries database using SQLAlchemy

5. **Database Query**:
   ```python
   # Service executes async query
   query = select(TemplateModel).where(
       TemplateModel.stack == "frontend"
   ).limit(10)
   result = await session.execute(query)
   templates = result.scalars().all()
   ```

6. **Response**:
   - Templates converted to Pydantic models
   - Serialized to JSON
   - Returned to client with status 200

---

## 🔐 Authentication System

### Current State: **Optional Authentication**

Authentication is currently **optional** - all endpoints work without authentication tokens. This is configured in `dependencies.py`:

```python
async def require_auth(current_user = Depends(get_current_user)):
    # Returns None if no auth provided (doesn't raise exception)
    return current_user
```

### How Authentication Works (When Enabled)

1. **Registration/Login**:
   ```
   POST /api/v1/auth/login
   {
     "username": "user",
     "password": "password"
   }
   ```

2. **Server Response**:
   ```json
   {
     "access_token": "eyJhbGciOiJIUzI1NiIs...",
     "token_type": "bearer"
   }
   ```

3. **Authenticated Request**:
   ```
   GET /api/v1/templates
   Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
   ```

4. **Token Validation**:
   - Token extracted from header
   - JWT decoded and verified
   - User loaded from database
   - User object injected into route

---

## 🗄️ Database Schema

### Tables

1. **users**: User accounts
   - `id` (UUID, primary key)
   - `username` (unique)
   - `email` (unique)
   - `hashed_password`
   - `is_active`, `is_superuser`
   - `created_at`, `updated_at`

2. **templates**: Template metadata
   - `id` (UUID, primary key)
   - `name`, `description`
   - `stack_id` (foreign key to stacks)
   - `path`, `upstream_url`
   - `version`, `quality_score`
   - `tags`, `technologies` (JSON)
   - `is_active`, `created_at`, `updated_at`

3. **stacks**: Technology stacks
   - `id` (UUID, primary key)
   - `name`, `display_name`
   - `description`
   - `technologies` (JSON)

4. **roles**: User roles
   - `id` (UUID, primary key)
   - `name` (unique)
   - `description`

5. **user_roles**: User-role relationships
   - `user_id`, `role_id` (foreign keys)

6. **api_keys**: API keys
   - `id` (UUID, primary key)
   - `user_id` (foreign key)
   - `key_hash` (hashed API key)
   - `is_active`, `expires_at`

7. **template_downloads**: Download tracking
   - `id` (UUID, primary key)
   - `template_id`, `user_id` (foreign keys)
   - `ip_address`, `user_agent`
   - `created_at`

---

## 🚀 Running the System

### 1. Initialize Database

```bash
python3 scripts/init_database.py
```

This creates `templateheaven.db` with all tables.

### 2. Start API Server

```bash
# Option 1: Direct Python
python3 -m templateheaven.api.main

# Option 2: Uvicorn
uvicorn templateheaven.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 4. Test Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List templates
curl http://localhost:8000/api/v1/templates

# Root endpoint
curl http://localhost:8000/
```

---

## 🧪 Testing

Run the test script:

```bash
python3 test_system.py
```

This tests:
- ✅ Database initialization and connection
- ✅ Template service functionality
- ✅ Auth service (password hashing, JWT tokens)

---

## 📝 Key Design Decisions

1. **Async/Await**: All database operations are async for better performance
2. **SQLAlchemy 2.0**: Modern async SQLAlchemy with type hints
3. **Pydantic Models**: Type-safe data validation and serialization
4. **Dependency Injection**: FastAPI's dependency system for clean code
5. **Optional Authentication**: Currently disabled for easier development
6. **SQLite Default**: Simple file-based database (can switch to PostgreSQL)
7. **Modular Architecture**: Services, routes, and models are separated

---

## 🔧 Configuration

Settings can be configured via:

1. **Environment Variables**:
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost/db"
   export SECRET_KEY="your-secret-key"
   ```

2. **`.env` File**:
   ```env
   DATABASE_URL=sqlite+aiosqlite:///./templateheaven.db
   SECRET_KEY=your-secret-key-change-in-production
   HOST=0.0.0.0
   PORT=8000
   ```

3. **Default Values**: Defined in `Settings` class

---

## 📚 Next Steps

1. **Populate Database**: Add templates using the populate endpoint
2. **Enable Authentication**: Modify `require_auth()` to raise exceptions
3. **Add Templates**: Use template service to create templates
4. **Configure Stacks**: Define technology stacks
5. **Set Up Production**: Switch to PostgreSQL, configure secrets

---

## 🎯 Summary

Template Heaven is a **well-structured, async FastAPI application** with:

- ✅ Database layer with SQLAlchemy
- ✅ Service layer for business logic
- ✅ API layer with FastAPI routes
- ✅ Authentication system (optional)
- ✅ Template and stack management
- ✅ Comprehensive error handling
- ✅ API documentation
- ✅ Testing infrastructure

The system is **ready for development** and can be extended with additional features as needed.

