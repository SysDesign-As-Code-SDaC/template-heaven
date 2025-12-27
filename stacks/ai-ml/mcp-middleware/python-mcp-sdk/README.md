# Python MCP SDK Template

A production-ready, containerized template for building Model Context Protocol (MCP) servers and clients using the official Python SDK.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![MCP](https://img.shields.io/badge/MCP-1.18.0+-green.svg)](https://modelcontextprotocol.io/)

## 🚀 Features

- **Full MCP Protocol Support** - Complete implementation of the Model Context Protocol
- **Containerized Deployment** - Ready-to-use Docker containers for development and production
- **Database Integration** - PostgreSQL with async SQLAlchemy and connection pooling
- **Authentication & Security** - JWT-based auth with role-based access control
- **Monitoring & Observability** - Prometheus metrics, health checks, and logging
- **Comprehensive Testing** - Unit tests, integration tests, and example clients
- **Development Tools** - Jupyter notebooks, database admin, and debugging tools
- **Production Ready** - Security best practices, error handling, and performance optimization

## 📋 Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- PostgreSQL 15+ (for production)
- Redis 7+ (for caching)

## 🛠️ Quick Start

### Using Docker Compose (Recommended)

1. **Clone and navigate to the template:**
   ```bash
   cd stacks/ai-ml/mcp-middleware/python-mcp-sdk
   ```

2. **Start the services:**
   ```bash
   docker-compose up -d
   ```

3. **Check the server status:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **Run the client example:**
   ```bash
   docker-compose run --rm mcp-client
   ```

### Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start the server:**
   ```bash
   python -m src.mcp_sdk.main server
   ```

4. **In another terminal, start the client:**
   ```bash
   python -m src.mcp_sdk.main client
   ```

## 🏗️ Architecture

### Core Components

- **MCP Server** (`src/mcp_sdk/core/server.py`) - Main MCP protocol server
- **MCP Client** (`src/mcp_sdk/core/client.py`) - MCP client implementation
- **Database Manager** (`src/mcp_sdk/core/database.py`) - Database operations
- **Authentication** (`src/mcp_sdk/core/auth.py`) - JWT auth and RBAC
- **Monitoring** (`src/mcp_sdk/core/monitoring.py`) - Metrics and health checks

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   MCP Server    │    │   Database      │
│                 │◄──►│                 │◄──►│   PostgreSQL    │
│ • Tool calls    │    │ • Tool registry │    │                 │
│ • Resources     │    │ • Auth & RBAC   │    │ • User data     │
│ • Prompts       │    │ • Monitoring    │    │ • Sessions      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │                 │
                       │ • Caching       │
                       │ • Sessions      │
                       └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Application
APP_NAME="Python MCP SDK Template"
APP_VERSION="1.0.0"
ENVIRONMENT="development"

# MCP Server
MCP_HOST="0.0.0.0"
MCP_PORT="8000"
MCP_LOG_LEVEL="INFO"
MCP_DEBUG="false"

# Database
DATABASE_URL="postgresql://mcp_user:mcp_pass@localhost:5432/mcp_db"
DATABASE_POOL_SIZE="20"
DATABASE_MAX_OVERFLOW="30"

# Redis
REDIS_URL="redis://localhost:6379/0"
REDIS_MAX_CONNECTIONS="20"

# Security
SECURITY_SECRET_KEY="your-secret-key-here"
SECURITY_JWT_ALGORITHM="HS256"
SECURITY_JWT_EXPIRATION_HOURS="24"

# Monitoring
MONITORING_ENABLED="true"
MONITORING_METRICS_PORT="9090"
MONITORING_LOG_REQUESTS="true"
```

### Docker Configuration

The template includes multiple Docker configurations:

- **`Dockerfile`** - Production server container
- **`Dockerfile.client`** - Client application container
- **`Dockerfile.jupyter`** - Development environment with Jupyter
- **`docker-compose.yml`** - Complete development stack

## 📚 Usage Examples

### Basic Server Usage

```python
from mcp_sdk import MCPServer

# Create server
server = MCPServer()

# Register a custom tool
async def my_tool(arguments):
    return f"Hello, {arguments.get('name', 'World')}!"

server.register_tool(
    name="greet",
    handler=my_tool,
    description="Greet someone",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name to greet"}
        }
    }
)

# Start server
await server.start()
```

### Basic Client Usage

```python
from mcp_sdk import MCPClient

# Create client
client = MCPClient("http://localhost:8000")
client.configure_http("http://localhost:8000")

# Connect and use
async with client.connection():
    # List available tools
    tools = client.list_tools()
    print(f"Available tools: {tools}")
    
    # Call a tool
    result = await client.call_tool("greet", {"name": "Alice"})
    print(f"Result: {result}")
    
    # Read a resource
    content = await client.read_resource("config://server")
    print(f"Server config: {content}")
```

### Command Line Interface

```bash
# Start server
mcp-sdk server --host 0.0.0.0 --port 8000

# Start client
mcp-sdk client --url http://localhost:8000

# Run tests
mcp-sdk test --url http://localhost:8000

# Show info
mcp-sdk info
```

## 🧪 Testing

### Run All Tests

```bash
# Using pytest
pytest tests/ -v

# Using Docker
docker-compose run --rm mcp-sdk-server pytest tests/ -v
```

### Test Categories

- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end workflow testing
- **Client Tests** - MCP client functionality
- **Server Tests** - MCP server functionality

### Example Test

```python
import pytest
from mcp_sdk import MCPClient

@pytest.mark.asyncio
async def test_echo_tool():
    client = MCPClient("http://localhost:8000")
    client.configure_http("http://localhost:8000")
    
    async with client.connection():
        result = await client.call_tool("echo", {"message": "test"})
        assert "test" in result[0].text
```

## 📊 Monitoring

### Health Checks

```bash
# Server health
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Available Metrics

- **Request metrics** - Count, duration, status codes
- **Tool execution** - Success/failure rates, duration
- **Resource access** - Read counts and performance
- **System metrics** - CPU, memory, disk usage
- **Connection metrics** - Active connections, errors

### Grafana Dashboard

Access Grafana at `http://localhost:3001` (admin/admin123) to view:
- System performance metrics
- Request patterns and errors
- Database performance
- Custom business metrics

## 🔒 Security

### Authentication

The template includes JWT-based authentication with:

- **Password hashing** using bcrypt
- **JWT tokens** with configurable expiration
- **Refresh tokens** for long-term sessions
- **Role-based access control** (Admin, User, ReadOnly, Guest)

### Security Features

- **Input validation** and sanitization
- **Rate limiting** to prevent abuse
- **CORS configuration** for web clients
- **Security headers** (HSTS, CSP, etc.)
- **Audit logging** for security events

### Production Security

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set secure environment variables
export SECURITY_SECRET_KEY="your-secure-secret-key"
export SECURITY_JWT_EXPIRATION_HOURS="1"
export MCP_DEBUG="false"
```

## 🚀 Deployment

### Production Deployment

1. **Build production image:**
   ```bash
   docker build -t mcp-sdk:latest .
   ```

2. **Deploy with Docker Compose:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Set up reverse proxy (nginx):**
   ```bash
   # Configure nginx for SSL termination and load balancing
   ```

4. **Monitor deployment:**
   ```bash
   # Check health
   curl https://your-domain.com/health
   
   # View metrics
   curl https://your-domain.com/metrics
   ```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-sdk
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-sdk
  template:
    metadata:
      labels:
        app: mcp-sdk
    spec:
      containers:
      - name: mcp-sdk
        image: mcp-sdk:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: database-url
```

## 📖 API Reference

### MCP Server API

#### Tools

- **`echo`** - Echo tool for testing
- **`health`** - Server health check
- **Custom tools** - Register your own tools

#### Resources

- **`config://server`** - Server configuration
- **Custom resources** - Register your own resources

#### Prompts

- **`help`** - Help prompt with topic support
- **Custom prompts** - Register your own prompts

### REST API Endpoints

- **`GET /health`** - Health check
- **`GET /metrics`** - Prometheus metrics
- **`GET /tools`** - List available tools
- **`GET /resources`** - List available resources
- **`GET /prompts`** - List available prompts

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite:** `pytest tests/ -v`
5. **Commit your changes:** `git commit -m 'Add amazing feature'`
6. **Push to the branch:** `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
flake8 src/ tests/
mypy src/

# Run tests
pytest tests/ -v --cov=src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) - The official MCP specification
- [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk) - Official Python SDK
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database ORM
- [Prometheus](https://prometheus.io/) - Monitoring system

## 📞 Support

- **Documentation:** [Template Heaven Docs](https://templateheaven.dev/docs/mcp-sdk)
- **Issues:** [GitHub Issues](https://github.com/templateheaven/python-mcp-sdk-template/issues)
- **Discussions:** [GitHub Discussions](https://github.com/templateheaven/python-mcp-sdk-template/discussions)

---

**Built with ❤️ by the Template Heaven team**
