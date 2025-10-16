# MCP Middleware Server

A containerized middleware application that accepts and manages multiple MCP (Model Context Protocol) servers, providing a unified interface for AI assistants to communicate with external tools and data sources.

## 🚀 Features

- **Complete MCP Protocol Support**: Full Model Context Protocol (2024-11-05) implementation
- **10 Built-in MCP Servers**: Filesystem, Database, Web, Git, API, Search, Execution, Vector, and ClickUp
- **Multi-Server Management**: Handle multiple MCP servers simultaneously with intelligent routing
- **Containerized Deployment**: Production-ready Docker setup with monitoring stack
- **Dynamic Configuration**: Runtime server management via REST API
- **Comprehensive Monitoring**: Health checks, metrics, and distributed tracing
- **Security Framework**: Authentication, authorization, rate limiting, and input validation
- **High Availability**: Load balancing, health checks, and failover mechanisms
- **ClickUp Integration**: Full ClickUp workspace, task, and team management capabilities

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for development)
- Node.js 18+ (for frontend components)

## 🛠️ Quick Start

### Development Setup

```bash
# Clone the repository
git checkout stack/ai-ml
cd stacks/ai-ml/mcp-middleware

# Copy environment configuration
cp env.example .env

# Edit configuration with your API keys
# Required: ClickUp API token, OpenAI API key (optional), etc.
nano .env

# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Check server health
curl http://localhost:8000/health
```

### Production Setup

```bash
# Use production compose file
docker-compose -f docker/docker-compose.prod.yml up -d

# Check all services are running
docker-compose ps

# View logs
docker-compose logs -f mcp-middleware
```

### Testing ClickUp Integration

```bash
# Test ClickUp server health
curl http://localhost:8000/api/servers/clickup/health

# Add ClickUp server configuration
curl -X POST http://localhost:8000/api/servers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "clickup-prod",
    "type": "clickup",
    "config": {},
    "auth": {
      "api_token": "your_clickup_api_token_here"
    }
  }'

# List ClickUp workspaces
curl -X POST http://localhost:8000/api/mcp/clickup-prod \
  -H "Content-Type: application/json" \
  -d '{
    "method": "get_workspaces",
    "params": {}
  }'
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn mcp_middleware.main:app --host 0.0.0.0 --port 8000 --reload

# Or use Docker
docker build -t mcp-middleware .
docker run -p 8000:8000 mcp-middleware
```

## 🚀 Deployment

### Docker Deployment Options

#### Development Environment
```bash
# Start all development services
docker-compose -f docker/docker-compose.dev.yml up -d

# View logs
docker-compose -f docker/docker-compose.dev.yml logs -f mcp-middleware

# Debug with attached debugger
docker-compose -f docker/docker-compose.dev.yml exec mcp-middleware python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn src.mcp_middleware.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Production Environment
```bash
# Start production stack with monitoring
docker-compose -f docker/docker-compose.prod.yml up -d

# Scale the application
docker-compose -f docker/docker-compose.prod.yml up -d --scale mcp-middleware=3

# Update with zero downtime
docker-compose -f docker/docker-compose.prod.yml pull
docker-compose -f docker/docker-compose.prod.yml up -d --no-deps mcp-middleware
```

#### High Availability Setup
```bash
# Deploy with load balancer
docker-compose -f docker/docker-compose.prod.yml -f docker/docker-compose.ha.yml up -d

# Access via load balancer
curl http://localhost/health  # Nginx load balancer
```

### Kubernetes Deployment

#### Basic Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-middleware
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-middleware
  template:
    metadata:
      labels:
        app: mcp-middleware
    spec:
      containers:
      - name: mcp-middleware
        image: mcp-middleware:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: mcp-config
        - secretRef:
            name: mcp-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

#### With Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mcp-middleware-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: mcp-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mcp-middleware-service
            port:
              number: 8000
```

### Manual Installation

#### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
cp env.example .env
# Edit .env with your settings

# Run database migrations (if needed)
python -m alembic upgrade head

# Start the server
uvicorn src.mcp_middleware.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Production Server
```bash
# Install production dependencies
pip install -r requirements.txt gunicorn

# Configure environment
export $(cat .env | xargs)

# Start with Gunicorn
gunicorn src.mcp_middleware.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## 📁 Project Structure

```
mcp-middleware/
├── src/
│   ├── mcp_middleware/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── core/
│   │   │   ├── server_manager.py # MCP server management
│   │   │   ├── protocol_handler.py # MCP protocol handling
│   │   │   ├── message_router.py # Request routing
│   │   │   └── health_monitor.py # Health monitoring
│   │   ├── servers/
│   │   │   ├── filesystem.py     # Filesystem MCP server
│   │   │   ├── database.py       # Database MCP server
│   │   │   ├── web.py            # Web scraping MCP server
│   │   │   └── git.py            # Git repository MCP server
│   │   ├── middleware/
│   │   │   ├── auth.py           # Authentication middleware
│   │   │   ├── rate_limit.py     # Rate limiting
│   │   │   ├── logging.py        # Request logging
│   │   │   └── cors.py           # CORS handling
│   │   └── utils/
│   │       ├── config.py         # Configuration management
│   │       ├── validation.py     # Request validation
│   │       └── metrics.py        # Metrics collection
│   ├── tests/
│   │   ├── test_server_manager.py
│   │   ├── test_protocol_handler.py
│   │   └── test_integration.py
│   └── scripts/
│       ├── setup_servers.py      # Server setup script
│       └── migrate_config.py     # Configuration migration
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── docs/
│   ├── api.md                    # API documentation
│   ├── servers.md                # Supported MCP servers
│   └── deployment.md             # Deployment guide
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

### Environment Variables

```bash
# Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8000
MCP_WORKERS=4

# Security
MCP_SECRET_KEY=your-secret-key-here
MCP_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Database (for server configurations)
DATABASE_URL=sqlite:///./mcp_servers.db

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20
```

### MCP Server Configuration

Add servers via API or configuration file:

```json
{
  "name": "example-server",
  "type": "filesystem",
  "version": "1.0",
  "enabled": true,
  "config": {
    "root_path": "/data",
    "allowed_operations": ["read", "list", "write"],
    "max_file_size": "10MB",
    "rate_limit": 100
  },
  "authentication": {
    "type": "bearer",
    "token": "your-token-here"
  }
}
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/test_integration.py -v

# Run with coverage
pytest --cov=mcp_middleware --cov-report=html

# Test with Docker
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 🚀 Deployment

### Production Deployment

```bash
# Build production image
docker build -f docker/Dockerfile.prod -t mcp-middleware:prod .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment health
curl https://your-domain.com/health
```

### Scaling

```bash
# Scale the application
docker-compose up -d --scale mcp-middleware=3

# Use load balancer for high availability
docker-compose -f docker-compose.lb.yml up -d
```

## 📚 API Reference

### Core Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /api/servers` - List MCP servers
- `POST /api/servers` - Add MCP server
- `DELETE /api/servers/{name}` - Remove MCP server
- `POST /api/mcp/{server_name}` - Send MCP request

### MCP Protocol Endpoints

- `POST /mcp/initialize` - Initialize MCP connection
- `POST /mcp/tools/list` - List available tools
- `POST /mcp/tools/call` - Call a tool
- `POST /mcp/resources/list` - List resources
- `POST /mcp/resources/read` - Read resource

## 🔒 Security

### Authentication
- Bearer token authentication for API access
- MCP server-specific authentication
- Rate limiting and request validation

### Data Protection
- Encrypted communication (HTTPS/TLS)
- Secure credential storage
- Input validation and sanitization
- Audit logging for all operations

## 📊 Monitoring

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# MCP server health
curl http://localhost:8000/api/servers/health

# Metrics
curl http://localhost:8000/metrics
```

### Logging
- Structured JSON logging
- Configurable log levels
- Request/response logging
- Error tracking and alerts

## 🔗 Supported MCP Servers

### Built-in Servers
- **Filesystem**: File system operations
- **Database**: SQL database access
- **Web**: Web scraping and API calls
- **Git**: Git repository operations

### Custom Servers
- Plugin architecture for custom MCP servers
- Dynamic loading of server implementations
- Configuration-based server setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This template is part of the Template Heaven project.

## 🔗 Related Templates

- [FastAPI Microservice](../fastapi-microservice/)
- [Python Web Service](../python-web-service/)
- [Docker Application](../docker-app/)

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review the troubleshooting guide
