# Template Heaven Microservices Architecture

A complete containerized microservices architecture that transforms Template Heaven into a distributed, scalable MCP (Model Context Protocol) server ecosystem.

## 🏗️ Architecture Overview

Template Heaven has been transformed from a monolithic repository into a **distributed microservices architecture** centered around the **Model Context Protocol (MCP)**. The system provides unified access to all template operations through standardized MCP interfaces while maintaining high availability, scalability, and maintainability.

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Template Heaven MCP Server                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    MCP Protocol Layer                   │    │
│  │  - WebSocket MCP Server                                 │    │
│  │  - Message Routing                                       │    │
│  │  - Protocol Validation                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Service Registry                        │    │
│  │  - Service Discovery                                    │    │
│  │  - Load Balancing                                       │    │
│  │  - Health Monitoring                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Middleware Stack                      │    │
│  │  - Authentication                                       │    │
│  │  - Rate Limiting                                        │    │
│  │  - Request Logging                                      │    │
│  │  - Error Handling                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Microservices Layer                        │
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Template     │ │Validation   │ │Generation   │ │Analysis     │ │
│  │Service      │ │Service      │ │Service      │ │Service      │ │
│  │Port: 8001   │ │Port: 8002   │ │Port: 8003   │ │Port: 8004   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │User         │ │Sync         │ │API Gateway  │ │Infrastructure│ │
│  │Service      │ │Service      │ │Service      │ │Services      │ │
│  │Port: 8005   │ │Port: 8006   │ │Port: 8007   │ │(DB, Cache,   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │ Monitoring)  │ │
│                                                 └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker 20.10+** with Docker Compose V2
- **8GB RAM** minimum (16GB recommended)
- **Git** for repository access
- **curl** for API testing

### Complete System Launch

```bash
# Clone the repository
git clone <repository-url>
cd template-heaven

# Launch the complete microservices stack
docker-compose -f docker-compose.microservices.yml up -d

# Check system health
curl http://localhost:3000/health
```

### Service Endpoints

| Service | Port | Health Check | Description |
|---------|------|--------------|-------------|
| **MCP Server** | 3000 | `ws://localhost:3000` | Central MCP protocol server |
| **Template Service** | 8001 | `http://localhost:8001/health` | Template management |
| **Validation Service** | 8002 | `http://localhost:8002/health` | Template validation |
| **Generation Service** | 8003 | `http://localhost:8003/health` | Project generation |
| **Analysis Service** | 8004 | `http://localhost:8004/health` | Code analysis |
| **User Service** | 8005 | `http://localhost:8005/health` | Authentication & users |
| **Sync Service** | 8006 | `http://localhost:8006/health` | Template synchronization |
| **API Gateway** | 80 | `http://localhost/health` | Central entry point |
| **Grafana** | 3001 | `http://localhost:3001` | Monitoring dashboards |
| **Prometheus** | 9090 | `http://localhost:9090` | Metrics collection |

## 🔧 MCP Protocol Integration

### Connecting to Template Heaven MCP

Template Heaven now serves as a complete MCP server that AI assistants can connect to for template operations.

#### WebSocket Connection

```javascript
// Connect to Template Heaven MCP Server
const ws = new WebSocket('ws://localhost:3000');

// Initialize connection
ws.onopen = () => {
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2024-11-05",
      capabilities: {
        tools: { listChanged: true },
        resources: { subscribe: true }
      },
      clientInfo: {
        name: "MyAIAssistant",
        version: "1.0.0"
      }
    }
  }));
};

// Handle MCP messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('MCP Response:', message);
};
```

#### Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `template-inspect` | Inspect template metadata | `template_name: string` |
| `template-search` | Search templates by keywords | `query: string, stack?: string` |
| `validate-template` | Validate template quality | `path: string, rules: string[]` |
| `generate-project` | Generate project from template | `template: string, destination: string` |
| `analyze-code` | Analyze code quality and metrics | `path: string, language: string` |
| `sync-templates` | Sync templates from upstream | `source: string, target: string` |

#### MCP Resources

| Resource URI | Description | MIME Type |
|--------------|-------------|-----------|
| `template://catalog` | Complete template catalog | `application/json` |
| `template://stacks` | Available template stacks | `application/json` |
| `project://generated/{id}` | Generated project details | `application/json` |
| `analysis://report/{id}` | Code analysis reports | `application/json` |

## 🏢 Microservices Architecture

### Service Responsibilities

#### **MCP Server** (`mcp-server/`)
- **Role**: Central protocol hub and middleware
- **Responsibilities**:
  - MCP WebSocket protocol handling
  - Request routing to microservices
  - Authentication and authorization
  - Rate limiting and security
  - Request/response logging
- **Technology**: Python, WebSockets, AsyncIO

#### **Template Service** (`services/template-service/`)
- **Role**: Template metadata and catalog management
- **Responsibilities**:
  - Template discovery and listing
  - Template metadata storage
  - Stack organization
  - Template search and filtering
- **Technology**: FastAPI, PostgreSQL, Redis

#### **Validation Service** (`services/validation-service/`)
- **Role**: Template quality assurance
- **Responsibilities**:
  - Template structure validation
  - Code quality checks
  - Security vulnerability scanning
  - Compliance verification
- **Technology**: Python, ESLint, Bandit, Safety

#### **Generation Service** (`services/generation-service/`)
- **Role**: Project scaffolding and generation
- **Responsibilities**:
  - Template instantiation
  - File generation and copying
  - Variable substitution
  - Dependency installation
- **Technology**: Python, Jinja2, Cookiecutter

#### **Analysis Service** (`services/analysis-service/`)
- **Role**: Code analysis and insights
- **Responsibilities**:
  - Code complexity analysis
  - Performance profiling
  - Security analysis
  - Best practice checking
- **Technology**: Python, Radon, Bandit, Pylint

#### **User Service** (`services/user-service/`)
- **Role**: Authentication and user management
- **Responsibilities**:
  - User registration and authentication
  - JWT token management
  - User profile management
  - Permission and role management
- **Technology**: FastAPI, PostgreSQL, Redis, JWT

#### **Sync Service** (`services/sync-service/`)
- **Role**: Template synchronization
- **Responsibilities**:
  - Upstream template monitoring
  - Template updates and patches
  - Version management
  - Change detection
- **Technology**: Python, GitPython, GitHub API

#### **API Gateway** (`services/api-gateway/`)
- **Role**: Central entry point and routing
- **Responsibilities**:
  - Request routing and load balancing
  - API composition
  - Rate limiting
  - Response caching
- **Technology**: Nginx, Lua, Redis

### Infrastructure Services

#### **PostgreSQL**: Primary data storage
- Template metadata, user data, analytics
- Connection pooling with PgBouncer

#### **Redis**: Caching and session storage
- Template cache, user sessions, rate limiting
- Pub/Sub for real-time notifications

#### **Prometheus**: Metrics collection
- Service health monitoring
- Performance metrics
- Custom business metrics

#### **Grafana**: Visualization and dashboards
- Real-time monitoring dashboards
- Alert management
- Custom panels and reports

## 🚀 Development Workflow

### Local Development

```bash
# Start all services
docker-compose -f docker-compose.microservices.yml up -d

# View service logs
docker-compose logs -f mcp-server

# Access individual services
curl http://localhost:8001/health  # Template service
curl http://localhost:8002/health  # Validation service

# Stop all services
docker-compose down
```

### Development with Specific Services

```bash
# Start only core services
docker-compose -f docker-compose.microservices.yml up -d mcp-server template-service postgres redis

# Start with monitoring
docker-compose -f docker-compose.microservices.yml --profile monitoring up -d

# Start with development tools
docker-compose -f docker-compose.microservices.yml --profile dev up -d
```

### Testing MCP Integration

```bash
# Test MCP connection
node test-mcp-connection.js

# Test template operations via MCP
node test-template-operations.js

# Load testing
npm run load-test
```

## 📊 Monitoring & Observability

### Health Checks

```bash
# Check all service health
curl http://localhost:80/health

# Individual service health
curl http://localhost:8001/health  # Template service
curl http://localhost:8002/health  # Validation service
# ... etc
```

### Metrics & Dashboards

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3001` (admin/admin123)
- **Service Metrics**: Each service exposes `/metrics` endpoint

### Logging

All services use structured JSON logging with:
- Request correlation IDs
- Service identification
- Error tracking
- Performance metrics

## 🔒 Security

### Authentication & Authorization

- **JWT-based authentication** across all services
- **Role-based access control** (RBAC)
- **API key management** for external integrations
- **OAuth2 support** for third-party logins

### Network Security

- **Service mesh** with mutual TLS
- **API Gateway** with request validation
- **Rate limiting** and DDoS protection
- **Network segmentation** between services

### Data Protection

- **Database encryption** at rest and in transit
- **Secure secrets management** with HashiCorp Vault
- **Audit logging** for sensitive operations
- **Data anonymization** for analytics

## 🔄 Service Communication

### Inter-Service Communication

Services communicate via:
- **REST APIs** for synchronous operations
- **gRPC** for high-performance internal calls
- **Redis Pub/Sub** for event-driven communication
- **Message queues** for asynchronous processing

### Service Discovery

- **Consul** for service registration and discovery
- **Health checking** with automatic failover
- **Load balancing** with round-robin and least-loaded strategies

## 🚀 Deployment

### Production Deployment

```bash
# Build all services
docker-compose -f docker-compose.microservices.yml build

# Deploy to production
docker-compose -f docker-compose.microservices.yml up -d

# Scale services as needed
docker-compose up -d --scale generation-service=3
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

### CI/CD Integration

The system includes GitHub Actions workflows for:
- **Automated testing** across all services
- **Security scanning** and vulnerability checks
- **Performance testing** and benchmarking
- **Docker image building** and publishing
- **Multi-environment deployment**

## 🧪 Testing Strategy

### Testing Pyramid

```
End-to-End Tests (E2E)
    ├── API Integration Tests
    │   ├── Component Tests
    │   │   ├── Unit Tests
    │   │   │   ├── Static Analysis
    │   │   │   └── Code Coverage
    │   │   └── Contract Tests
    │   └── Service Integration Tests
    └── User Journey Tests
```

### Test Categories

- **Unit Tests**: Individual function and class testing
- **Integration Tests**: Service-to-service communication
- **Contract Tests**: API contract validation
- **E2E Tests**: Complete user workflow testing
- **Performance Tests**: Load and stress testing
- **Chaos Engineering**: Failure injection testing

## 📈 Scaling & Performance

### Horizontal Scaling

```bash
# Scale individual services
docker-compose up -d --scale template-service=3
docker-compose up -d --scale generation-service=5

# Auto-scaling based on metrics
kubectl autoscale deployment template-service --cpu-percent=70 --min=2 --max=10
```

### Performance Optimization

- **Database indexing** and query optimization
- **Redis caching** for frequently accessed data
- **CDN integration** for static assets
- **Message queuing** for asynchronous processing
- **Circuit breakers** for fault tolerance

## 🔧 Maintenance & Operations

### Backup & Recovery

```bash
# Database backup
docker exec template-heaven-postgres pg_dump -U postgres template_heaven > backup.sql

# Service backup
docker-compose exec redis redis-cli save

# Full system backup
docker-compose exec backup backup-full-system
```

### Updates & Rollbacks

```bash
# Rolling updates
docker-compose up -d --no-deps template-service

# Rollback specific service
docker-compose up -d --no-deps template-service:previous-version

# Blue-green deployment
kubectl set image deployment/template-service template-service=new-version
```

## 📚 API Documentation

### MCP Protocol Documentation

- **MCP Specification**: Complete protocol documentation
- **Tool Reference**: All available MCP tools and their parameters
- **Resource Guide**: Available MCP resources and access patterns
- **Integration Examples**: Code samples for different languages

### Service APIs

Each microservice provides:
- **OpenAPI/Swagger** documentation at `/docs`
- **Health endpoints** at `/health`
- **Metrics endpoints** at `/metrics`
- **API versioning** with semantic versioning

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd template-heaven

# Start development environment
make dev-setup

# Run tests
make test-all

# Check code quality
make lint-all
```

### Service Development

```bash
# Create new service
make create-service name=my-service

# Add to docker-compose
make add-to-compose service=my-service

# Test service integration
make test-integration service=my-service
```

### MCP Tool Development

```bash
# Create new MCP tool
make create-mcp-tool name=my-tool service=my-service

# Register tool with MCP server
make register-tool tool=my-tool

# Test tool integration
make test-mcp-tool tool=my-tool
```

## 📞 Support & Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service logs
docker-compose logs template-service

# Check service health
curl http://localhost:8001/health

# Restart service
docker-compose restart template-service
```

#### MCP Connection Issues
```bash
# Check MCP server logs
docker-compose logs mcp-server

# Test MCP connection
websocat ws://localhost:3000

# Verify protocol compliance
npm run test-mcp-protocol
```

#### Database Issues
```bash
# Check database logs
docker-compose logs postgres

# Test database connection
docker-compose exec postgres psql -U postgres -d template_heaven

# Reset database
make db-reset
```

### Monitoring Commands

```bash
# System status
make status

# Service health
make health-check

# Performance metrics
make metrics

# Log aggregation
make logs service=all
```

## 🎯 Roadmap

### Phase 1 ✅ (Current)
- Core MCP server implementation
- Basic microservices architecture
- Template management services
- Container orchestration

### Phase 2 🔄 (Next)
- Advanced MCP protocol features
- Machine learning model serving
- Real-time collaboration features
- Advanced monitoring and tracing

### Phase 3 📋 (Future)
- Multi-cloud deployment
- Edge computing support
- AI-powered code generation
- Advanced analytics and insights

---

**Template Heaven** is now a complete **distributed MCP server ecosystem** that provides unified, scalable access to all template operations through standardized protocols. The microservices architecture ensures high availability, maintainability, and extensibility for future growth.

For detailed documentation on individual services, see the respective service README files. For MCP protocol integration, refer to the MCP specification and integration guides. 🚀
