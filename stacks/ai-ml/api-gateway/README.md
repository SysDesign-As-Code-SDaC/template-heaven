# API Gateway for MCP Middleware

A containerized API Gateway service that provides unified access to the MCP Middleware, with advanced routing, load balancing, security, and monitoring capabilities.

## 🌟 Features

- **Unified API Access**: Single entry point for all MCP server interactions
- **Intelligent Routing**: Smart request routing based on server capabilities and load
- **Load Balancing**: Distribute requests across multiple MCP middleware instances
- **Security Layer**: Authentication, authorization, and request validation
- **Rate Limiting**: Advanced rate limiting with different tiers and burst handling
- **Request Transformation**: Modify and transform requests before forwarding
- **Response Caching**: Intelligent caching of responses to improve performance
- **Monitoring & Observability**: Comprehensive metrics, logging, and tracing
- **Health Checks**: Automated health monitoring of backend services
- **Circuit Breaking**: Fault tolerance with automatic failover
- **API Versioning**: Support for multiple API versions
- **Documentation**: Auto-generated API documentation

## 📋 Prerequisites

- Docker and Docker Compose
- MCP Middleware running (can be containerized or standalone)
- Redis (for caching and rate limiting)
- PostgreSQL (optional, for advanced analytics)

## 🚀 Quick Start

1. **Configure the gateway:**
```bash
cp .env.example .env
# Edit .env with your MCP middleware URLs and configuration
```

2. **Run with Docker:**
```bash
docker-compose up -d
```

3. **Test the API:**
```bash
# Check health
curl http://localhost:8080/health

# List available servers
curl http://localhost:8080/api/servers

# Make a request through the gateway
curl -X POST "http://localhost:8080/api/mcp/filesystem" \
  -H "Content-Type: application/json" \
  -d '{"method": "list_tools", "params": {}}'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │────│   API Gateway   │────│ MCP Middleware  │
│                 │    │                 │    │   Instances     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Redis Cache   │
                       │                 │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Metrics & Logs │
                       └─────────────────┘
```

## ⚙️ Configuration

### Environment Variables

```bash
# Gateway Configuration
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8080
GATEWAY_WORKERS=4

# MCP Middleware Backends
MCP_BACKENDS=mcp1:http://mcp-middleware-1:8000,mcp2:http://mcp-middleware-2:8000

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET_KEY=your-secret-key
API_KEY_HEADER=X-API-Key
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Caching
CACHE_ENABLED=true
CACHE_TTL_SECONDS=300

# Monitoring
METRICS_ENABLED=true
LOG_LEVEL=INFO

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
```

### Backend Configuration

Configure multiple MCP middleware backends:

```yaml
backends:
  - name: primary
    url: http://mcp-middleware-1:8000
    weight: 3
    health_check: /health
    timeout: 30

  - name: secondary
    url: http://mcp-middleware-2:8000
    weight: 1
    health_check: /health
    timeout: 30

load_balancer:
  strategy: weighted_round_robin
  health_check_interval: 30
  unhealthy_threshold: 3
  healthy_threshold: 2
```

## 🔧 API Reference

### Gateway Endpoints

#### GET /health
Health check endpoint for the gateway.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "backends": {
    "total": 2,
    "healthy": 2,
    "unhealthy": 0
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET /api/servers
List all available MCP servers through connected backends.

**Response:**
```json
{
  "servers": [
    {
      "name": "filesystem",
      "backend": "primary",
      "status": "healthy",
      "tools_count": 5,
      "last_health_check": "2024-01-15T10:29:45Z"
    },
    {
      "name": "database",
      "backend": "primary",
      "status": "healthy",
      "tools_count": 8,
      "last_health_check": "2024-01-15T10:29:50Z"
    }
  ]
}
```

#### POST /api/mcp/{server_name}
Proxy requests to specific MCP servers.

**Request:**
```json
{
  "method": "call_tool",
  "params": {
    "tool_name": "read_file",
    "arguments": {
      "path": "/etc/hosts"
    }
  }
}
```

**Response:**
```json
{
  "result": [
    {
      "type": "text",
      "text": "127.0.0.1 localhost\n::1 localhost\n..."
    }
  ],
  "backend": "primary",
  "response_time": 0.245,
  "cached": false
}
```

#### GET /metrics
Prometheus metrics endpoint.

#### GET /docs
Interactive API documentation (Swagger UI).

### Authentication

The gateway supports multiple authentication methods:

#### API Key Authentication
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/servers
```

#### JWT Token Authentication
```bash
curl -H "Authorization: Bearer your-jwt-token" http://localhost:8080/api/servers
```

#### OAuth2 Authentication
```bash
curl -H "Authorization: Bearer oauth2-token" http://localhost:8080/api/servers
```

## 🏗️ Core Components

### Request Router
Intelligent routing engine that:
- **Load Balancing**: Distributes requests across healthy backends
- **Server Affinity**: Routes requests to appropriate server types
- **Failover**: Automatic failover to healthy backends
- **Circuit Breaking**: Prevents cascading failures

### Security Middleware
Comprehensive security layer with:
- **Input Validation**: Request sanitization and validation
- **Authentication**: Multiple auth methods support
- **Authorization**: Role-based access control
- **Rate Limiting**: Configurable rate limits per user/endpoint
- **Request Filtering**: IP allowlists and denylists

### Caching Layer
Intelligent caching system:
- **Response Caching**: Cache identical responses
- **Cache Invalidation**: Smart cache invalidation strategies
- **Cache Compression**: Reduce storage and transfer sizes
- **Cache Analytics**: Monitor cache hit rates and performance

### Monitoring & Observability
Complete observability stack:
- **Metrics Collection**: Request counts, response times, error rates
- **Distributed Tracing**: Request tracing across services
- **Structured Logging**: Consistent log format with correlation IDs
- **Health Monitoring**: Automated health checks for all components

## 🚀 Deployment

### Docker Compose Deployment
```yaml
version: '3.8'
services:
  api-gateway:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MCP_BACKENDS=mcp1:http://mcp-middleware-1:8000
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - mcp-middleware-1

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  mcp-middleware-1:
    # MCP middleware service configuration
    # ...
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: api-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: MCP_BACKENDS
          value: "mcp1:http://mcp-service-1:8000,mcp2:http://mcp-service-2:8000"
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### High Availability Setup
```yaml
# Multiple gateway instances with load balancer
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-lb
spec:
  selector:
    app: api-gateway
  ports:
    - port: 80
      targetPort: 8080
  type: LoadBalancer

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/ -v --cov=src
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8080/health

# Using wrk
wrk -t4 -c100 -d30s http://localhost:8080/api/servers
```

### End-to-End Testing
```bash
# Test complete request flow
python tests/e2e/test_full_flow.py
```

## 📊 Monitoring

### Metrics Dashboard
Access metrics at `/metrics` endpoint for Prometheus integration.

### Key Metrics
- **Request Rate**: Requests per second by endpoint
- **Response Time**: P95, P99 response times
- **Error Rate**: 4xx and 5xx error percentages
- **Backend Health**: Health status of MCP backends
- **Cache Hit Rate**: Cache effectiveness metrics
- **Circuit Breaker Status**: Circuit breaker state information

### Alerting Rules
```yaml
# Prometheus alerting rules
groups:
  - name: api_gateway
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: BackendUnhealthy
        expr: up{job="mcp_backend"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "MCP backend is down"
```

## 🔒 Security

### Authentication & Authorization
- **Multi-tenant Support**: Isolate requests by tenant/API key
- **Role-based Access**: Different permission levels for users
- **API Key Management**: Secure key generation and rotation
- **JWT Integration**: Support for external identity providers

### Request Security
- **Input Validation**: Comprehensive request validation
- **SQL Injection Prevention**: Parameterized queries
- **XSS Protection**: Output sanitization
- **CSRF Protection**: Token-based CSRF prevention

### Network Security
- **TLS Encryption**: End-to-end encryption
- **IP Filtering**: Allow/deny lists for source IPs
- **Rate Limiting**: Distributed rate limiting with Redis
- **DDoS Protection**: Request throttling and filtering

## 🔧 Customization

### Custom Routing Logic
```python
from gateway.routing import BaseRouter

class CustomRouter(BaseRouter):
    async def route_request(self, request, backends):
        """Custom routing logic based on request content."""
        # Analyze request content
        if request.method == "POST" and "database" in request.url.path:
            # Route database requests to specialized backend
            return self.get_database_backend(backends)
        elif self._is_heavy_computation(request):
            # Route heavy computations to GPU-enabled backends
            return self.get_gpu_backend(backends)
        else:
            # Default load balancing
            return await super().route_request(request, backends)
```

### Custom Authentication
```python
from gateway.auth import BaseAuthenticator

class CustomAuthenticator(BaseAuthenticator):
    async def authenticate(self, request):
        """Custom authentication logic."""
        # Check for custom headers
        custom_token = request.headers.get("X-Custom-Auth")
        if custom_token:
            # Validate custom token
            return await self.validate_custom_token(custom_token)

        # Fallback to standard auth
        return await super().authenticate(request)
```

### Custom Middleware
```python
from gateway.middleware import BaseMiddleware

class CustomMiddleware(BaseMiddleware):
    async def process_request(self, request):
        """Custom request processing."""
        # Add custom headers
        request.headers["X-Processed-By"] = "CustomMiddleware"

        # Log request details
        self.logger.info(f"Processing request: {request.method} {request.url}")

        return request

    async def process_response(self, response):
        """Custom response processing."""
        # Add custom response headers
        response.headers["X-Processed-At"] = str(datetime.utcnow())

        return response
```

## 🤝 Integration Examples

### With MCP Middleware
```python
# Direct integration with MCP middleware
from gateway.backends import MCPBackend

backend = MCPBackend(
    name="mcp-primary",
    url="http://mcp-middleware:8000",
    health_check_path="/health",
    capabilities=["filesystem", "database", "web"]
)

# Register backend
gateway.register_backend(backend)
```

### With External Services
```python
# Integration with external API management
from gateway.integrations import KongGateway

kong = KongGateway(
    admin_url="http://kong-admin:8001",
    services=[
        {
            "name": "mcp-api",
            "url": "http://api-gateway:8080",
            "routes": ["/api/*"],
            "plugins": ["cors", "rate-limiting"]
        }
    ]
)
```

### With Monitoring Systems
```python
# Integration with monitoring stack
from gateway.monitoring import DataDogIntegration

datadog = DataDogIntegration(
    api_key="your-datadog-api-key",
    app_key="your-datadog-app-key",
    metrics=[
        "request_count",
        "response_time",
        "error_rate",
        "backend_health"
    ]
)

gateway.set_monitoring_integration(datadog)
```

## 📈 Performance Optimization

### Caching Strategies
- **Multi-level Caching**: In-memory + Redis caching
- **Intelligent Invalidation**: Smart cache invalidation based on data changes
- **Cache Compression**: Reduce memory usage with compression
- **Cache Warming**: Pre-populate cache with frequently accessed data

### Optimization Techniques
- **Request Batching**: Batch multiple requests to backends
- **Response Streaming**: Stream large responses to clients
- **Connection Pooling**: Reuse connections to backends
- **Async Processing**: Non-blocking request processing

### Scaling Strategies
- **Horizontal Scaling**: Multiple gateway instances
- **Backend Sharding**: Distribute load across backend shards
- **Regional Deployment**: Deploy closer to users
- **Auto-scaling**: Scale based on load metrics

## 🔄 Advanced Features

### API Versioning
Support for multiple API versions with automatic routing:
```
/api/v1/mcp/filesystem
/api/v2/mcp/filesystem
```

### Request Transformation
Transform requests before forwarding to backends:
```python
# Add default parameters
request.params["api_version"] = "v1"

# Transform response format
response.data = self.transform_response(response.data)
```

### Circuit Breaker Pattern
Automatic failure detection and recovery:
```python
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=RequestException
)

@circuit_breaker
async def call_backend(request):
    return await self.backend.call(request)
```

### Service Discovery
Automatic backend discovery and registration:
```python
# Integration with service registries
etcd_client = etcd3.client(host='etcd', port=2379)

# Watch for backend changes
etcd_client.add_watch_callback('/backends/', self.on_backend_change)
```

## 🆘 Troubleshooting

### Common Issues

#### Backend Connection Failures
```bash
# Check backend health
curl http://localhost:8080/health

# Test backend connectivity
curl http://mcp-middleware-1:8000/health

# Check gateway logs
docker-compose logs api-gateway
```

#### High Latency
```bash
# Check cache hit rate
curl http://localhost:8080/metrics | grep cache

# Monitor backend response times
curl http://localhost:8080/metrics | grep response_time

# Check for circuit breaker activation
curl http://localhost:8080/health
```

#### Rate Limiting Issues
```bash
# Check current rate limits
curl http://localhost:8080/metrics | grep rate_limit

# Monitor Redis connection
redis-cli ping

# Adjust rate limits in configuration
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Enable request tracing
export TRACING_ENABLED=true

# Run with verbose output
docker-compose up --scale api-gateway=1
```

## 📄 License

Licensed under Apache 2.0 License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Complete docs at `/docs`
- **GitHub Issues**: Bug reports and feature requests
- **Community Forum**: Help and discussion
- **Professional Support**: Enterprise support available

## 🔄 Changelog

### v1.0.0
- Production-ready API Gateway for MCP middleware
- Load balancing and intelligent routing
- Comprehensive security and authentication
- Advanced monitoring and observability
- Containerized deployment with scaling

### v0.9.0 (Beta)
- Core gateway functionality
- Backend management and health checks
- Basic security and rate limiting
- Request/response caching
- API documentation generation

### v0.8.0 (Alpha)
- Initial gateway implementation
- Basic request routing
- Backend registration
- Health monitoring
- Proof of concept architecture
