# Agent-to-Agent Communication Protocols Template

A comprehensive containerized framework for implementing secure, scalable, and intelligent agent-to-agent communication protocols. Enables complex multi-agent systems with standardized messaging, state synchronization, and collaborative decision-making capabilities.

## 🌟 Features

- **Standardized Protocols**: Multiple communication protocols (HTTP, WebSocket, Message Queue)
- **State Synchronization**: Real-time state sharing and conflict resolution
- **Message Routing**: Intelligent message routing with priority handling
- **Security Framework**: End-to-end encryption and authentication
- **Scalability**: Horizontal scaling with load balancing and sharding
- **Fault Tolerance**: Automatic failover and recovery mechanisms
- **Monitoring**: Comprehensive observability and performance metrics
- **Protocol Negotiation**: Dynamic protocol selection based on capabilities
- **Quality of Service**: Configurable QoS levels and delivery guarantees

## 📋 Prerequisites

- Python 3.9+
- Redis (for message queuing and state management)
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL (optional, for advanced persistence)
- RabbitMQ (optional, for advanced messaging)

## 🚀 Quick Start

1. **Clone and setup:**
```bash
git checkout stack/ai-ml
cp -r agent-communication my-project
cd my-project
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Run with Docker:**
```bash
docker-compose up -d
```

4. **Test the API:**
```bash
curl -X POST "http://localhost:8000/agents/register" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_1",
    "capabilities": ["text_processing", "data_analysis"],
    "protocols": ["websocket", "http"]
  }'

curl -X POST "http://localhost:8000/messages/send" \
  -H "Content-Type: application/json" \
  -d '{
    "from_agent": "agent_1",
    "to_agent": "agent_2",
    "protocol": "websocket",
    "message": {
      "type": "task_request",
      "content": "Analyze this dataset",
      "priority": "high"
    }
  }'
```

## 🏗️ Project Structure

```
agent-communication/
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration management
│   │   ├── models.py            # Pydantic models
│   │   └── middleware/
│   │       ├── cors.py
│   │       ├── rate_limit.py
│   │       └── logging.py
│   ├── core/
│   │   ├── protocol_manager.py  # Protocol management
│   │   ├── message_router.py    # Message routing engine
│   │   ├── state_manager.py     # State synchronization
│   │   ├── agent_registry.py    # Agent discovery and registration
│   │   └── security_manager.py  # Security and authentication
│   ├── protocols/               # Communication protocols
│   │   ├── websocket/
│   │   │   ├── protocol.py
│   │   │   ├── handler.py
│   │   │   └── connection_manager.py
│   │   ├── http/
│   │   │   ├── protocol.py
│   │   │   ├── client.py
│   │   │   └── server.py
│   │   ├── mqtt/
│   │   │   ├── protocol.py
│   │   │   ├── publisher.py
│   │   │   └── subscriber.py
│   │   └── custom/              # Custom protocols
│   ├── agents/                  # Example agents
│   │   ├── base_agent.py
│   │   ├── worker_agent.py
│   │   ├── coordinator_agent.py
│   │   └── custom/              # Custom agents
│   └── utils/
│       ├── cache.py
│       ├── metrics.py
│       └── validation.py
├── data/
│   ├── messages/                # Message persistence
│   ├── state/                   # State storage
│   ├── registry/                # Agent registry
│   └── cache/                   # Cache storage
├── tests/
│   ├── test_protocols.py
│   ├── test_routing.py
│   ├── test_agents.py
│   ├── test_integration.py
│   └── test_performance.py
├── docs/
│   ├── api.md
│   ├── protocols.md
│   ├── agents.md
│   └── deployment.md
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## ⚙️ Configuration

### Environment Variables

```bash
# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_CLUSTER=false

# PostgreSQL Configuration (optional)
POSTGRES_URL=postgresql://user:pass@localhost/db

# RabbitMQ Configuration (optional)
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# Security Configuration
JWT_SECRET_KEY=your_secret_key
ENABLE_ENCRYPTION=true
ENCRYPTION_KEY=your_encryption_key

# Protocol Configuration
WEBSOCKET_MAX_CONNECTIONS=1000
HTTP_TIMEOUT=30
MQTT_KEEPALIVE=60

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
TRACING_ENABLED=true

# Scalability
ENABLE_SHARDING=false
SHARD_COUNT=4
LOAD_BALANCER=nginx
```

### Protocol Configuration

Protocols are configured in YAML format:

```yaml
protocols:
  websocket:
    enabled: true
    max_connections: 1000
    heartbeat_interval: 30
    compression: true
    security:
      ssl_enabled: true
      certificate_path: /path/to/cert.pem
      key_path: /path/to/key.pem

  http:
    enabled: true
    timeout: 30
    retry_attempts: 3
    rate_limiting:
      requests_per_minute: 100
      burst_limit: 20

  mqtt:
    enabled: true
    broker_url: mqtt://localhost:1883
    client_id: agent_communication_system
    keepalive: 60
    qos_levels:
      default: 1
      high_priority: 2

  custom_protocol:
    enabled: false
    implementation: my_custom_protocol.Protocol
    config:
      custom_param: value
```

## 🔧 API Reference

### POST /agents/register

Register an agent with the communication system.

**Request:**
```json
{
  "agent_id": "agent_123",
  "name": "Data Processor Agent",
  "capabilities": ["data_processing", "analysis", "reporting"],
  "protocols": ["websocket", "http", "mqtt"],
  "metadata": {
    "version": "1.0.0",
    "owner": "data_team",
    "environment": "production"
  },
  "endpoints": {
    "websocket": "ws://agent:8080/ws",
    "http": "http://agent:8080/api",
    "mqtt": "mqtt://agent:1883"
  }
}
```

**Response:**
```json
{
  "agent_id": "agent_123",
  "registration_id": "reg_456",
  "status": "registered",
  "protocols_accepted": ["websocket", "http"],
  "registered_at": "2024-01-15T10:30:00Z"
}
```

### POST /messages/send

Send a message between agents.

**Request:**
```json
{
  "message_id": "msg_789",
  "from_agent": "agent_123",
  "to_agent": "agent_456",
  "protocol": "websocket",
  "priority": "normal",
  "qos": 1,
  "ttl": 3600,
  "headers": {
    "correlation_id": "corr_123",
    "message_type": "task_request"
  },
  "body": {
    "task": "process_data",
    "data": {"dataset": "sales_2024", "format": "csv"},
    "requirements": {"deadline": "2024-01-15T12:00:00Z"}
  }
}
```

**Response:**
```json
{
  "message_id": "msg_789",
  "status": "sent",
  "routing_info": {
    "protocol_used": "websocket",
    "route_taken": ["router_1", "shard_2"],
    "estimated_delivery": "2024-01-15T10:30:01Z"
  },
  "tracking_id": "track_101"
}
```

### GET /messages/{message_id}/status

Check message delivery status.

**Response:**
```json
{
  "message_id": "msg_789",
  "status": "delivered",
  "delivered_at": "2024-01-15T10:30:05Z",
  "delivery_attempts": 1,
  "recipient_acknowledged": true,
  "acknowledged_at": "2024-01-15T10:30:10Z"
}
```

### POST /groups/create

Create an agent communication group.

```json
{
  "group_id": "analysis_team",
  "name": "Data Analysis Team",
  "agents": ["agent_123", "agent_456", "agent_789"],
  "communication_mode": "broadcast",
  "protocols": ["websocket"],
  "policies": {
    "max_message_size": 1048576,
    "message_retention": 86400,
    "member_permissions": {
      "send_messages": true,
      "invite_members": false,
      "moderate": false
    }
  }
}
```

### POST /state/sync

Synchronize agent state.

```json
{
  "agent_id": "agent_123",
  "state_type": "conversation_context",
  "state_data": {
    "current_task": "data_processing",
    "progress": 0.75,
    "last_updated": "2024-01-15T10:30:00Z"
  },
  "sync_mode": "incremental",
  "conflict_resolution": "last_write_wins"
}
```

### GET /protocols

List available communication protocols.

### POST /protocols/negotiate

Negotiate protocol capabilities between agents.

```json
{
  "agent_a": "agent_123",
  "agent_b": "agent_456",
  "preferred_protocols": ["websocket", "http"],
  "required_capabilities": ["encryption", "compression"],
  "qos_requirements": {
    "reliability": "at_least_once",
    "latency": "<100ms"
  }
}
```

## 🌐 Communication Protocols

### WebSocket Protocol
- **Real-time Communication**: Bidirectional, low-latency messaging
- **Connection Management**: Automatic reconnection and heartbeat monitoring
- **Message Compression**: Optional gzip compression for large messages
- **Security**: WSS support with certificate validation

### HTTP Protocol
- **Request-Response**: RESTful communication with proper HTTP semantics
- **Authentication**: JWT token and API key support
- **Rate Limiting**: Configurable rate limits per agent
- **Caching**: HTTP caching headers and ETag support

### MQTT Protocol
- **Publish-Subscribe**: Topic-based messaging with QoS levels
- **Lightweight**: Minimal protocol overhead for resource-constrained agents
- **Offline Buffering**: Message queuing for disconnected agents
- **Wildcard Subscriptions**: Flexible topic matching

### Custom Protocols
- **Extensibility**: Plugin architecture for custom protocols
- **Protocol Negotiation**: Automatic capability detection and selection
- **Performance Monitoring**: Built-in metrics collection for custom protocols

## 🤖 Agent Types

### Base Agent
- **Core Functionality**: Basic communication and state management
- **Protocol Support**: All standard protocols
- **Security**: Built-in authentication and encryption
- **Monitoring**: Health checks and performance metrics

### Worker Agent
- **Task Processing**: Execute assigned tasks and report results
- **Resource Management**: Monitor and report resource usage
- **Error Handling**: Comprehensive error reporting and recovery
- **Load Balancing**: Participate in distributed task processing

### Coordinator Agent
- **Task Distribution**: Assign tasks to worker agents
- **Progress Tracking**: Monitor task execution and report status
- **Resource Allocation**: Optimize resource usage across agents
- **Failure Recovery**: Handle agent failures and redistribute tasks

### Specialized Agents
- **Data Processing Agent**: Handle large-scale data processing tasks
- **Communication Agent**: Manage inter-agent communication routing
- **Monitoring Agent**: Collect and analyze system metrics
- **Security Agent**: Handle authentication and authorization

## 🚀 Deployment

### Docker Deployment
```bash
# Single node deployment
docker-compose up -d

# Clustered deployment
docker-compose -f docker-compose.cluster.yml up -d

# High availability with load balancer
docker-compose -f docker-compose.ha.yml up -d
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-communication
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: agent-communication:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: RABBITMQ_URL
          value: "amqp://rabbitmq-service:5672"
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
```

### Cloud Deployment
- **AWS**: ECS with ElastiCache and MSK (Managed Streaming for Kafka)
- **GCP**: Cloud Run with Memorystore and Pub/Sub
- **Azure**: Container Apps with Cache for Redis and Service Bus

## 🧪 Testing

### Test Categories
```bash
# Protocol tests
pytest tests/test_protocols.py -v

# Message routing tests
pytest tests/test_routing.py -v

# Agent communication tests
pytest tests/test_agents.py -v

# Integration tests
pytest tests/test_integration.py -v

# Performance tests
pytest tests/test_performance.py -v
```

### Test Scenarios
- **Protocol Negotiation**: Verify protocol selection and capability matching
- **Message Delivery**: Test reliable message delivery across protocols
- **State Synchronization**: Validate state consistency across agents
- **Failure Recovery**: Test system behavior during network failures
- **Scalability**: Performance testing with multiple concurrent agents

## 📊 Monitoring

### Metrics Collection
- **Message Metrics**: Delivery rates, latency, throughput, error rates
- **Protocol Metrics**: Connection counts, bandwidth usage, protocol efficiency
- **Agent Metrics**: Online status, message counts, response times
- **System Metrics**: CPU usage, memory usage, network I/O

### Observability
- **Distributed Tracing**: Request tracing across agent communications
- **Log Correlation**: Correlated logs with message and agent IDs
- **Performance Dashboards**: Real-time monitoring dashboards
- **Alerting**: Configurable alerts for system issues

### Health Checks
- `/health` - Overall system health
- `/ready` - Readiness for accepting traffic
- `/metrics` - Prometheus-compatible metrics
- `/protocols/health` - Protocol-specific health status

## 🔒 Security

### Authentication & Authorization
- **JWT Tokens**: Secure agent authentication with token-based access
- **API Keys**: Simple key-based authentication for API access
- **Certificate-based**: Mutual TLS for high-security environments
- **Role-based Access**: Granular permissions for different agent types

### Message Security
- **End-to-End Encryption**: AES-256 encryption for message contents
- **Protocol Encryption**: TLS/SSL encryption at protocol level
- **Message Signing**: Digital signatures for message integrity
- **Key Management**: Secure key distribution and rotation

### Network Security
- **Firewall Configuration**: Restrict network access to authorized agents
- **VPN Support**: Secure communication over VPN tunnels
- **IP Whitelisting**: Allow access only from trusted IP addresses
- **DDoS Protection**: Rate limiting and traffic filtering

## 🔧 Customization

### Creating Custom Protocols

1. **Implement protocol interface:**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio

class BaseProtocol(ABC):
    """Base class for communication protocols."""

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the protocol."""
        pass

    @abstractmethod
    async def send_message(self, message: Dict[str, Any], target: str) -> bool:
        """Send a message using this protocol."""
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive a message using this protocol."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the protocol connection."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Get protocol capabilities."""
        pass
```

2. **Implement custom protocol:**
```python
class CustomProtocol(BaseProtocol):
    def __init__(self):
        self.connection = None
        self.config = {}

    async def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        # Initialize custom protocol connection
        self.connection = await self._create_connection(config)
        return True

    async def send_message(self, message: Dict[str, Any], target: str) -> bool:
        # Custom message sending logic
        encoded_message = self._encode_message(message)
        return await self.connection.send(target, encoded_message)

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        # Custom message receiving logic
        raw_message = await self.connection.receive()
        if raw_message:
            return self._decode_message(raw_message)
        return None

    async def close(self) -> None:
        if self.connection:
            await self.connection.close()

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {
            "supports_encryption": True,
            "max_message_size": 1048576,
            "qos_levels": [0, 1, 2],
            "compression": True
        }
```

3. **Register custom protocol:**
```python
from app.protocol_manager import ProtocolManager

manager = ProtocolManager()
manager.register_protocol("custom", CustomProtocol)
```

### Creating Custom Agents

1. **Define agent class:**
```python
from app.agents.base_agent import BaseAgent
from typing import Dict, Any, Optional
import asyncio

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self.specialized_capabilities = config.get("specialized_capabilities", [])

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming messages."""
        message_type = message.get("type")

        if message_type == "custom_task":
            return await self._handle_custom_task(message)
        else:
            return await super().handle_message(message)

    async def _handle_custom_task(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom task messages."""
        task_data = message.get("task_data", {})

        # Custom task processing logic
        result = await self._process_custom_task(task_data)

        return {
            "type": "task_result",
            "task_id": message.get("task_id"),
            "result": result,
            "status": "completed"
        }

    async def _process_custom_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process custom task logic."""
        # Implement custom task processing
        return {"processed": True, "data": task_data}
```

2. **Register custom agent:**
```python
from app.agent_registry import AgentRegistry

registry = AgentRegistry()
registry.register_agent_type("custom_agent", CustomAgent)
```

## 🤝 Integration Examples

### With External Systems
```python
# External API integration
class APIBridgeAgent(BaseAgent):
    def __init__(self, agent_id: str, api_config: Dict[str, Any]):
        super().__init__(agent_id, api_config)
        self.api_client = self._create_api_client(api_config)

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if message.get("type") == "api_call":
            return await self._handle_api_call(message)
        return await super().handle_message(message)

    async def _handle_api_call(self, message: Dict[str, Any]) -> Dict[str, Any]:
        api_request = message.get("api_request", {})
        response = await self.api_client.request(
            method=api_request.get("method", "GET"),
            url=api_request.get("url"),
            data=api_request.get("data")
        )

        return {
            "type": "api_response",
            "request_id": message.get("request_id"),
            "response": response,
            "status": "completed"
        }
```

### With Message Queues
```python
# Message queue integration
class QueueAgent(BaseAgent):
    def __init__(self, agent_id: str, queue_config: Dict[str, Any]):
        super().__init__(agent_id, queue_config)
        self.queue_client = self._create_queue_client(queue_config)
        self.subscriptions = []

    async def initialize(self) -> bool:
        # Subscribe to relevant queues
        for queue_name in self.config.get("subscribe_to", []):
            await self.queue_client.subscribe(queue_name, self._handle_queue_message)
            self.subscriptions.append(queue_name)
        return True

    async def _handle_queue_message(self, message: Dict[str, Any]) -> None:
        # Process message from queue and potentially forward to other agents
        processed_message = await self._process_queue_message(message)

        if processed_message:
            await self.send_message(processed_message, processed_message["target_agent"])
```

### With Databases
```python
# Database integration
class DatabaseAgent(BaseAgent):
    def __init__(self, agent_id: str, db_config: Dict[str, Any]):
        super().__init__(agent_id, db_config)
        self.db_connection = self._create_db_connection(db_config)

    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if message.get("type") == "database_query":
            return await self._handle_database_query(message)
        return await super().handle_message(message)

    async def _handle_database_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        query = message.get("query", "")
        params = message.get("params", [])

        try:
            results = await self.db_connection.execute(query, params)
            return {
                "type": "query_result",
                "query_id": message.get("query_id"),
                "results": results,
                "status": "success"
            }
        except Exception as e:
            return {
                "type": "query_error",
                "query_id": message.get("query_id"),
                "error": str(e),
                "status": "failed"
            }
```

## 📈 Performance Optimization

### Optimization Techniques
- **Message Batching**: Group multiple messages for efficient transmission
- **Connection Pooling**: Reuse connections to reduce overhead
- **Message Compression**: Compress large messages to reduce bandwidth
- **Caching**: Cache frequently accessed data and routing information

### Scalability Features
- **Horizontal Scaling**: Multiple instances with load balancing
- **Sharding**: Distribute load across multiple database shards
- **Message Partitioning**: Route messages based on content or sender
- **Rate Limiting**: Prevent system overload with intelligent throttling

## 🔄 Advanced Features

### Protocol Negotiation
- **Capability Discovery**: Automatic detection of agent capabilities
- **Protocol Selection**: Choose optimal protocol based on requirements
- **Fallback Mechanisms**: Graceful degradation when preferred protocols unavailable
- **Dynamic Reconfiguration**: Adjust protocol usage based on network conditions

### Quality of Service
- **Delivery Guarantees**: At most once, at least once, exactly once delivery
- **Priority Queuing**: High-priority messages processed first
- **Message TTL**: Automatic message expiration and cleanup
- **Duplicate Detection**: Prevent duplicate message processing

### State Synchronization
- **Conflict Resolution**: Multiple strategies for handling concurrent updates
- **Consistency Models**: Configurable consistency levels (strong, eventual)
- **State Compression**: Efficient state representation and transmission
- **Snapshot Management**: Periodic state snapshots for recovery

## 🆘 Troubleshooting

### Common Issues
- **Connection Failures**: Check network connectivity and firewall settings
- **Message Loss**: Verify QoS settings and protocol reliability
- **Performance Degradation**: Monitor resource usage and scale as needed
- **Protocol Compatibility**: Ensure agents support required protocols

### Debug Tools
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Monitor message flow
curl http://localhost:8000/debug/messages

# Check protocol status
curl http://localhost:8000/protocols/status

# View agent connections
curl http://localhost:8000/agents/connections
```

### Diagnostic Commands
```bash
# Test protocol connectivity
python -m app.utils.diagnostics test_protocols

# Analyze message routing
python -m app.utils.diagnostics analyze_routing

# Performance profiling
python -m app.utils.diagnostics profile_performance
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-protocol`
3. **Implement with comprehensive tests**
4. **Update documentation and examples**
5. **Submit pull request with detailed description**

## 📄 License

Licensed under MIT License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Complete API docs at `/docs`
- **GitHub Issues**: Bug reports and feature requests
- **Community Forum**: Help and discussion
- **Professional Support**: Enterprise support available

## 🔄 Changelog

### v1.0.0
- Production-ready agent communication framework
- Multiple protocol support (WebSocket, HTTP, MQTT)
- Comprehensive security and monitoring
- Horizontal scaling and high availability
- Containerized deployment

### v0.9.0 (Beta)
- Core communication protocols
- Agent registration and discovery
- Message routing and delivery
- Basic security features
- API interface and monitoring

### v0.8.0 (Alpha)
- Initial protocol implementations
- Basic agent communication
- Message queuing system
- Proof of concept architecture
