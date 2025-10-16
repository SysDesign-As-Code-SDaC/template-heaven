# Anthropic Tool Use Template

A comprehensive containerized application that implements Anthropic's tool use capabilities, enabling AI assistants to use tools and functions through structured interactions with Claude models.

## 🌟 Features

- **Anthropic Claude Integration**: Full integration with Claude models supporting tool use
- **Tool Registry**: Dynamic registration and management of tools and functions
- **Safe Tool Execution**: Isolated execution environment with resource controls
- **Conversation Context**: Persistent conversation management with tool call history
- **Streaming Support**: Real-time response streaming for enhanced user experience
- **Rate Limiting**: Built-in rate limiting respecting API quotas
- **Caching Layer**: Intelligent response caching to optimize costs
- **Comprehensive Monitoring**: Detailed logging, metrics, and health checks

## 📋 Prerequisites

- Python 3.9+
- Anthropic API key
- Docker and Docker Compose (for containerized deployment)
- Redis (optional, for advanced caching)

## 🚀 Quick Start

1. **Clone and setup:**
```bash
git checkout stack/ai-ml
cp -r anthropic-tool-use my-project
cd my-project
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your Anthropic API key
```

3. **Run with Docker:**
```bash
docker-compose up -d
```

4. **Test the API:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Calculate 15 * 23 and tell me about the weather in London"}],
    "tools": ["calculator", "weather"]
  }'
```

## 🏗️ Project Structure

```
anthropic-tool-use/
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
│   │   ├── anthropic_client.py  # Anthropic API client
│   │   ├── tool_registry.py     # Tool registration system
│   │   ├── tool_executor.py     # Tool execution engine
│   │   └── conversation.py      # Conversation management
│   ├── tools/                   # Built-in tools
│   │   ├── calculator.py
│   │   ├── weather.py
│   │   ├── search.py
│   │   ├── file_ops.py
│   │   └── custom/              # Custom tools
│   └── utils/
│       ├── cache.py
│       ├── metrics.py
│       └── validation.py
├── tests/
│   ├── test_tools.py
│   ├── test_anthropic.py
│   └── test_integration.py
├── docs/
│   ├── api.md
│   ├── tools.md
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
# Anthropic Configuration
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229
ANTHROPIC_MAX_TOKENS=4000
ANTHROPIC_TEMPERATURE=0.7

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=50

# Tool Execution
TOOL_EXECUTION_TIMEOUT=30
TOOL_MEMORY_LIMIT_MB=100

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Tool Configuration

Tools are defined in JSON format:

```json
{
  "name": "get_weather",
  "description": "Get current weather information for a location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature unit"
      }
    },
    "required": ["location"]
  }
}
```

## 🔧 API Reference

### POST /chat

Main chat endpoint with tool use support.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What's 15 * 23 and what's the weather in London?"}
  ],
  "tools": ["calculator", "weather"],
  "stream": false,
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "id": "chat_123",
  "content": [
    {
      "type": "text",
      "text": "15 * 23 = 345, and the weather in London is currently cloudy with 18°C."
    }
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 45,
    "output_tokens": 32
  }
}
```

### POST /tools/register

Register a new tool.

```json
{
  "name": "custom_tool",
  "description": "My custom tool",
  "input_schema": {...},
  "handler": "my_module.my_tool"
}
```

### GET /tools

List all registered tools.

### POST /conversations

Create a new conversation.

### GET /conversations/{conversation_id}

Retrieve conversation with tool call history.

## 🛠️ Built-in Tools

### Calculator Tool
- **Name**: `calculator`
- **Description**: Perform mathematical calculations safely
- **Capabilities**: Basic arithmetic, advanced math functions, unit conversions
- **Safety**: Expression validation and timeout protection

### Weather Tool
- **Name**: `weather`
- **Description**: Get current weather and forecasts
- **Parameters**: location, timeframe, unit
- **API Integration**: OpenWeatherMap or similar service

### Web Search Tool
- **Name**: `web_search`
- **Description**: Search the web for information
- **Parameters**: query, num_results, safe_search
- **Backends**: DuckDuckGo, Bing, or Google Custom Search

### File Operations Tool
- **Name**: `file_ops`
- **Description**: Safe file system operations
- **Capabilities**: read, write, list, search files
- **Security**: Path validation and permission checks

## 🔒 Security

### Tool Sandboxing
- **Isolated Execution**: Tools run in separate processes or containers
- **Resource Limits**: CPU, memory, and time restrictions
- **Network Controls**: Configurable network access policies
- **Input Validation**: Strict parameter validation

### API Security
- **Request Validation**: Comprehensive input sanitization
- **Rate Limiting**: Per-client and global rate limits
- **Authentication**: API key and JWT support
- **Audit Logging**: Complete request/response logging

### Data Protection
- **No Data Persistence**: Sensitive data not stored
- **Secure Key Management**: Encrypted API key storage
- **Privacy Compliance**: GDPR and privacy regulation support

## 📊 Monitoring

### Metrics Collection
- **Tool Usage Statistics**: Call counts, success rates, durations
- **API Usage Tracking**: Token consumption, cost monitoring
- **Performance Metrics**: Response times, throughput
- **Error Tracking**: Failure rates and error types

### Health Endpoints
- `/health` - Overall application health
- `/metrics` - Prometheus-compatible metrics
- `/ready` - Kubernetes readiness probe
- `/tools/health` - Tool-specific health checks

## 🚀 Deployment

### Docker Deployment
```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anthropic-tool-use
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: anthropic-tool-use:latest
        envFrom:
        - secretRef:
            name: anthropic-secrets
```

### Cloud Deployment Options
- **AWS**: ECS with Fargate, API Gateway integration
- **GCP**: Cloud Run, Cloud Functions for tools
- **Azure**: Container Apps, Functions integration

## 🧪 Testing

### Test Categories
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Tool-specific tests
pytest tests/tools/test_calculator.py -v

# Load testing
pytest tests/load/ -v
```

### Test Coverage
- **Unit Tests**: >90% code coverage
- **Integration Tests**: Full API workflows
- **Tool Tests**: Each tool's functionality
- **Security Tests**: Input validation and sandboxing

## 📈 Performance

### Optimization Features
- **Response Caching**: Cache tool results and API responses
- **Connection Pooling**: Reuse HTTP connections
- **Async Execution**: Non-blocking tool calls
- **Result Memoization**: Cache expensive computations

### Performance Targets
- **API Response Time**: <1s for simple requests
- **Tool Execution**: <5s for complex operations
- **Concurrent Users**: 1000+ simultaneous connections
- **Throughput**: 500+ requests/minute

## 🔧 Customization

### Creating Custom Tools

1. **Implement the tool function:**
```python
def my_custom_tool(param1: str, param2: int) -> dict:
    """
    Custom tool implementation.

    Args:
        param1: First parameter
        param2: Second parameter

    Returns:
        Tool execution result
    """
    # Tool logic here
    result = {"output": f"Processed {param1} with {param2}"}
    return result
```

2. **Define the tool schema:**
```json
{
  "name": "my_custom_tool",
  "description": "Description of what the tool does",
  "input_schema": {
    "type": "object",
    "properties": {
      "param1": {
        "type": "string",
        "description": "Description of param1"
      },
      "param2": {
        "type": "integer",
        "description": "Description of param2"
      }
    },
    "required": ["param1", "param2"]
  },
  "handler": "my_module.my_custom_tool",
  "timeout": 30,
  "memory_limit": 50
}
```

3. **Register the tool:**
```python
from app.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register_tool({
    "name": "my_custom_tool",
    "description": "My custom tool",
    "input_schema": {...},
    "handler": "my_module.my_custom_tool"
})
```

### Tool Categories
- **Data Processing**: Transform, analyze, or manipulate data
- **API Integration**: Call external APIs and services
- **File Operations**: Read, write, and manipulate files
- **Computation**: Mathematical and scientific calculations
- **Search**: Information retrieval and search operations

## 🤝 Integration Examples

### With LangChain
```python
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool

# Integrate with existing tools
anthropic_llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key=api_key
)

# Use custom tools
tools = [my_custom_tool]
agent = initialize_agent(tools, anthropic_llm)
```

### With CrewAI
```python
from crewai import Agent, Task, Crew
from tools.anthropic_tools import AnthropicToolSet

# Create agent with Anthropic tool use
researcher = Agent(
    role="Research Assistant",
    goal="Gather and analyze information",
    backstory="Expert researcher with access to various tools",
    tools=AnthropicToolSet.get_tools(),
    llm=ChatAnthropic(model="claude-3-sonnet-20240229")
)
```

## 📚 Advanced Features

### Multi-turn Conversations
- **Context Preservation**: Maintain conversation state across turns
- **Tool Call History**: Track previous tool executions
- **Session Management**: Handle long-running conversations

### Tool Chaining
- **Sequential Execution**: Chain multiple tools together
- **Conditional Logic**: Execute tools based on previous results
- **Error Recovery**: Handle tool failures gracefully

### Custom Embeddings
- **Semantic Search**: Find relevant tools by description
- **Tool Recommendations**: Suggest tools based on user intent
- **Similarity Matching**: Match user queries to tool capabilities

## 🔄 Version Compatibility

### Claude Model Support
- **Claude 3 Opus**: Full tool use support
- **Claude 3 Sonnet**: Complete tool integration
- **Claude 3 Haiku**: Tool use with optimization
- **Claude 2**: Legacy tool use support

### Backward Compatibility
- **API Compatibility**: Maintain consistent interfaces
- **Tool Schema Evolution**: Handle tool definition changes
- **Migration Support**: Smooth upgrades between versions

## 🆘 Troubleshooting

### Common Issues
- **Tool Timeouts**: Increase timeout limits or optimize tool code
- **Memory Limits**: Adjust memory limits or optimize resource usage
- **API Rate Limits**: Implement request queuing and backoff strategies
- **Network Issues**: Configure proper proxy settings and retry logic

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m app.main --debug

# Check tool execution logs
docker-compose logs tool-executor
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-tool`
3. **Add comprehensive tests**
4. **Update documentation**
5. **Submit pull request**

## 📄 License

Licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Complete API docs at `/docs`
- **GitHub Issues**: Bug reports and feature requests
- **Community Forum**: Discussion and help
- **Professional Support**: Enterprise support options

## 🔄 Changelog

### v1.0.0
- Complete Anthropic tool use implementation
- Production-ready containerization
- Comprehensive testing suite
- Multi-tool support with sandboxing

### v0.9.0 (Beta)
- Core tool use functionality
- Basic tool registry and execution
- REST API interface
- Initial documentation
