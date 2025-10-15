# OpenAI Function Calling Template

A comprehensive containerized application that implements OpenAI's function calling capabilities, allowing AI assistants to execute functions and tools through structured API calls.

## 🌟 Features

- **OpenAI API Integration**: Full integration with GPT models supporting function calling
- **Function Registry**: Dynamic registration and management of callable functions
- **Tool Execution**: Safe execution of functions with proper error handling
- **Conversation Memory**: Persistent conversation context across interactions
- **Streaming Responses**: Real-time response streaming for better UX
- **Rate Limiting**: Built-in rate limiting to respect API quotas
- **Caching**: Response caching to reduce API calls and costs
- **Monitoring**: Comprehensive logging and metrics collection

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- Docker and Docker Compose (for containerized deployment)
- Redis (optional, for caching)

## 🚀 Quick Start

1. **Clone and setup:**
```bash
git checkout stack/ai-ml
cp -r openai-function-calling my-project
cd my-project
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
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
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "functions": ["get_weather"]
  }'
```

## 🏗️ Project Structure

```
openai-function-calling/
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
│   │   ├── openai_client.py     # OpenAI API client
│   │   ├── function_registry.py # Function registration
│   │   ├── tool_executor.py     # Tool execution engine
│   │   └── conversation.py      # Conversation management
│   ├── functions/               # Built-in functions
│   │   ├── weather.py
│   │   ├── calculator.py
│   │   ├── search.py
│   │   └── custom/              # Custom functions
│   └── utils/
│       ├── cache.py
│       ├── metrics.py
│       └── validation.py
├── tests/
│   ├── test_functions.py
│   ├── test_openai.py
│   └── test_integration.py
├── docs/
│   ├── api.md
│   ├── functions.md
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
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.7

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Function Configuration

Functions are defined in JSON format:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "default": "celsius"
      }
    },
    "required": ["location"]
  }
}
```

## 🔧 API Reference

### POST /chat

Main chat endpoint with function calling support.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather like in Paris?"}
  ],
  "functions": ["get_weather"],
  "stream": false,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chat_123",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The weather in Paris is currently sunny with 22°C.",
        "function_call": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 25,
    "total_tokens": 75
  }
}
```

### POST /functions/register

Register a new function.

```json
{
  "name": "custom_function",
  "description": "My custom function",
  "parameters": {...},
  "handler": "my_module.my_function"
}
```

### GET /functions

List all registered functions.

### GET /conversations/{conversation_id}

Retrieve conversation history.

## 🛠️ Built-in Functions

### Weather Function
- **Name**: `get_weather`
- **Description**: Get current weather information
- **Parameters**: location (string), unit (celsius/fahrenheit)
- **API**: OpenWeatherMap integration

### Calculator Function
- **Name**: `calculate`
- **Description**: Perform mathematical calculations
- **Parameters**: expression (string)
- **Engine**: SafeEval for secure computation

### Search Function
- **Name**: `web_search`
- **Description**: Search the web for information
- **Parameters**: query (string), num_results (integer)
- **API**: DuckDuckGo instant answers

## 🔒 Security

### Function Sandboxing
- Functions run in isolated environments
- Resource limits (CPU, memory, time)
- Network access restrictions
- Safe evaluation of expressions

### API Security
- Request validation and sanitization
- Rate limiting per API key
- Input size limits
- CORS configuration

### Data Protection
- No persistent storage of sensitive data
- Secure API key handling
- Audit logging of function calls

## 📊 Monitoring

### Metrics Collected
- Function call counts and durations
- API usage statistics
- Error rates by function
- Response time percentiles

### Health Checks
- `/health` - Application health status
- `/metrics` - Prometheus metrics
- `/ready` - Readiness probe

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run
docker-compose up --build

# Scale the application
docker-compose up -d --scale app=3
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### Cloud Deployment
- **AWS**: ECS/Fargate with API Gateway
- **GCP**: Cloud Run with Cloud Functions
- **Azure**: Container Instances with Functions

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_functions.py -v
```

### Integration Tests
```bash
pytest tests/test_integration.py -v
```

### Load Testing
```bash
locust -f tests/load_test.py
```

## 📈 Performance

### Optimization Features
- **Response Caching**: Cache identical requests
- **Function Memoization**: Cache function results
- **Connection Pooling**: Reuse HTTP connections
- **Async Processing**: Non-blocking function execution

### Benchmarks
- **Response Time**: <500ms for cached responses
- **Throughput**: 1000+ requests/minute
- **Function Execution**: <2s for complex operations

## 🔧 Customization

### Adding Custom Functions

1. Create function implementation:
```python
def my_custom_function(param1: str, param2: int) -> dict:
    """Custom function implementation."""
    return {"result": f"Processed {param1} with {param2}"}
```

2. Register in configuration:
```json
{
  "name": "my_custom_function",
  "description": "My custom function",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {"type": "string"},
      "param2": {"type": "integer"}
    },
    "required": ["param1"]
  },
  "handler": "my_module.my_custom_function"
}
```

### Custom Embeddings
Support for custom embedding models for function similarity search.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Full API docs available at `/docs`
- **Issues**: GitHub issues for bug reports
- **Discussions**: GitHub discussions for questions
- **Slack**: Community support channel

## 🔄 Changelog

### v1.0.0
- Initial release with OpenAI function calling support
- Built-in weather, calculator, and search functions
- Docker containerization
- Comprehensive testing suite

### v0.9.0 (Beta)
- Core function calling implementation
- Basic function registry
- REST API interface
