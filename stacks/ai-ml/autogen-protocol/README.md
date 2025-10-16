# AutoGen Protocol Template

A comprehensive containerized application implementing Microsoft's AutoGen framework for multi-agent conversations and tool-integrated AI systems. Enables complex agent-to-agent interactions, automated task solving, and intelligent workflow orchestration.

## 🌟 Features

- **Multi-Agent Conversations**: Complex agent-to-agent communication protocols
- **Tool Integration**: Seamless integration with external tools and APIs
- **Automated Workflows**: Intelligent task decomposition and execution
- **Group Chat Support**: Multi-agent group conversations with coordination
- **Code Execution**: Safe code execution environments for agents
- **Human-in-the-Loop**: Configurable human intervention points
- **Conversation Memory**: Persistent conversation state and history
- **Customizable Agents**: Flexible agent creation with specialized roles
- **Workflow Orchestration**: Complex multi-step task automation

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key (for LLM integration)
- Docker and Docker Compose (for containerized deployment)
- Redis (optional, for conversation memory)
- PostgreSQL (optional, for advanced persistence)

## 🚀 Quick Start

1. **Clone and setup:**
```bash
git checkout stack/ai-ml
cp -r autogen-protocol my-project
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
curl -X POST "http://localhost:8000/conversations/start" \
  -H "Content-Type: application/json" \
  -d '{
    "agents": ["project_manager", "developer", "reviewer"],
    "task": "Create a Python web application with user authentication",
    "workflow": "software_development"
  }'
```

## 🏗️ Project Structure

```
autogen-protocol/
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
│   │   ├── autogen_manager.py   # AutoGen integration
│   │   ├── agent_factory.py     # Agent creation and management
│   │   ├── conversation_manager.py # Conversation orchestration
│   │   ├── workflow_engine.py   # Workflow execution
│   │   └── tool_integrator.py   # Tool integration layer
│   ├── agents/                  # Pre-built agents
│   │   ├── project_manager.py
│   │   ├── developer.py
│   │   ├── reviewer.py
│   │   ├── researcher.py
│   │   └── custom/              # Custom agents
│   ├── workflows/               # Workflow definitions
│   │   ├── software_dev.py
│   │   ├── research.py
│   │   ├── analysis.py
│   │   └── custom/              # Custom workflows
│   ├── tools/                   # Tool implementations
│   │   ├── code_executor.py
│   │   ├── file_manager.py
│   │   ├── api_client.py
│   │   ├── search_engine.py
│   │   └── custom/              # Custom tools
│   └── utils/
│       ├── cache.py
│       ├── metrics.py
│       └── validation.py
├── data/
│   ├── conversations/           # Conversation storage
│   ├── workflows/               # Workflow persistence
│   └── cache/                   # Cache storage
├── tests/
│   ├── test_agents.py
│   ├── test_workflows.py
│   ├── test_conversations.py
│   └── test_integration.py
├── docs/
│   ├── api.md
│   ├── agents.md
│   ├── workflows.md
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
# LLM Configuration
OPENAI_API_KEY=your_openai_key
DEFAULT_MODEL=gpt-4
TEMPERATURE=0.7
MAX_TOKENS=4000

# AutoGen Configuration
AUTOGEN_CODE_EXECUTION=local
AUTOGEN_TIMEOUT=300
AUTOGEN_MAX_ROUNDS=50

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false

# Storage Configuration
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://user:pass@localhost/db

# Security Configuration
API_KEY_SECRET=your_secret_key
ENABLE_HUMAN_IN_LOOP=true

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
```

### Agent Configuration

Agents are defined in YAML format:

```yaml
project_manager:
  name: "Project Manager"
  role: "Project coordination and task management"
  system_message: |
    You are a project manager responsible for coordinating software development tasks.
    Break down complex requirements into manageable tasks and assign them to appropriate agents.
  llm_config:
    model: "gpt-4"
    temperature: 0.3
  tools: ["task_planner", "progress_tracker"]
  human_input_mode: "NEVER"

developer:
  name: "Developer"
  role: "Software development and implementation"
  system_message: |
    You are a skilled developer who writes clean, efficient, and well-documented code.
    Follow best practices and ensure code quality.
  llm_config:
    model: "gpt-4"
    temperature: 0.1
  tools: ["code_writer", "code_executor", "file_manager"]
  human_input_mode: "TERMINATE"

reviewer:
  name: "Code Reviewer"
  role: "Code review and quality assurance"
  system_message: |
    You are a code reviewer who ensures code quality, security, and best practices.
    Provide constructive feedback and suggest improvements.
  llm_config:
    model: "gpt-3.5-turbo"
    temperature: 0.2
  tools: ["code_analyzer", "security_scanner"]
  human_input_mode: "NEVER"
```

## 🔧 API Reference

### POST /conversations/start

Start a new multi-agent conversation.

**Request:**
```json
{
  "agents": ["project_manager", "developer", "reviewer"],
  "task": "Create a REST API for user management with authentication",
  "workflow": "software_development",
  "human_input": false,
  "max_rounds": 20
}
```

**Response:**
```json
{
  "conversation_id": "conv_123",
  "status": "running",
  "agents": ["project_manager", "developer", "reviewer"],
  "task": "Create a REST API for user management with authentication",
  "started_at": "2024-01-15T10:30:00Z"
}
```

### GET /conversations/{conversation_id}

Get conversation status and messages.

**Response:**
```json
{
  "conversation_id": "conv_123",
  "status": "completed",
  "messages": [
    {
      "agent": "project_manager",
      "content": "I'll break this task into components...",
      "timestamp": "2024-01-15T10:30:05Z"
    },
    {
      "agent": "developer",
      "content": "Starting implementation of user authentication...",
      "timestamp": "2024-01-15T10:30:15Z"
    }
  ],
  "summary": "Successfully created user management API with authentication",
  "completed_at": "2024-01-15T10:45:30Z"
}
```

### POST /conversations/{conversation_id}/message

Send a message to an ongoing conversation.

```json
{
  "content": "Please add input validation to the API endpoints",
  "agent": "user"
}
```

### POST /workflows/execute

Execute a predefined workflow.

```json
{
  "workflow_id": "code_review",
  "parameters": {
    "repository_url": "https://github.com/user/repo",
    "branch": "main",
    "focus_areas": ["security", "performance", "maintainability"]
  }
}
```

### GET /agents

List available agents.

### POST /agents/register

Register a custom agent.

```json
{
  "name": "custom_agent",
  "config": {
    "role": "Custom specialized role",
    "system_message": "You are a specialized agent...",
    "llm_config": {"model": "gpt-4"},
    "tools": ["custom_tool"]
  }
}
```

## 🤖 Built-in Agents

### Project Manager Agent
- **Role**: Task coordination and project management
- **Capabilities**: Task decomposition, progress tracking, resource allocation
- **Tools**: Task planner, progress tracker, deadline manager

### Developer Agent
- **Role**: Software development and implementation
- **Capabilities**: Code writing, debugging, testing, documentation
- **Tools**: Code executor, file manager, version control, linter

### Reviewer Agent
- **Role**: Code review and quality assurance
- **Capabilities**: Code analysis, security scanning, performance review
- **Tools**: Code analyzer, security scanner, test runner

### Researcher Agent
- **Role**: Information gathering and research
- **Capabilities**: Web research, data analysis, report generation
- **Tools**: Web search, document reader, data analyzer

## 🔄 Workflow Types

### Software Development Workflow
- **Stages**: Planning → Development → Testing → Review → Deployment
- **Agents Involved**: Project Manager, Developer, Reviewer
- **Deliverables**: Code, tests, documentation, deployment scripts

### Research Workflow
- **Stages**: Topic Analysis → Information Gathering → Data Synthesis → Report Generation
- **Agents Involved**: Researcher, Analyst, Writer
- **Deliverables**: Research reports, data analysis, recommendations

### Analysis Workflow
- **Stages**: Data Collection → Processing → Analysis → Visualization → Insights
- **Agents Involved**: Data Scientist, Analyst, Visualizer
- **Deliverables**: Analysis reports, charts, recommendations

## 🛠️ Tool Integration

### Code Execution Tools
- **Language Support**: Python, JavaScript, Java, C++, Go, Rust
- **Safety Features**: Sandboxed execution, resource limits, timeout protection
- **Capabilities**: Code running, testing, debugging, performance profiling

### File Management Tools
- **Operations**: Read, write, modify, search, organize files
- **Formats**: Text, JSON, XML, CSV, binary files
- **Security**: Path validation, permission checks, backup creation

### API Integration Tools
- **Protocols**: REST, GraphQL, SOAP, WebSocket
- **Authentication**: API keys, OAuth, JWT, Basic Auth
- **Features**: Request/response handling, error management, retry logic

### Search and Research Tools
- **Sources**: Web search, document databases, code repositories
- **Capabilities**: Keyword search, semantic search, filtering, ranking
- **Integration**: Google, Bing, DuckDuckGo, internal databases

## 🚀 Deployment

### Docker Deployment
```bash
# Development environment
docker-compose -f docker-compose.dev.yml up

# Production environment
docker-compose -f docker-compose.prod.yml up -d

# GPU-enabled for local models
docker-compose -f docker-compose.gpu.yml up -d
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autogen-protocol
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: autogen-protocol:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Cloud Deployment Options
- **AWS**: ECS with Fargate, EKS with AutoGen scaling
- **GCP**: Cloud Run, Vertex AI integration
- **Azure**: Container Apps, OpenAI service integration

## 🧪 Testing

### Test Categories
```bash
# Unit tests for individual components
pytest tests/unit/ -v --cov=src

# Integration tests for agent interactions
pytest tests/integration/ -v

# Workflow end-to-end tests
pytest tests/workflows/ -v

# Load and performance tests
pytest tests/performance/ -v
```

### Test Scenarios
- **Agent Communication**: Message passing between agents
- **Tool Execution**: Tool calling and result handling
- **Workflow Completion**: End-to-end workflow execution
- **Error Handling**: Failure scenarios and recovery
- **Human Intervention**: Human-in-the-loop functionality

## 📊 Monitoring

### Metrics Collection
- **Agent Performance**: Response times, success rates, error rates
- **Conversation Metrics**: Message counts, conversation duration, completion rates
- **Tool Usage**: Tool call frequency, execution times, failure rates
- **Workflow Analytics**: Stage completion times, bottleneck identification

### Health Monitoring
- `/health` - Application health status
- `/metrics` - Prometheus-compatible metrics
- `/conversations/health` - Conversation system status
- `/agents/health` - Agent health and availability

## 🔒 Security

### Execution Sandboxing
- **Isolated Environments**: Code execution in containers or VMs
- **Resource Limits**: CPU, memory, disk, and network restrictions
- **Access Controls**: File system and network access restrictions

### Data Protection
- **Conversation Encryption**: End-to-end encryption for sensitive conversations
- **Access Logging**: Comprehensive audit logging
- **Data Retention**: Configurable conversation history retention

### Authentication & Authorization
- **API Key Authentication**: Secure API access
- **Role-Based Access**: Different permission levels for users
- **Conversation Privacy**: Private conversations with access controls

## 🔧 Customization

### Creating Custom Agents

1. **Define agent class:**
```python
from autogen import AssistantAgent
from typing import Dict, Any

class CustomAgent(AssistantAgent):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            system_message=config.get("system_message"),
            llm_config=config.get("llm_config"),
            human_input_mode=config.get("human_input_mode", "NEVER")
        )
        self.specialized_tools = config.get("tools", [])

    def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        # Custom message processing logic
        enhanced_message = self.enhance_message(message, context)
        return super().process_message(enhanced_message)
```

2. **Register the agent:**
```python
from app.agent_factory import AgentFactory

factory = AgentFactory()
factory.register_agent_type("custom_agent", CustomAgent)
```

### Creating Custom Workflows

1. **Define workflow class:**
```python
from app.workflows.base import BaseWorkflow
from typing import List, Dict, Any

class CustomWorkflow(BaseWorkflow):
    def __init__(self, agents: List[str], config: Dict[str, Any]):
        super().__init__(agents, config)
        self.stages = ["analysis", "planning", "execution", "review"]

    def execute_stage(self, stage: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Custom stage execution logic
        if stage == "analysis":
            return self.analyze_requirements(context)
        elif stage == "planning":
            return self.create_plan(context)
        # ... other stages
```

2. **Register the workflow:**
```python
from app.workflow_engine import WorkflowEngine

engine = WorkflowEngine()
engine.register_workflow("custom_workflow", CustomWorkflow)
```

### Custom Tools

1. **Create tool class:**
```python
from autogen import tool
from typing import str

@tool
def custom_tool(query: str, options: Dict[str, Any] = None) -> str:
    """
    Custom tool for specialized functionality.

    Args:
        query: Input query
        options: Additional options

    Returns:
        Tool execution result
    """
    # Tool implementation
    result = perform_custom_operation(query, options)
    return f"Custom operation result: {result}"
```

2. **Register with agents:**
```python
# Add to agent configuration
agent_config = {
    "tools": ["custom_tool"],
    "tool_config": {
        "custom_tool": {
            "enabled": True,
            "timeout": 30,
            "retries": 3
        }
    }
}
```

## 🤝 Integration Examples

### With External Systems
```python
# Database integration
class DatabaseAgent(AssistantAgent):
    def __init__(self, db_config):
        super().__init__("Database Agent", llm_config=llm_config)
        self.db_connection = create_db_connection(db_config)

    @tool
    def query_database(self, sql_query: str) -> str:
        """Execute SQL query safely."""
        try:
            results = self.db_connection.execute(sql_query)
            return format_results(results)
        except Exception as e:
            return f"Database error: {str(e)}"
```

### With APIs
```python
# API integration
class APIAgent(AssistantAgent):
    def __init__(self, api_config):
        super().__init__("API Agent", llm_config=llm_config)
        self.api_client = create_api_client(api_config)

    @tool
    def call_external_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> str:
        """Make API calls with proper error handling."""
        try:
            response = self.api_client.request(method, endpoint, json=data)
            return format_api_response(response)
        except Exception as e:
            return f"API call failed: {str(e)}"
```

### With File Systems
```python
# File system integration
class FileSystemAgent(AssistantAgent):
    def __init__(self, fs_config):
        super().__init__("File System Agent", llm_config=llm_config)
        self.allowed_paths = fs_config.get("allowed_paths", [])

    @tool
    def read_file(self, file_path: str) -> str:
        """Read file content safely."""
        if not self._is_path_allowed(file_path):
            return "Access denied: Path not allowed"

        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"File read error: {str(e)}"
```

## 📈 Performance Optimization

### Optimization Techniques
- **Agent Caching**: Cache agent responses and tool results
- **Conversation Chunking**: Break long conversations into manageable chunks
- **Parallel Execution**: Run independent tasks concurrently
- **Resource Pooling**: Reuse expensive resources (database connections, API clients)

### Scaling Strategies
- **Horizontal Scaling**: Multiple instances behind load balancer
- **Agent Specialization**: Different agents for different workloads
- **Workflow Optimization**: Streamline conversation flows
- **Caching Layers**: Multi-level caching for frequently used data

## 🔄 Advanced Features

### Human-in-the-Loop
- **Intervention Points**: Configurable points for human input
- **Approval Workflows**: Human approval for critical decisions
- **Feedback Integration**: Human feedback into agent learning
- **Escalation Paths**: Automatic escalation to human operators

### Dynamic Agent Creation
- **Runtime Agent Creation**: Create agents based on task requirements
- **Agent Cloning**: Create specialized copies of existing agents
- **Agent Evolution**: Agents that learn and adapt over time
- **Agent Marketplace**: Share and reuse agent configurations

### Complex Workflow Orchestration
- **Conditional Branching**: Different paths based on intermediate results
- **Loop Constructs**: Repeat operations until conditions are met
- **Parallel Processing**: Multiple agents working on different aspects
- **State Management**: Complex state tracking across workflow stages

## 🆘 Troubleshooting

### Common Issues
- **Agent Communication**: Check message routing and serialization
- **Tool Execution**: Verify tool permissions and resource limits
- **Memory Issues**: Monitor conversation size and implement cleanup
- **Timeout Problems**: Adjust timeout settings and optimize operations

### Debug Tools
```bash
# Enable detailed logging
export AUTOGEN_LOG_LEVEL=DEBUG

# Monitor agent conversations
curl http://localhost:8000/conversations/{id}/debug

# Check agent health
curl http://localhost:8000/agents/health

# View workflow status
curl http://localhost:8000/workflows/{id}/status
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-agent-type`
3. **Implement with tests**: Add comprehensive test coverage
4. **Update documentation**: Keep docs synchronized with code
5. **Submit pull request**: Provide detailed description and examples

## 📄 License

Licensed under MIT License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Complete API docs at `/docs`
- **GitHub Issues**: Bug reports and feature requests
- **Community Discord**: Real-time help and discussion
- **Professional Support**: Enterprise support options available

## 🔄 Changelog

### v1.0.0
- Production-ready AutoGen integration
- Multi-agent conversation system
- Comprehensive workflow engine
- Tool integration framework
- Containerized deployment

### v0.9.0 (Beta)
- Core AutoGen functionality
- Basic agent communication
- Tool execution framework
- Workflow orchestration
- API interface

### v0.8.0 (Alpha)
- Initial AutoGen integration
- Basic agent creation
- Simple tool support
- Proof of concept workflows
