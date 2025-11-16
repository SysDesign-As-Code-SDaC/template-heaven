# CrewAI Workflows Template

A comprehensive containerized application implementing CrewAI framework for collaborative AI agent workflows. Enables complex multi-agent collaboration, role-based task execution, and intelligent process automation with human oversight capabilities.

## 🌟 Features

- **Multi-Agent Collaboration**: Complex agent-to-agent workflows with role specialization
- **Role-Based Task Assignment**: Intelligent task distribution based on agent capabilities
- **Workflow Orchestration**: Advanced process automation with conditional logic
- **Human-in-the-Loop**: Configurable human intervention and approval workflows
- **Dynamic Crew Formation**: Runtime crew assembly based on task requirements
- **Tool Integration**: Extensive tool library with secure execution
- **Memory Management**: Persistent conversation and task state management
- **Performance Monitoring**: Comprehensive metrics and workflow analytics
- **Scalable Architecture**: Horizontal scaling with load balancing

## 📋 Prerequisites

- Python 3.9+
- OpenAI or Anthropic API key (for LLM integration)
- Docker and Docker Compose (for containerized deployment)
- Redis (optional, for memory and caching)
- PostgreSQL (optional, for workflow persistence)

## 🚀 Quick Start

1. **Clone and setup:**
```bash
git checkout stack/ai-ml
cp -r crewai-workflows my-project
cd my-project
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Run with Docker:**
```bash
docker-compose up -d
```

4. **Test the API:**
```bash
curl -X POST "http://localhost:8000/crews/start" \
  -H "Content-Type: application/json" \
  -d '{
    "crew_config": {
      "name": "Software Development Team",
      "agents": ["product_manager", "senior_developer", "qa_engineer"],
      "tasks": ["analyze_requirements", "implement_solution", "test_and_validate"]
    },
    "goal": "Build a secure user authentication system",
    "max_iterations": 10
  }'
```

## 🏗️ Project Structure

```
crewai-workflows/
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
│   │   ├── crewai_manager.py    # CrewAI integration
│   │   ├── agent_factory.py     # Agent creation and management
│   │   ├── task_orchestrator.py # Task distribution and monitoring
│   │   ├── workflow_engine.py   # Workflow execution
│   │   └── memory_system.py     # Memory and state management
│   ├── agents/                  # Agent definitions
│   │   ├── product_manager.py
│   │   ├── developer.py
│   │   ├── designer.py
│   │   ├── qa_engineer.py
│   │   ├── researcher.py
│   │   └── custom/              # Custom agents
│   ├── crews/                   # Pre-built crews
│   │   ├── software_dev_crew.py
│   │   ├── research_crew.py
│   │   ├── marketing_crew.py
│   │   └── custom/              # Custom crews
│   ├── tasks/                   # Task definitions
│   │   ├── analysis_tasks.py
│   │   ├── development_tasks.py
│   │   ├── testing_tasks.py
│   │   └── custom/              # Custom tasks
│   ├── tools/                   # Tool implementations
│   │   ├── development_tools.py
│   │   ├── research_tools.py
│   │   ├── communication_tools.py
│   │   └── custom/              # Custom tools
│   └── utils/
│       ├── cache.py
│       ├── metrics.py
│       └── validation.py
├── data/
│   ├── crews/                   # Crew configurations
│   ├── workflows/               # Workflow executions
│   ├── memory/                  # Memory storage
│   └── cache/                   # Cache storage
├── tests/
│   ├── test_agents.py
│   ├── test_crews.py
│   ├── test_tasks.py
│   ├── test_integration.py
│   └── test_performance.py
├── docs/
│   ├── api.md
│   ├── agents.md
│   ├── crews.md
│   ├── tasks.md
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
ANTHROPIC_API_KEY=your_anthropic_key
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4
TEMPERATURE=0.7
MAX_TOKENS=4000

# CrewAI Configuration
CREWAI_MAX_ITERATIONS=25
CREWAI_VERBOSE=true
CREWAI_MEMORY=true
CREWAI_CACHE=true

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
APPROVAL_REQUIRED_FOR_PRODUCTION=true

# Monitoring
LOG_LEVEL=INFO
METRICS_ENABLED=true
TRACING_ENABLED=false
```

### Agent Configuration

Agents are defined in YAML format:

```yaml
product_manager:
  name: "Product Manager"
  role: "Product strategy and requirements analysis"
  goal: "Define clear product requirements and guide development"
  backstory: |
    Experienced product manager with 10+ years in software development.
    Expert at translating business needs into technical requirements.
    Strong focus on user experience and business value.
  llm: "gpt-4"
  tools:
    - requirements_analyzer
    - user_research
    - prioritization_matrix
  allow_delegation: true
  verbose: true

senior_developer:
  name: "Senior Developer"
  role: "Software architecture and implementation"
  goal: "Write clean, efficient, and maintainable code"
  backstory: |
    Senior software engineer with expertise in multiple programming languages.
    Follows best practices for code quality, testing, and documentation.
    Strong advocate for scalable and maintainable architectures.
  llm: "gpt-4"
  tools:
    - code_writer
    - code_reviewer
    - testing_framework
    - documentation_generator
  allow_delegation: false
  verbose: true

qa_engineer:
  name: "QA Engineer"
  role: "Quality assurance and testing"
  goal: "Ensure software quality through comprehensive testing"
  backstory: |
    Quality assurance specialist with focus on automated testing,
    performance testing, and bug tracking. Ensures software meets
    quality standards and user requirements.
  llm: "gpt-3.5-turbo"
  tools:
    - test_case_generator
    - automated_tester
    - performance_analyzer
    - bug_tracker
  allow_delegation: false
  verbose: true
```

## 🔧 API Reference

### POST /crews/start

Start a new crew workflow.

**Request:**
```json
{
  "crew_config": {
    "name": "Software Development Crew",
    "agents": ["product_manager", "senior_developer", "qa_engineer"],
    "process": "sequential",
    "manager_llm": "gpt-4"
  },
  "tasks": [
    {
      "description": "Analyze user requirements and create detailed specifications",
      "agent": "product_manager",
      "expected_output": "Detailed requirements document with user stories and acceptance criteria"
    },
    {
      "description": "Implement the authentication system based on requirements",
      "agent": "senior_developer",
      "expected_output": "Complete authentication system with API endpoints and database models"
    },
    {
      "description": "Test the authentication system and create test reports",
      "agent": "qa_engineer",
      "expected_output": "Comprehensive test report with coverage metrics and bug analysis"
    }
  ],
  "goal": "Build a secure and user-friendly authentication system",
  "max_iterations": 15
}
```

**Response:**
```json
{
  "crew_id": "crew_123",
  "status": "running",
  "crew_name": "Software Development Crew",
  "agents": ["product_manager", "senior_developer", "qa_engineer"],
  "tasks_count": 3,
  "started_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T11:15:00Z"
}
```

### GET /crews/{crew_id}

Get crew execution status and results.

**Response:**
```json
{
  "crew_id": "crew_123",
  "status": "completed",
  "progress": {
    "completed_tasks": 3,
    "total_tasks": 3,
    "current_iteration": 8,
    "max_iterations": 15
  },
  "results": [
    {
      "task": "Requirements Analysis",
      "agent": "product_manager",
      "status": "completed",
      "output": "Detailed requirements document...",
      "duration": 245
    },
    {
      "task": "Implementation",
      "agent": "senior_developer",
      "status": "completed",
      "output": "Authentication system implementation...",
      "duration": 420
    },
    {
      "task": "Testing",
      "agent": "qa_engineer",
      "status": "completed",
      "output": "Test results and quality metrics...",
      "duration": 180
    }
  ],
  "summary": "Successfully completed authentication system development",
  "completed_at": "2024-01-15T11:02:30Z"
}
```

### POST /crews/{crew_id}/interrupt

Interrupt running crew execution.

```json
{
  "reason": "Need human review before proceeding",
  "action": "pause_for_approval"
}
```

### POST /crews/{crew_id}/resume

Resume interrupted crew execution.

```json
{
  "approved": true,
  "comments": "Looks good, proceed with implementation",
  "modifications": {
    "max_iterations": 20
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
    "role": "Specialized role",
    "goal": "Specific objective",
    "backstory": "Agent background and expertise",
    "llm": "gpt-4",
    "tools": ["custom_tool"],
    "allow_delegation": true
  }
}
```

### GET /crews/templates

Get available crew templates.

### POST /crews/from-template

Create crew from template.

```json
{
  "template_id": "software_development",
  "customizations": {
    "goal": "Build a custom CRM system",
    "agents": {
      "senior_developer": {
        "llm": "gpt-4-turbo",
        "additional_tools": ["database_designer"]
      }
    }
  }
}
```

## 🤖 Built-in Agents

### Product Manager Agent
- **Role**: Product strategy and requirements gathering
- **Capabilities**: Requirements analysis, user story creation, prioritization
- **Tools**: Requirements analyzer, user research, prioritization matrix

### Senior Developer Agent
- **Role**: Software architecture and implementation
- **Capabilities**: Code writing, architecture design, technical documentation
- **Tools**: Code writer, code reviewer, documentation generator, testing framework

### QA Engineer Agent
- **Role**: Quality assurance and testing
- **Capabilities**: Test planning, automated testing, bug tracking, performance testing
- **Tools**: Test case generator, automated tester, performance analyzer, bug tracker

### Research Agent
- **Role**: Information gathering and analysis
- **Capabilities**: Web research, data analysis, competitive analysis, report writing
- **Tools**: Web search, document analyzer, data visualizer, report generator

### Designer Agent
- **Role**: UI/UX design and user experience
- **Capabilities**: Wireframing, prototyping, user testing, design systems
- **Tools**: Design tools, user testing, accessibility checker

## 🚀 Crew Types

### Sequential Crew
- **Process**: Tasks executed one after another
- **Use Case**: Linear workflows with dependencies
- **Example**: Software development pipeline

### Parallel Crew
- **Process**: Tasks executed simultaneously
- **Use Case**: Independent tasks that can run concurrently
- **Example**: Multi-market research analysis

### Hierarchical Crew
- **Process**: Manager agent coordinates worker agents
- **Use Case**: Complex projects requiring oversight
- **Example**: Large-scale system development

### Collaborative Crew
- **Process**: Agents work together with shared context
- **Use Case**: Creative and brainstorming tasks
- **Example**: Content creation and marketing campaigns

## 🛠️ Tool Library

### Development Tools
- **CodeWriter**: Generate code from specifications
- **CodeReviewer**: Review and suggest improvements
- **TestGenerator**: Create comprehensive test suites
- **DocumentationGenerator**: Generate technical documentation

### Research Tools
- **WebSearch**: Intelligent web searching and summarization
- **DocumentAnalyzer**: Extract insights from documents
- **DataVisualizer**: Create charts and graphs from data
- **ReportGenerator**: Generate comprehensive reports

### Communication Tools
- **EmailSender**: Send professional emails
- **SlackNotifier**: Send notifications to Slack channels
- **ReportExporter**: Export reports in various formats
- **PresentationCreator**: Generate presentation materials

### Specialized Tools
- **DatabaseDesigner**: Design database schemas
- **APIDesigner**: Design REST API specifications
- **SecurityAuditor**: Perform security assessments
- **PerformanceAnalyzer**: Analyze system performance

## 🚀 Deployment

### Docker Deployment
```bash
# Development environment
docker-compose -f docker-compose.dev.yml up

# Production environment
docker-compose -f docker-compose.prod.yml up -d

# High-availability setup
docker-compose -f docker-compose.ha.yml up -d
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crewai-workflows
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  template:
    spec:
      containers:
      - name: app
        image: crewai-workflows:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

### Cloud Deployment Options
- **AWS**: ECS/Fargate with Application Load Balancer
- **GCP**: Cloud Run with Cloud Build automation
- **Azure**: Container Apps with Azure Functions integration

## 🧪 Testing

### Test Structure
```bash
# Unit tests for individual components
pytest tests/unit/ -v --cov=src

# Integration tests for crew workflows
pytest tests/integration/ -v

# Performance tests for scalability
pytest tests/performance/ -v

# End-to-end workflow tests
pytest tests/e2e/ -v
```

### Test Scenarios
- **Single Agent Tasks**: Individual agent capability testing
- **Multi-Agent Collaboration**: Crew interaction and communication
- **Workflow Completion**: End-to-end process execution
- **Error Handling**: Failure scenarios and recovery mechanisms
- **Human Intervention**: Human-in-the-loop functionality
- **Scalability Testing**: Concurrent crew execution

## 📊 Monitoring

### Metrics Collection
- **Crew Performance**: Execution times, success rates, iteration counts
- **Agent Metrics**: Task completion rates, tool usage statistics
- **Workflow Analytics**: Bottleneck identification, optimization opportunities
- **Resource Usage**: Memory, CPU, API costs tracking

### Health Endpoints
- `/health` - Overall application health
- `/metrics` - Prometheus-compatible metrics
- `/crews/health` - Crew execution system status
- `/agents/health` - Agent availability and performance

### Logging and Tracing
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Distributed Tracing**: Request tracing across microservices
- **Audit Logging**: Complete audit trail for compliance
- **Performance Monitoring**: Real-time performance dashboards

## 🔒 Security

### Execution Security
- **Sandboxed Tool Execution**: Isolated tool execution environments
- **Resource Limits**: CPU, memory, and network restrictions
- **Input Validation**: Comprehensive input sanitization and validation
- **Access Controls**: Role-based permissions for crew operations

### Data Protection
- **Encryption**: End-to-end encryption for sensitive data
- **Data Retention**: Configurable data retention policies
- **Privacy Compliance**: GDPR and privacy regulation support
- **Audit Trails**: Complete logging for security monitoring

### API Security
- **Authentication**: JWT token-based authentication
- **Authorization**: Granular permission system
- **Rate Limiting**: API rate limiting and abuse prevention
- **Input Validation**: Request validation and sanitization

## 🔧 Customization

### Creating Custom Agents

1. **Define agent class:**
```python
from crewai import Agent
from typing import List, Dict, Any

class CustomAgent(Agent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            llm=config.get("llm", "gpt-4"),
            tools=config.get("tools", []),
            allow_delegation=config.get("allow_delegation", False),
            verbose=config.get("verbose", True)
        )
        self.specialized_capabilities = config.get("specialized_capabilities", [])

    def execute_task(self, task: str, context: Dict[str, Any] = None) -> str:
        # Custom task execution logic
        enhanced_task = self.enhance_task_description(task, context)
        return super().execute_task(enhanced_task)
```

2. **Register the agent:**
```python
from app.agent_factory import AgentFactory

factory = AgentFactory()
factory.register_agent("custom_agent", CustomAgent, config)
```

### Creating Custom Tools

1. **Implement tool class:**
```python
from crewai.tools import BaseTool
from typing import Any, Dict

class CustomTool(BaseTool):
    name: str = "custom_tool"
    description: str = "A custom tool for specialized functionality"

    def _run(self, **kwargs) -> str:
        """
        Execute the custom tool logic.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool execution result
        """
        # Custom tool implementation
        parameter = kwargs.get("parameter", "")
        result = perform_custom_operation(parameter)
        return f"Custom tool result: {result}"

    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "description": "Input parameter for the tool"
                    }
                },
                "required": ["parameter"]
            }
        }
```

2. **Register with agent factory:**
```python
from app.agent_factory import AgentFactory

factory = AgentFactory()
factory.register_tool(CustomTool())
```

### Creating Custom Crews

1. **Define crew configuration:**
```python
from crewai import Crew, Process
from typing import List, Dict, Any

class CustomCrew:
    def __init__(self, config: Dict[str, Any]):
        self.name = config["name"]
        self.agents = config["agents"]
        self.tasks = config["tasks"]
        self.process = config.get("process", Process.sequential)

    def create_crew(self, agent_factory, task_factory) -> Crew:
        """Create and configure the crew."""
        # Get agent instances
        agent_instances = [
            agent_factory.create_agent(agent_name)
            for agent_name in self.agents
        ]

        # Get task instances
        task_instances = [
            task_factory.create_task(task_config)
            for task_config in self.tasks
        ]

        # Create crew
        crew = Crew(
            agents=agent_instances,
            tasks=task_instances,
            process=self.process,
            verbose=True
        )

        return crew
```

2. **Register crew template:**
```python
from app.crew_factory import CrewFactory

factory = CrewFactory()
factory.register_crew_template("custom_crew", CustomCrew)
```

## 🤝 Integration Examples

### With External APIs
```python
# API integration tool
class APIIntegrationTool(BaseTool):
    name: str = "api_integrator"
    description: str = "Integrate with external APIs"

    def __init__(self, api_config: Dict[str, Any]):
        super().__init__()
        self.api_config = api_config
        self.session = self._create_session()

    def _run(self, endpoint: str, method: str = "GET", data: Dict = None) -> str:
        """Execute API call."""
        try:
            url = f"{self.api_config['base_url']}{endpoint}"
            headers = self.api_config.get('headers', {})

            response = self.session.request(method, url, json=data, headers=headers)
            response.raise_for_status()

            return self._format_response(response.json())
        except Exception as e:
            return f"API call failed: {str(e)}"
```

### With Databases
```python
# Database integration tool
class DatabaseTool(BaseTool):
    name: str = "database_query"
    description: str = "Query and manipulate databases"

    def __init__(self, db_config: Dict[str, Any]):
        super().__init__()
        self.db_config = db_config
        self.connection = self._create_connection()

    def _run(self, query: str, params: List = None) -> str:
        """Execute database query."""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or [])

            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                return self._format_results(results)
            else:
                self.connection.commit()
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"

        except Exception as e:
            self.connection.rollback()
            return f"Database error: {str(e)}"
```

### With File Systems
```python
# File system tool
class FileSystemTool(BaseTool):
    name: str = "file_manager"
    description: str = "Manage files and directories"

    def __init__(self, fs_config: Dict[str, Any]):
        super().__init__()
        self.allowed_paths = fs_config.get('allowed_paths', [])
        self.max_file_size = fs_config.get('max_file_size', 10 * 1024 * 1024)  # 10MB

    def _run(self, operation: str, path: str, **kwargs) -> str:
        """Execute file system operation."""
        if not self._is_path_allowed(path):
            return f"Access denied: Path {path} is not allowed"

        try:
            if operation == "read":
                return self._read_file(path)
            elif operation == "write":
                return self._write_file(path, kwargs.get('content', ''))
            elif operation == "list":
                return self._list_directory(path)
            elif operation == "delete":
                return self._delete_file(path)
            else:
                return f"Unknown operation: {operation}"
        except Exception as e:
            return f"File operation failed: {str(e)}"
```

## 📈 Performance Optimization

### Optimization Strategies
- **Crew Caching**: Cache crew configurations and agent states
- **Task Parallelization**: Execute independent tasks concurrently
- **Result Memoization**: Cache tool results and agent responses
- **Resource Pooling**: Reuse expensive resources efficiently

### Scaling Considerations
- **Horizontal Scaling**: Multiple instances with load balancing
- **Crew Specialization**: Dedicated crews for different task types
- **Asynchronous Processing**: Queue-based task processing
- **Resource Optimization**: Dynamic resource allocation based on load

## 🔄 Advanced Features

### Human-in-the-Loop Workflows
- **Approval Gates**: Human approval required for critical decisions
- **Intervention Points**: Configurable points for human input
- **Feedback Integration**: Human feedback improves agent performance
- **Escalation Protocols**: Automatic escalation for complex issues

### Dynamic Crew Formation
- **Task-Based Assembly**: Crews formed based on task requirements
- **Skill Matching**: Agents selected based on required capabilities
- **Runtime Optimization**: Crew composition optimized for efficiency
- **Adaptive Scaling**: Crew size adjusted based on complexity

### Complex Workflow Patterns
- **Conditional Branching**: Different execution paths based on results
- **Loop Constructs**: Repeat tasks until conditions are met
- **Parallel Execution**: Multiple tasks running simultaneously
- **State Management**: Complex state tracking across workflow stages

## 🆘 Troubleshooting

### Common Issues
- **Crew Formation**: Check agent availability and configuration
- **Task Assignment**: Verify agent capabilities match task requirements
- **Tool Execution**: Check tool permissions and resource limits
- **Memory Issues**: Monitor conversation length and implement cleanup

### Debug Tools
```bash
# Enable verbose logging
export CREWAI_LOG_LEVEL=DEBUG

# Monitor crew execution
curl http://localhost:8000/crews/{crew_id}/debug

# Check agent status
curl http://localhost:8000/agents/status

# View workflow metrics
curl http://localhost:8000/metrics/crews
```

## 🤝 Contributing

1. **Fork and clone**: `git clone https://github.com/your-org/crewai-workflows.git`
2. **Create feature branch**: `git checkout -b feature/new-agent-type`
3. **Implement with tests**: Add comprehensive test coverage
4. **Update documentation**: Keep docs synchronized with code changes
5. **Submit PR**: Provide detailed description and usage examples

## 📄 License

Licensed under Apache 2.0 License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Complete API docs at `/docs`
- **GitHub Issues**: Bug reports and feature requests
- **Community Slack**: Real-time help and discussion
- **Professional Support**: Enterprise support options available

## 🔄 Changelog

### v1.0.0
- Production-ready CrewAI integration
- Multi-agent collaboration system
- Comprehensive workflow orchestration
- Tool integration framework
- Containerized deployment with scaling

### v0.9.0 (Beta)
- Core CrewAI functionality
- Agent creation and management
- Task orchestration system
- Tool execution framework
- API interface and monitoring

### v0.8.0 (Alpha)
- Initial CrewAI integration
- Basic agent communication
- Simple workflow support
- Tool registry system
- Proof of concept implementation
