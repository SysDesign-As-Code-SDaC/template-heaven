# LangChain Integration Template

A comprehensive containerized application that integrates LangChain and LlamaIndex frameworks with AI assistants, providing advanced tool integration, agent capabilities, and RAG (Retrieval-Augmented Generation) workflows.

## 🌟 Features

- **LangChain Integration**: Full LangChain framework integration with custom tools
- **LlamaIndex Support**: Advanced RAG capabilities with multiple index types
- **Multi-Agent Systems**: Support for multiple AI agents with specialized roles
- **Tool Integration**: Extensive tool library with custom tool creation
- **Vector Stores**: Multiple vector database backends (Chroma, Pinecone, Weaviate)
- **Document Processing**: Advanced document ingestion and processing pipelines
- **Streaming Responses**: Real-time response streaming for enhanced UX
- **Memory Management**: Conversation memory with multiple persistence options
- **Chain Orchestration**: Complex chain and workflow management

## 📋 Prerequisites

- Python 3.9+
- OpenAI or Anthropic API key (for LLM integration)
- Docker and Docker Compose (for containerized deployment)
- Redis (optional, for caching and memory)
- PostgreSQL (optional, for advanced memory persistence)

## 🚀 Quick Start

1. **Clone and setup:**
```bash
git checkout stack/ai-ml
cp -r langchain-integration my-project
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
curl -X POST "http://localhost:8000/agents/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "research_assistant",
    "message": "Research the latest developments in quantum computing",
    "tools": ["web_search", "document_reader"]
  }'
```

## 🏗️ Project Structure

```
langchain-integration/
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration management
│   │   └── middleware/
│   │       ├── cors.py
│   │       ├── rate_limit.py
│   │       └── logging.py
│   ├── core/
│   │   ├── langchain_manager.py # LangChain integration
│   │   ├── llama_index_manager.py # LlamaIndex integration
│   │   ├── agent_orchestrator.py # Multi-agent management
│   │   ├── tool_registry.py     # Tool management
│   │   └── memory_manager.py    # Memory persistence
│   ├── agents/                  # Pre-built agents
│   │   ├── research_agent.py
│   │   ├── coding_agent.py
│   │   ├── analysis_agent.py
│   │   └── custom/              # Custom agents
│   ├── tools/                   # Tool implementations
│   │   ├── web_tools.py
│   │   ├── file_tools.py
│   │   ├── api_tools.py
│   │   ├── search_tools.py
│   │   └── custom/              # Custom tools
│   ├── chains/                  # LangChain chains
│   │   ├── rag_chain.py
│   │   ├── analysis_chain.py
│   │   ├── generation_chain.py
│   │   └── custom/              # Custom chains
│   ├── indexes/                 # LlamaIndex indexes
│   │   ├── vector_index.py
│   │   ├── document_index.py
│   │   ├── knowledge_base.py
│   │   └── custom/              # Custom indexes
│   └── utils/
│       ├── cache.py
│       ├── metrics.py
│       └── validation.py
├── data/
│   ├── documents/               # Document storage
│   ├── indexes/                 # Index persistence
│   └── cache/                   # Cache storage
├── tests/
│   ├── test_agents.py
│   ├── test_tools.py
│   ├── test_chains.py
│   └── test_integration.py
├── docs/
│   ├── api.md
│   ├── agents.md
│   ├── tools.md
│   └── deployment.md
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose.gpu.yml
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

# Vector Database Configuration
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment

# Memory Configuration
MEMORY_TYPE=redis
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://user:pass@localhost/db

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=false

# Document Processing
DOCUMENT_CHUNK_SIZE=1000
DOCUMENT_CHUNK_OVERLAP=200

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

### Agent Configuration

Agents are defined in YAML format:

```yaml
research_agent:
  name: "Research Assistant"
  role: "Research and information gathering specialist"
  goal: "Gather comprehensive information on given topics"
  backstory: "Expert researcher with access to web search and document analysis tools"
  llm: "gpt-4"
  tools:
    - web_search
    - document_reader
    - summarizer
  memory:
    type: "conversation_buffer_window"
    k: 10
  temperature: 0.7
  max_tokens: 2000
```

## 🔧 API Reference

### POST /agents/chat

Main agent interaction endpoint.

**Request:**
```json
{
  "agent_id": "research_agent",
  "message": "Analyze the current state of AI development",
  "tools": ["web_search", "document_reader"],
  "stream": true,
  "context": {
    "conversation_id": "conv_123",
    "additional_context": "Focus on 2024 developments"
  }
}
```

**Response:**
```json
{
  "agent_id": "research_agent",
  "response": "Based on my research, AI development in 2024 shows significant progress...",
  "tool_calls": [
    {
      "tool": "web_search",
      "input": {"query": "AI developments 2024"},
      "output": {"results": [...]}
    }
  ],
  "conversation_id": "conv_123",
  "usage": {
    "tokens": 1500,
    "cost": 0.03
  }
}
```

### POST /chains/execute

Execute a LangChain workflow.

```json
{
  "chain_id": "rag_chain",
  "input": {
    "query": "What are the benefits of microservices?",
    "documents": ["doc1", "doc2"]
  },
  "config": {
    "temperature": 0.5,
    "max_tokens": 1000
  }
}
```

### POST /indexes/query

Query LlamaIndex knowledge base.

```json
{
  "index_id": "knowledge_base",
  "query": "Explain quantum computing",
  "similarity_top_k": 5,
  "filters": {
    "subject": "physics"
  }
}
```

### POST /documents/ingest

Ingest documents for RAG.

```json
{
  "documents": [
    {
      "content": "Document content here...",
      "metadata": {
        "title": "Document Title",
        "author": "Author Name",
        "category": "Technology"
      }
    }
  ],
  "index_id": "knowledge_base",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

## 🤖 Built-in Agents

### Research Agent
- **Purpose**: Information gathering and research
- **Tools**: Web search, document analysis, summarization
- **Capabilities**: Multi-source research, fact-checking, report generation

### Coding Agent
- **Purpose**: Software development assistance
- **Tools**: Code execution, file operations, documentation
- **Capabilities**: Code generation, debugging, testing, documentation

### Analysis Agent
- **Purpose**: Data analysis and insights
- **Tools**: Statistical analysis, visualization, reporting
- **Capabilities**: Data processing, trend analysis, recommendation generation

## 🛠️ Tool Library

### Web Tools
- **WebSearchTool**: Search the web with multiple engines
- **WebScraperTool**: Extract content from web pages
- **APITool**: Make HTTP requests to REST APIs
- **GraphQLTool**: Query GraphQL endpoints

### File Tools
- **FileReaderTool**: Read various file formats
- **FileWriterTool**: Write and modify files
- **DirectoryTool**: Navigate and manage directories
- **SearchTool**: Search within files and directories

### AI/ML Tools
- **EmbeddingTool**: Generate text embeddings
- **SimilarityTool**: Calculate text similarity
- **ClassificationTool**: Text classification and categorization
- **SummarizationTool**: Generate text summaries

### Custom Tools
- **DatabaseTool**: Query databases with natural language
- **EmailTool**: Send and manage emails
- **CalendarTool**: Schedule and manage events
- **NotificationTool**: Send notifications via various channels

## 📚 LangChain Integration

### Chain Types
- **LLMChain**: Basic language model chains
- **StuffDocumentsChain**: Document processing chains
- **ReduceDocumentsChain**: Large document processing
- **MapReduceChain**: Parallel document processing
- **ConversationalRetrievalChain**: RAG with conversation memory

### Custom Chains
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Custom analysis chain
analysis_prompt = PromptTemplate(
    input_variables=["topic", "data"],
    template="Analyze the following data about {topic}: {data}"
)

analysis_chain = LLMChain(
    llm=llm,
    prompt=analysis_prompt,
    output_key="analysis"
)
```

## 🔍 LlamaIndex Integration

### Index Types
- **VectorStoreIndex**: Standard vector similarity search
- **SummaryIndex**: Document summarization
- **TreeIndex**: Hierarchical document organization
- **KeywordTableIndex**: Keyword-based retrieval
- **SQLIndex**: SQL database integration

### Query Engines
- **RetrieverQueryEngine**: Basic retrieval
- **TransformQueryEngine**: Query transformation
- **MultiStepQueryEngine**: Multi-step reasoning
- **SubQuestionQueryEngine**: Complex question decomposition

## 🚀 Deployment

### Docker Deployment
```bash
# Single container
docker run -p 8000:8000 langchain-integration:latest

# Full stack with dependencies
docker-compose up -d

# GPU-enabled deployment
docker-compose -f docker-compose.gpu.yml up -d
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-integration
spec:
  template:
    spec:
      containers:
      - name: app
        image: langchain-integration:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Cloud Deployment
- **AWS**: SageMaker, ECS, or EKS
- **GCP**: Vertex AI, Cloud Run
- **Azure**: Machine Learning, Container Apps

## 🧪 Testing

### Test Structure
```bash
# Unit tests
pytest tests/unit/ -v --cov=src

# Integration tests
pytest tests/integration/ -v

# Agent tests
pytest tests/agents/ -v

# Performance tests
pytest tests/performance/ -v
```

### Test Categories
- **Agent Tests**: Individual agent functionality
- **Tool Tests**: Tool execution and error handling
- **Chain Tests**: LangChain workflow validation
- **Index Tests**: LlamaIndex query and retrieval
- **Integration Tests**: End-to-end workflows

## 📊 Monitoring

### Metrics Collection
- **Agent Performance**: Response times, success rates
- **Tool Usage**: Call counts, error rates, execution times
- **Resource Usage**: Memory, CPU, API costs
- **Index Performance**: Query times, hit rates

### Health Checks
- `/health` - Application health
- `/metrics` - Prometheus metrics
- `/agents/health` - Agent-specific health
- `/indexes/health` - Index health status

## 🔒 Security

### Tool Sandboxing
- **Isolated Execution**: Tools run in separate processes
- **Resource Limits**: CPU, memory, and network restrictions
- **Input Validation**: Strict parameter validation and sanitization

### Data Protection
- **Encryption**: Sensitive data encryption at rest and in transit
- **Access Control**: Role-based access to agents and tools
- **Audit Logging**: Complete operation logging for compliance

## 🔧 Customization

### Creating Custom Agents

1. **Define agent configuration:**
```python
from app.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.specialized_tools = config.get("specialized_tools", [])

    def process_message(self, message, context=None):
        # Custom processing logic
        return self.llm.generate_response(message, self.tools)
```

2. **Register the agent:**
```python
from app.agent_orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
orchestrator.register_agent("custom_agent", CustomAgent, config)
```

### Creating Custom Tools

1. **Implement the tool:**
```python
from langchain.tools import BaseTool
from typing import Optional

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "A custom tool for specific functionality"

    def _run(self, query: str) -> str:
        # Tool implementation
        return f"Processed: {query}"

    async def _arun(self, query: str) -> str:
        # Async implementation
        return await self._run(query)
```

2. **Register the tool:**
```python
from app.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register_tool(CustomTool())
```

### Custom Chains

1. **Create chain configuration:**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Custom chain for specific workflow
custom_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["input"],
        template="Process this input: {input}"
    )
)
```

2. **Integrate with application:**
```python
from app.chains.chain_manager import ChainManager

chain_manager = ChainManager()
chain_manager.register_chain("custom_chain", custom_chain)
```

## 🤝 Integration Examples

### With External APIs
```python
# Custom API tool
class ExternalAPITool(BaseTool):
    name = "external_api"
    description = "Call external API endpoints"

    def _run(self, endpoint: str, params: dict) -> str:
        response = requests.get(f"https://api.example.com/{endpoint}", params=params)
        return response.json()
```

### With Databases
```python
# Database query tool
class DatabaseTool(BaseTool):
    name = "database_query"
    description = "Query database with natural language"

    def _run(self, query: str) -> str:
        # Convert natural language to SQL
        sql = self.nl_to_sql(query)
        results = self.execute_sql(sql)
        return json.dumps(results)
```

### With File Systems
```python
# File processing tool
class FileProcessingTool(BaseTool):
    name = "file_processor"
    description = "Process and analyze files"

    def _run(self, file_path: str, operation: str) -> str:
        if operation == "read":
            with open(file_path, 'r') as f:
                return f.read()
        elif operation == "analyze":
            return self.analyze_file(file_path)
```

## 📈 Performance Optimization

### Caching Strategies
- **Response Caching**: Cache agent responses and tool results
- **Embedding Caching**: Cache text embeddings to reduce API calls
- **Index Caching**: Cache frequently accessed index results

### Optimization Techniques
- **Batch Processing**: Process multiple requests together
- **Async Execution**: Non-blocking tool and agent execution
- **Connection Pooling**: Reuse database and API connections
- **Memory Management**: Efficient memory usage for large indexes

## 🔄 Advanced Features

### Multi-Agent Collaboration
- **Agent Communication**: Agents can call other agents
- **Task Decomposition**: Break complex tasks into subtasks
- **Result Aggregation**: Combine results from multiple agents

### Dynamic Tool Loading
- **Plugin Architecture**: Load tools at runtime
- **Tool Discovery**: Automatically discover available tools
- **Version Management**: Handle tool version compatibility

### Workflow Orchestration
- **Complex Chains**: Multi-step processing pipelines
- **Conditional Logic**: Execute different paths based on results
- **Error Handling**: Robust error recovery and retry logic

## 🆘 Troubleshooting

### Common Issues
- **Memory Issues**: Increase memory limits or optimize indexes
- **API Rate Limits**: Implement request queuing and backoff
- **Tool Timeouts**: Adjust timeout settings or optimize tool code
- **Index Performance**: Rebuild indexes or adjust chunk sizes

### Debug Tools
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Profile agent performance
python -m cProfile -s time app/main.py

# Monitor resource usage
docker stats

# Check index health
curl http://localhost:8000/indexes/health
```

## 🤝 Contributing

1. **Fork and clone**: `git clone https://github.com/your-org/langchain-integration.git`
2. **Create feature branch**: `git checkout -b feature/new-agent`
3. **Add tests**: Ensure comprehensive test coverage
4. **Update docs**: Keep documentation current
5. **Submit PR**: Create pull request with detailed description

## 📄 License

Licensed under Apache 2.0 License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Complete docs at `/docs`
- **GitHub Issues**: Bug reports and feature requests
- **Community Discord**: Real-time help and discussion
- **Professional Support**: Enterprise support available

## 🔄 Changelog

### v1.0.0
- Production-ready LangChain and LlamaIndex integration
- Multi-agent orchestration system
- Comprehensive tool library
- Advanced RAG capabilities
- Containerized deployment

### v0.9.0 (Beta)
- Core LangChain integration
- Basic agent system
- Tool registry and execution
- LlamaIndex support
- API interface

### v0.8.0 (Alpha)
- Initial framework integration
- Basic tool support
- Simple agent implementation
- Proof of concept RAG
