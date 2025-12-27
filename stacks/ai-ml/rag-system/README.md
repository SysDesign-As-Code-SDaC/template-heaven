# RAG System Template

A complete Retrieval-Augmented Generation (RAG) system template with document ingestion, vector embeddings, similarity search, and LLM integration for building intelligent question-answering applications.

## 🚀 Features

- **Document Ingestion**: Support for PDF, DOCX, TXT, HTML, and markdown files
- **Text Chunking**: Intelligent document splitting with overlap control
- **Vector Embeddings**: Multiple embedding models (OpenAI, HuggingFace, SentenceTransformers)
- **Vector Storage**: Integration with vector databases (ChromaDB, Pinecone, Weaviate, Qdrant)
- **Similarity Search**: Advanced retrieval with re-ranking and filtering
- **LLM Integration**: Support for OpenAI GPT, Anthropic Claude, and local models
- **Conversational Memory**: Context-aware conversation management
- **Web Interface**: Streamlit-based UI for interaction
- **API Endpoints**: RESTful API for integration
- **Containerized**: Full Docker deployment support
- **Monitoring**: Built-in logging and performance metrics

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for development)
- 8GB+ RAM (for embedding models and LLM inference)
- API keys for embedding/LLM services (optional, can use local models)

## 🛠️ Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone and navigate to the template
git clone <repository-url> rag-system
cd rag-system

# Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# Start the RAG system
docker-compose up -d

# Check system health
curl http://localhost:8000/health

# Access web interface
open http://localhost:8501

# Ingest documents
curl -X POST http://localhost:8000/api/documents/ingest \
  -F "files=@document.pdf" \
  -F "files=@article.txt"

# Ask questions
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main benefits of RAG systems?",
    "session_id": "user123"
  }'
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Configure your settings

# Start vector database
docker-compose up chromadb -d

# Start the RAG system
python -m uvicorn rag_system.main:app --host 0.0.0.0 --port 8000 --reload

# Start web interface (in another terminal)
streamlit run src/rag_system/ui/app.py
```

## 📁 Project Structure

```
rag-system/
├── src/
│   ├── rag_system/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── core/
│   │   │   ├── document_processor.py # Document ingestion and processing
│   │   │   ├── embedding_service.py # Text embedding generation
│   │   │   ├── vector_store.py   # Vector database operations
│   │   │   ├── retrieval_engine.py # Similarity search and retrieval
│   │   │   ├── llm_service.py    # LLM integration and generation
│   │   │   ├── conversation_manager.py # Chat session management
│   │   │   └── rag_pipeline.py   # Main RAG orchestration
│   │   ├── ui/
│   │   │   ├── app.py           # Streamlit web interface
│   │   │   ├── components.py    # UI components
│   │   │   └── utils.py         # UI utilities
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── documents.py  # Document management endpoints
│   │   │   │   ├── chat.py       # Chat and Q&A endpoints
│   │   │   │   ├── search.py     # Search endpoints
│   │   │   │   └── admin.py      # Administrative endpoints
│   │   │   └── models.py         # Pydantic models
│   │   └── utils/
│   │       ├── config.py         # Configuration management
│   │       ├── logging.py        # Logging configuration
│   │       └── metrics.py        # Metrics collection
│   ├── tests/
│   │   ├── test_document_processing.py
│   │   ├── test_embedding_service.py
│   │   ├── test_retrieval_engine.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_api_integration.py
│   └── scripts/
│       ├── setup_database.py     # Vector database setup
│       ├── ingest_documents.py   # Bulk document ingestion
│       └── evaluate_rag.py       # RAG system evaluation
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   └── nginx.conf
├── docs/
│   ├── architecture.md           # System architecture
│   ├── api.md                    # API documentation
│   ├── deployment.md             # Deployment guide
│   ├── evaluation.md             # Performance evaluation
│   └── customization.md          # Customization guide
├── data/
│   ├── documents/                # Document storage
│   ├── embeddings/               # Embedding cache
│   └── logs/                     # Application logs
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🔧 Configuration

### Environment Variables

```bash
# Server Configuration
RAG_HOST=0.0.0.0
RAG_PORT=8000
RAG_WORKERS=4

# Vector Database
VECTOR_DB_TYPE=chromadb
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=8001
COLLECTION_NAME=rag_documents

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_PROVIDER=openai
ANTHROPIC_API_KEY=your-anthropic-key
LOCAL_LLM_MODEL=llama-2-7b-chat

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SUPPORTED_FORMATS=pdf,txt,md,docx,html

# Retrieval Configuration
TOP_K_RETRIEVAL=5
RERANKING_ENABLED=true
SEMANTIC_SIMILARITY_THRESHOLD=0.7

# Conversation Management
MAX_CONVERSATION_LENGTH=50
SESSION_TIMEOUT_HOURS=24

# Security
API_KEY=your-api-key
ENABLE_AUTH=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Logging and Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 📚 Key Components

### Document Processor
- **Multi-format Support**: PDF, DOCX, TXT, HTML, Markdown
- **Intelligent Chunking**: Sentence-aware splitting with configurable overlap
- **Metadata Extraction**: Automatic title, author, and content type detection
- **Preprocessing**: Text cleaning, normalization, and filtering

### Embedding Service
- **Multiple Providers**: OpenAI, HuggingFace, SentenceTransformers, Cohere
- **Batch Processing**: Efficient embedding generation for large document sets
- **Caching**: Local embedding cache to reduce API calls and costs
- **Dimensionality Control**: Configurable embedding dimensions

### Vector Store Integration
- **ChromaDB**: Fast, local vector database for development
- **Pinecone**: Cloud-hosted, scalable vector database
- **Weaviate**: Hybrid vector and keyword search
- **Qdrant**: High-performance, distributed vector database

### Retrieval Engine
- **Similarity Search**: Cosine, Euclidean, and dot product similarity
- **Re-ranking**: Advanced re-ranking algorithms for better results
- **Metadata Filtering**: Filter results by document type, date, author, etc.
- **Hybrid Search**: Combine semantic and keyword-based retrieval

### LLM Integration
- **OpenAI GPT**: GPT-3.5, GPT-4, and GPT-4-turbo
- **Anthropic Claude**: Claude 2 and Claude Instant
- **Local Models**: Llama 2, Mistral, and other open-source models
- **Prompt Engineering**: Optimized prompts for RAG applications

### Conversation Management
- **Session Handling**: Persistent conversation sessions
- **Context Window Management**: Efficient use of LLM context limits
- **Memory Compression**: Summarize old conversations to save space
- **Multi-turn Conversations**: Maintain context across multiple interactions

## 🧪 Testing and Evaluation

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/test_api_integration.py -v

# Evaluate RAG performance
python scripts/evaluate_rag.py \
  --questions questions.json \
  --documents documents/ \
  --output evaluation_results.json

# Test with specific LLM
LLM_PROVIDER=anthropic pytest tests/test_llm_integration.py -v

# Load testing
python scripts/load_test.py --concurrency 10 --duration 60
```

## 🚀 Deployment Options

### Development Deployment
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-system

# Scale services
docker-compose up -d --scale rag-system=3
```

### Production Deployment
```bash
# Build production images
docker build -f docker/Dockerfile.prod -t rag-system:prod .

# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d

# Set up monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl logs -f deployment/rag-system
```

## 📚 API Reference

### Document Management
```bash
# Ingest documents
POST /api/documents/ingest
Content-Type: multipart/form-data

# List documents
GET /api/documents

# Delete document
DELETE /api/documents/{doc_id}

# Get document metadata
GET /api/documents/{doc_id}/metadata
```

### Chat and Q&A
```bash
# Send message
POST /api/chat
{
  "message": "What is RAG?",
  "session_id": "user123",
  "stream": false
}

# Get conversation history
GET /api/chat/history/{session_id}

# Clear conversation
DELETE /api/chat/session/{session_id}
```

### Search and Retrieval
```bash
# Semantic search
POST /api/search
{
  "query": "machine learning",
  "limit": 10,
  "filters": {
    "document_type": "pdf",
    "date_range": ["2023-01-01", "2024-01-01"]
  }
}

# Get similar documents
POST /api/search/similar
{
  "document_id": "doc123",
  "limit": 5
}
```

## 🔒 Security and Privacy

### Data Protection
- **Encryption**: Data encryption at rest and in transit
- **Access Control**: Role-based access control for documents and conversations
- **API Security**: JWT authentication and rate limiting
- **Audit Logging**: Comprehensive logging of all operations

### Privacy Features
- **Data Retention**: Configurable data retention policies
- **Anonymization**: User data anonymization options
- **Consent Management**: User consent tracking for data usage
- **GDPR Compliance**: Built-in GDPR compliance features

## 📊 Monitoring and Analytics

### Metrics Collection
- **Query Performance**: Response times and throughput
- **Retrieval Accuracy**: Precision and recall metrics
- **LLM Usage**: Token consumption and cost tracking
- **User Engagement**: Conversation length and satisfaction

### Dashboard
```bash
# Access metrics dashboard
open http://localhost:3000/dashboards/rag-metrics

# API metrics endpoint
curl http://localhost:9090/metrics
```

## 🔧 Customization and Extension

### Adding New Document Types
```python
# src/rag_system/core/document_processor.py
class CustomDocumentProcessor:
    def process_custom_format(self, file_path: str) -> List[Document]:
        # Implement custom document processing logic
        pass
```

### Custom Embedding Models
```python
# src/rag_system/core/embedding_service.py
class CustomEmbeddingService:
    def generate_embeddings_custom(self, texts: List[str]) -> List[List[float]]:
        # Implement custom embedding generation
        pass
```

### Extending LLM Support
```python
# src/rag_system/core/llm_service.py
class CustomLLMService:
    def generate_response_custom(self, prompt: str, context: str) -> str:
        # Implement custom LLM integration
        pass
```

## 📈 Performance Optimization

### Configuration Tuning
```python
# High-performance settings
CHUNK_SIZE=512
TOP_K_RETRIEVAL=3
EMBEDDING_BATCH_SIZE=32
LLM_MAX_TOKENS=1024
VECTOR_CACHE_SIZE=10000
```

### Hardware Acceleration
```bash
# GPU support for embeddings
docker run --gpus all rag-system:latest

# CPU optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 🤝 Integration Examples

### Web Application Integration
```javascript
// frontend/src/services/ragService.js
class RAGService {
  async askQuestion(question, sessionId) {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question, session_id: sessionId })
    });
    return response.json();
  }
}
```

### API Integration
```python
# client/rag_client.py
import requests

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def chat(self, message, session_id=None):
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"message": message, "session_id": session_id}
        )
        return response.json()

    def ingest_document(self, file_path):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/documents/ingest",
                files={"files": f}
            )
        return response.json()
```

## 📋 Troubleshooting

### Common Issues

**High Latency**
- Check vector database performance
- Optimize embedding batch size
- Consider using GPU acceleration

**Low Retrieval Quality**
- Adjust chunk size and overlap
- Experiment with different embedding models
- Fine-tune similarity thresholds

**Memory Issues**
- Reduce batch sizes
- Implement document pagination
- Use streaming for large responses

**LLM API Limits**
- Implement request queuing
- Add retry logic with backoff
- Monitor API usage and costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This template is part of the Template Heaven project.

## 🔗 Related Templates

- [Vector Database](../vector-database/) - Vector storage backend
- [MCP Middleware](../mcp-middleware/) - AI assistant integration
- [FastAPI Microservice](../fastapi-microservice/) - API foundation

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Join our Discord community
