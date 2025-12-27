# Vector Database Service

A containerized vector database service with REST API, supporting multiple vector storage backends (ChromaDB, Pinecone, Weaviate, Qdrant) for AI applications requiring semantic search and similarity matching.

## 🚀 Features

- **Multiple Backend Support**: ChromaDB, Pinecone, Weaviate, Qdrant, Milvus
- **RESTful API**: Complete CRUD operations for vectors and metadata
- **Batch Operations**: Efficient bulk vector insertion and querying
- **Similarity Search**: Cosine, Euclidean, and dot product similarity
- **Metadata Filtering**: Advanced filtering and search capabilities
- **Containerized Deployment**: Docker and Docker Compose support
- **Monitoring**: Built-in metrics and health checks
- **Persistence**: Configurable data persistence options

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for development)
- 4GB+ RAM (for in-memory vector operations)

## 🛠️ Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone and navigate to the template
git clone <repository-url> vector-database
cd vector-database

# Start the vector database service
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# Create a collection
curl -X POST http://localhost:8000/api/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "documents",
    "backend": "chromadb",
    "dimension": 1536,
    "metric": "cosine"
  }'

# Add vectors
curl -X POST http://localhost:8000/api/collections/documents/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...]],
    "metadata": [{"source": "doc1"}, {"source": "doc2"}],
    "ids": ["vec1", "vec2"]
  }'

# Search similar vectors
curl -X POST http://localhost:8000/api/collections/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [0.1, 0.2, 0.3, ...],
    "limit": 10,
    "include_metadata": true
  }'
```

## 📁 Project Structure

```
vector-database/
├── src/
│   ├── vector_database/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── core/
│   │   │   ├── vector_store.py  # Vector storage abstraction
│   │   │   ├── search_engine.py # Similarity search engine
│   │   │   └── collection_manager.py # Collection management
│   │   ├── backends/
│   │   │   ├── chromadb_backend.py # ChromaDB implementation
│   │   │   ├── pinecone_backend.py # Pinecone implementation
│   │   │   ├── weaviate_backend.py # Weaviate implementation
│   │   │   └── qdrant_backend.py   # Qdrant implementation
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── collections.py # Collection endpoints
│   │   │   │   ├── vectors.py     # Vector CRUD endpoints
│   │   │   │   └── search.py      # Search endpoints
│   │   │   └── models.py          # Pydantic models
│   │   └── utils/
│   │       ├── config.py          # Configuration management
│   │       ├── validation.py      # Input validation
│   │       └── metrics.py         # Metrics collection
│   ├── tests/
│   │   ├── test_vector_operations.py
│   │   ├── test_search_functionality.py
│   │   └── test_api_integration.py
│   └── scripts/
│       ├── init_database.py       # Database initialization
│       └── migrate_data.py        # Data migration utilities
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   └── nginx.conf
├── docs/
│   ├── api.md                     # API documentation
│   ├── backends.md                # Backend-specific guides
│   ├── deployment.md              # Deployment guide
│   └── performance.md             # Performance tuning
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

### Environment Variables

```bash
# Server Configuration
VECTOR_HOST=0.0.0.0
VECTOR_PORT=8000
VECTOR_WORKERS=4

# Backend Selection
VECTOR_BACKEND=chromadb  # chromadb, pinecone, weaviate, qdrant, milvus

# ChromaDB Configuration (when using chromadb backend)
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_PERSIST_DIR=./chroma_data

# Pinecone Configuration (when using pinecone backend)
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=your-environment
PINECONE_INDEX_NAME=vectors

# Weaviate Configuration (when using weaviate backend)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-api-key

# Security
VECTOR_API_KEY=your-api-key
VECTOR_CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Performance
VECTOR_MAX_BATCH_SIZE=1000
VECTOR_SIMILARITY_TIMEOUT=30
VECTOR_CACHE_TTL=3600
```

### Vector Collection Configuration

```json
{
  "name": "my-collection",
  "backend": "chromadb",
  "dimension": 1536,
  "metric": "cosine",
  "metadata": {
    "description": "Document embeddings collection",
    "created_by": "user123"
  },
  "config": {
    "persist": true,
    "replicas": 1,
    "shards": 1
  }
}
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/test_api_integration.py -v

# Run performance tests
pytest tests/test_performance.py -v

# Test with specific backend
VECTOR_BACKEND=pinecone pytest tests/test_vector_operations.py -v

# Test with Docker
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 🚀 Deployment

### Production Deployment

```bash
# Build production image
docker build -f docker/Dockerfile.prod -t vector-database:prod .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/

# Scale the service
docker-compose up -d --scale vector-database=3

# Check deployment health
curl https://your-domain.com/health
```

### Backend-Specific Deployment

#### ChromaDB (Self-hosted)
```bash
# Start ChromaDB
docker run -d \
  -p 8001:8000 \
  -v ./chroma_data:/chroma/chroma \
  chromadb/chroma:latest

# Start vector service with ChromaDB backend
VECTOR_BACKEND=chromadb docker-compose up -d
```

#### Pinecone (Cloud-hosted)
```bash
# Set Pinecone credentials
export PINECONE_API_KEY=your-api-key
export PINECONE_ENVIRONMENT=your-environment

# Start with Pinecone backend
VECTOR_BACKEND=pinecone docker-compose up -d
```

## 📚 API Reference

### Core Endpoints

- `GET /health` - Service health check
- `GET /metrics` - Prometheus metrics
- `GET /api/collections` - List collections
- `POST /api/collections` - Create collection
- `DELETE /api/collections/{name}` - Delete collection

### Vector Operations

- `POST /api/collections/{name}/vectors` - Add vectors
- `GET /api/collections/{name}/vectors/{id}` - Get vector by ID
- `PUT /api/collections/{name}/vectors/{id}` - Update vector
- `DELETE /api/collections/{name}/vectors/{id}` - Delete vector
- `POST /api/collections/{name}/vectors/batch` - Batch operations

### Search Operations

- `POST /api/collections/{name}/search` - Similarity search
- `POST /api/collections/{name}/search/filter` - Filtered search
- `POST /api/collections/{name}/search/hybrid` - Hybrid search

### Examples

#### Creating a Collection
```bash
curl -X POST http://localhost:8000/api/collections \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "name": "embeddings",
    "backend": "chromadb",
    "dimension": 1536,
    "metric": "cosine",
    "description": "Text embeddings for semantic search"
  }'
```

#### Adding Vectors
```bash
curl -X POST http://localhost:8000/api/collections/embeddings/vectors \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "vectors": [
      [0.1, 0.2, 0.3, ..., 0.1536],
      [0.4, 0.5, 0.6, ..., 0.1537]
    ],
    "metadata": [
      {"text": "Hello world", "source": "doc1"},
      {"text": "Goodbye world", "source": "doc2"}
    ],
    "ids": ["vec_1", "vec_2"]
  }'
```

#### Similarity Search
```bash
curl -X POST http://localhost:8000/api/collections/embeddings/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query_vector": [0.1, 0.2, 0.3, ..., 0.1536],
    "limit": 5,
    "include_metadata": true,
    "include_distances": true
  }'
```

## 🔒 Security

### Authentication
- API key authentication for all endpoints
- Configurable API key rotation
- Request rate limiting and throttling

### Data Protection
- Encryption at rest for persistent backends
- Secure communication (HTTPS/TLS)
- Input validation and sanitization
- Audit logging for sensitive operations

## 📊 Monitoring

### Metrics
- Request latency and throughput
- Vector operation performance
- Search accuracy and recall
- Backend-specific metrics

### Health Checks
- Service availability monitoring
- Backend connectivity checks
- Resource utilization tracking
- Automated alerting and notifications

## 🔧 Backend Comparison

| Backend | Self-hosted | Cloud | Performance | Features |
|---------|-------------|-------|-------------|----------|
| ChromaDB | ✅ | ❌ | Good | Simple, fast local development |
| Pinecone | ❌ | ✅ | Excellent | Production-ready, scalable |
| Weaviate | ✅ | ❌ | Good | Rich querying, hybrid search |
| Qdrant | ✅ | ❌ | Excellent | High performance, distributed |
| Milvus | ✅ | ❌ | Excellent | Enterprise-grade, scalable |

## 📈 Performance Tuning

### Configuration Tuning

```python
# High-performance configuration
VECTOR_MAX_BATCH_SIZE=5000
VECTOR_SIMILARITY_TIMEOUT=60
VECTOR_CACHE_TTL=7200
VECTOR_WORKERS=8

# Backend-specific optimizations
CHROMA_HNSW_M=32
PINECONE_TOP_K=100
QDRANT_OPTIMIZERS_CONFIG='{"memmap_threshold": 1000000}'
```

### Scaling Strategies

```bash
# Horizontal scaling
docker-compose up -d --scale vector-database=5

# Load balancing with nginx
docker-compose -f docker-compose.lb.yml up -d

# Database optimization
docker-compose -f docker-compose.optimized.yml up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This template is part of the Template Heaven project.

## 🔗 Related Templates

- [MCP Middleware](../mcp-middleware/) - Connect vector databases to AI assistants
- [RAG System](../rag-system/) - Retrieval-augmented generation using vector search
- [FastAPI Microservice](../fastapi-microservice/) - API service foundation

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Review the troubleshooting guide
