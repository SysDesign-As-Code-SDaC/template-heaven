# LLM RAG Application Template

A production-ready Retrieval-Augmented Generation (RAG) application template using modern LLM frameworks, vector databases, and AI orchestration tools for 2025.

## 🚀 Features

- **LangChain** - LLM application framework
- **LlamaIndex** - Data framework for LLM applications
- **OpenAI GPT-4** - Large language model integration
- **ChromaDB** - Vector database for embeddings
- **Pinecone** - Managed vector database (optional)
- **Weaviate** - Vector database with hybrid search
- **FastAPI** - High-performance API framework
- **Streamlit** - Interactive web interface
- **Docker** - Containerized deployment
- **Redis** - Caching and session management
- **PostgreSQL** - Document metadata storage
- **Celery** - Asynchronous task processing
- **Monitoring** - LLM performance tracking

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Docker & Docker Compose
- OpenAI API key
- Vector database access (ChromaDB/Pinecone/Weaviate)

## 🛠️ Quick Start

### 1. Create New RAG Application

```bash
git clone <this-repo> my-rag-app
cd my-rag-app
```

### 2. Environment Setup

```bash
cp .env.example .env
```

Configure your environment variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/rag_db
REDIS_URL=redis://localhost:6379
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Application Configuration
APP_NAME=RAG Application
APP_VERSION=1.0.0
DEBUG=True
LOG_LEVEL=INFO

# Vector Database Configuration
VECTOR_DB_TYPE=chroma  # chroma, pinecone, weaviate
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# LLM Configuration
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for frontend)
cd frontend
npm install
cd ..
```

### 4. Start Services

```bash
# Start with Docker Compose
docker-compose up -d

# Or start manually
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access Application

- **API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
├── app/                        # FastAPI application
│   ├── api/                   # API routes
│   │   ├── chat.py           # Chat endpoints
│   │   ├── documents.py      # Document management
│   │   └── search.py         # Search endpoints
│   ├── core/                 # Core functionality
│   │   ├── config.py         # Configuration
│   │   ├── security.py       # Authentication
│   │   └── database.py       # Database connection
│   ├── models/               # Data models
│   │   ├── document.py       # Document models
│   │   ├── chat.py           # Chat models
│   │   └── user.py           # User models
│   ├── services/             # Business logic
│   │   ├── llm_service.py    # LLM integration
│   │   ├── vector_service.py # Vector database operations
│   │   ├── rag_service.py    # RAG pipeline
│   │   └── document_service.py # Document processing
│   ├── utils/                # Utility functions
│   │   ├── embeddings.py     # Embedding utilities
│   │   ├── text_processing.py # Text processing
│   │   └── monitoring.py     # Monitoring utilities
│   └── main.py               # Application entry point
├── frontend/                 # Streamlit/React frontend
│   ├── components/           # React components
│   ├── pages/               # Application pages
│   └── utils/               # Frontend utilities
├── data/                     # Data storage
│   ├── documents/           # Uploaded documents
│   ├── embeddings/          # Generated embeddings
│   └── cache/               # Cached data
├── tests/                   # Test files
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── scripts/                 # Utility scripts
│   ├── ingest_documents.py  # Document ingestion
│   ├── generate_embeddings.py # Embedding generation
│   └── evaluate_rag.py      # RAG evaluation
└── docker/                  # Docker configurations
    ├── Dockerfile
    └── docker-compose.yml
```

## 🔧 Available Scripts

```bash
# Development
python -m uvicorn app.main:app --reload    # Start API server
streamlit run frontend/app.py              # Start frontend
celery -A app.tasks worker --loglevel=info # Start task worker

# Data Processing
python scripts/ingest_documents.py         # Ingest documents
python scripts/generate_embeddings.py      # Generate embeddings
python scripts/evaluate_rag.py             # Evaluate RAG performance

# Testing
pytest tests/                              # Run all tests
pytest tests/unit/                         # Run unit tests
pytest tests/integration/                  # Run integration tests

# Docker
docker-compose up -d                       # Start all services
docker-compose down                        # Stop all services
docker-compose logs -f                     # View logs
```

## 🤖 RAG Pipeline Implementation

### Document Processing Service

```python
# app/services/document_service.py
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from app.services.vector_service import VectorService
from app.models.document import Document

class DocumentService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings()
    
    async def process_document(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Process and ingest a document into the vector database."""
        
        # Load document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Generate embeddings and store in vector database
        document_id = await self.vector_service.store_documents(
            chunks, 
            metadata=metadata
        )
        
        return document_id
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for relevant documents using vector similarity."""
        
        # Generate query embedding
        query_embedding = await self.embeddings.aembed_query(query)
        
        # Search vector database
        results = await self.vector_service.search(
            query_embedding, 
            top_k=top_k
        )
        
        return results
```

### RAG Service

```python
# app/services/rag_service.py
from typing import List, Dict, Any
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.services.document_service import DocumentService
from app.models.chat import ChatMessage, ChatResponse

class RAGService:
    def __init__(self, document_service: DocumentService):
        self.document_service = document_service
        self.llm = OpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=4000
        )
        
        self.prompt_template = PromptTemplate(
            template="""
            You are a helpful AI assistant. Use the following context to answer the question.
            If you don't know the answer based on the context, say so.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """,
            input_variables=["context", "question"]
        )
    
    async def generate_response(
        self, 
        question: str, 
        conversation_history: List[ChatMessage] = None
    ) -> ChatResponse:
        """Generate a response using RAG pipeline."""
        
        # Search for relevant documents
        relevant_docs = await self.document_service.search_documents(question)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.content for doc in relevant_docs])
        
        # Generate response using LLM
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        response = await self.llm.agenerate([prompt])
        
        return ChatResponse(
            answer=response.generations[0][0].text,
            sources=[doc.metadata for doc in relevant_docs],
            confidence=self._calculate_confidence(response, relevant_docs)
        )
    
    def _calculate_confidence(self, response, relevant_docs) -> float:
        """Calculate confidence score for the response."""
        # Simple confidence calculation based on source relevance
        if not relevant_docs:
            return 0.0
        
        # Calculate average similarity score
        avg_similarity = sum(doc.similarity_score for doc in relevant_docs) / len(relevant_docs)
        return min(avg_similarity, 1.0)
```

### Vector Database Service

```python
# app/services/vector_service.py
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from app.models.document import Document

class VectorService:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def store_documents(
        self, 
        documents: List[Any], 
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store documents in the vector database."""
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}_{hash(text)}" for i, text in enumerate(texts)]
        
        # Generate embeddings
        embeddings = await self._generate_embeddings(texts)
        
        # Store in ChromaDB
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return f"stored_{len(documents)}_documents"
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """Search for similar documents."""
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        documents = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            documents.append(Document(
                content=doc,
                metadata=metadata,
                similarity_score=1 - distance  # Convert distance to similarity
            ))
        
        return documents
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        # This would typically use OpenAI embeddings or similar
        # For now, returning placeholder
        return [[0.0] * 1536 for _ in texts]  # OpenAI ada-002 dimension
```

## 🎨 Frontend Interface

### Streamlit Chat Interface

```python
# frontend/app.py
import streamlit as st
import requests
from typing import List

st.set_page_config(
    page_title="RAG Chat Application",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chat Application")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"**{source.get('title', 'Unknown')}**")
                    st.write(f"Page: {source.get('page', 'N/A')}")
                    st.write(f"Similarity: {source.get('similarity', 'N/A'):.2f}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://localhost:8000/api/chat",
                json={
                    "message": prompt,
                    "conversation_history": st.session_state.messages[-5:]  # Last 5 messages
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                st.markdown(data["answer"])
                
                # Display sources
                if data.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in data["sources"]:
                            st.write(f"**{source.get('title', 'Unknown')}**")
                            st.write(f"Page: {source.get('page', 'N/A')}")
                            st.write(f"Similarity: {source.get('similarity', 'N/A'):.2f}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": data["answer"],
                    "sources": data.get("sources", [])
                })
            else:
                st.error("Failed to get response from the API")
```

## 📊 Monitoring and Evaluation

### RAG Performance Metrics

```python
# app/utils/evaluation.py
from typing import List, Dict, Any
import asyncio
from app.services.rag_service import RAGService

class RAGEvaluator:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    async def evaluate_rag_pipeline(
        self, 
        test_questions: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """Evaluate RAG pipeline performance."""
        
        metrics = {
            "accuracy": 0.0,
            "relevance": 0.0,
            "coherence": 0.0,
            "response_time": 0.0
        }
        
        total_questions = len(test_questions)
        total_response_time = 0.0
        
        for question_data in test_questions:
            question = question_data["question"]
            expected_answer = question_data["expected_answer"]
            
            # Measure response time
            start_time = asyncio.get_event_loop().time()
            response = await self.rag_service.generate_response(question)
            end_time = asyncio.get_event_loop().time()
            
            response_time = end_time - start_time
            total_response_time += response_time
            
            # Calculate accuracy (simplified)
            accuracy = self._calculate_accuracy(response.answer, expected_answer)
            metrics["accuracy"] += accuracy
            
            # Calculate relevance (based on source similarity)
            relevance = sum(source.get("similarity", 0) for source in response.sources) / len(response.sources) if response.sources else 0
            metrics["relevance"] += relevance
        
        # Average metrics
        metrics["accuracy"] /= total_questions
        metrics["relevance"] /= total_questions
        metrics["response_time"] = total_response_time / total_questions
        
        return metrics
    
    def _calculate_accuracy(self, generated: str, expected: str) -> float:
        """Calculate accuracy between generated and expected answers."""
        # Simple word overlap calculation
        generated_words = set(generated.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(generated_words.intersection(expected_words))
        return overlap / len(expected_words)
```

## 🚀 Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/documents data/embeddings data/cache

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/rag_db
      - REDIS_URL=redis://redis:6379
      - CHROMA_HOST=chroma
      - CHROMA_PORT=8000
    depends_on:
      - db
      - redis
      - chroma
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - app

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=rag_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  worker:
    build: .
    command: celery -A app.tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/rag_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
  chroma_data:
```

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🔗 Upstream Source

- **Repository**: [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- **LlamaIndex**: [jerryjliu/llama_index](https://github.com/jerryjliu/llama_index)
- **ChromaDB**: [chroma-core/chroma](https://github.com/chroma-core/chroma)
- **License**: MIT
