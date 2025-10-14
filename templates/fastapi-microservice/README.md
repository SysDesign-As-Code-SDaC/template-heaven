# {{ project_name | title }} - FastAPI Microservice

{{ project_description }}

## 🚀 Features

This template provides a production-ready FastAPI microservice with:

- **⚡ FastAPI** - Modern, fast web framework for building APIs
- **🐍 Python 3.11+** - Latest Python with async/await support
- **🗄️ PostgreSQL** - Robust relational database with async support
- **🔐 JWT Authentication** - Secure token-based authentication
- **📊 Prometheus Metrics** - Built-in monitoring and observability
- **📝 OpenAPI/Swagger** - Auto-generated API documentation
- **🧪 Pytest** - Comprehensive testing with async support
- **🐳 Docker** - Containerized development and production
- **🔄 GitHub Actions** - CI/CD pipeline with quality gates
- **📈 Structured Logging** - JSON logging with correlation IDs
- **🛡️ Security** - Input validation, CORS, rate limiting
- **🌐 Async Support** - Full async/await throughout the stack
- **📦 Dependency Injection** - Clean architecture with DI

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.11+, SQLAlchemy 2.0
- **Database**: PostgreSQL, Alembic migrations
- **Authentication**: JWT tokens, OAuth2
- **Monitoring**: Prometheus, Grafana, Sentry
- **Testing**: Pytest, Testcontainers, Factory Boy
- **Deployment**: Docker, Kubernetes ready
- **Documentation**: Sphinx, OpenAPI/Swagger

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Docker (optional)
- Git

### Installation

```bash
# Clone and setup
git clone <your-repo-url>
cd {{ project_name }}
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -e ".[dev]"

# Setup database
createdb {{ project_name }}_dev
alembic upgrade head

# Run tests
pytest

# Start development server
uvicorn app.main:app --reload
```

### Docker

```bash
# Development with Docker Compose
docker-compose up --build

# Production build
docker build -t {{ project_name }} .
docker run -p 8000:8000 {{ project_name }}
```

## 📁 Project Structure

```
{{ project_name }}/
├── app/                          # Application code
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   ├── config.py                 # Configuration management
│   ├── database.py               # Database connection
│   ├── dependencies.py           # Dependency injection
│   ├── middleware.py             # Custom middleware
│   ├── models/                   # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── base.py              # Base model
│   │   ├── user.py              # User model
│   │   └── item.py              # Item model
│   ├── schemas/                  # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── user.py              # User schemas
│   │   ├── item.py              # Item schemas
│   │   └── auth.py              # Auth schemas
│   ├── api/                      # API routes
│   │   ├── __init__.py
│   │   ├── deps.py              # Route dependencies
│   │   ├── v1/                  # API v1 routes
│   │   │   ├── __init__.py
│   │   │   ├── auth.py          # Authentication routes
│   │   │   ├── users.py         # User routes
│   │   │   └── items.py         # Item routes
│   │   └── health.py            # Health check routes
│   ├── core/                     # Core business logic
│   │   ├── __init__.py
│   │   ├── auth.py              # Authentication logic
│   │   ├── security.py          # Security utilities
│   │   └── exceptions.py        # Custom exceptions
│   ├── services/                 # Service layer
│   │   ├── __init__.py
│   │   ├── user_service.py      # User service
│   │   └── item_service.py      # Item service
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── logging.py           # Logging configuration
│       └── helpers.py           # Helper functions
├── tests/                        # Test suite
│   ├── conftest.py              # Test configuration
│   ├── unit/                    # Unit tests
│   │   ├── test_models.py
│   │   ├── test_services.py
│   │   └── test_utils.py
│   ├── integration/             # Integration tests
│   │   ├── test_api.py
│   │   └── test_database.py
│   └── e2e/                     # End-to-end tests
│       └── test_api_e2e.py
├── alembic/                     # Database migrations
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── docs/                        # Documentation
│   ├── source/
│   └── build/
├── scripts/                     # Utility scripts
│   ├── init_db.py
│   └── create_user.py
├── docker/                      # Docker configurations
│   ├── Dockerfile
│   ├── Dockerfile.prod
│   └── docker-compose.yml
├── .github/                     # GitHub Actions
│   └── workflows/
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container image
├── docker-compose.yml          # Local development
└── README.md                   # This file
```

## 🛠️ Development

### Available Scripts

```bash
# Development
uvicorn app.main:app --reload    # Start development server
python -m app.main               # Run application

# Testing
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest --cov=app               # With coverage
pytest tests/unit/             # Run unit tests only
pytest tests/integration/      # Run integration tests only

# Database
alembic revision --autogenerate -m "Description"  # Create migration
alembic upgrade head           # Apply migrations
alembic downgrade -1           # Rollback migration

# Code Quality
black app tests                # Format code
isort app tests                # Sort imports
flake8 app tests               # Lint code
mypy app                       # Type checking
bandit -r app/                 # Security scan

# Documentation
sphinx-build docs/source docs/build  # Build docs
```

### Code Quality

```bash
# Run all quality checks
pre-commit run --all-files

# Or manually
black app tests && isort app tests && flake8 app tests && mypy app
```

### Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires database)
pytest tests/integration/ -v

# E2E tests
pytest tests/e2e/ -v

# Test with coverage
pytest --cov=app --cov-report=html

# Test specific file
pytest tests/unit/test_models.py -v
```

## 🔧 Configuration

Configuration is managed through environment variables and Pydantic settings:

```python
# app/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "{{ project_name | title }}"
    VERSION: str = "{{ version }}"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost/{{ project_name }}"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # Monitoring
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9090
    SENTRY_DSN: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### Environment Variables

Create a `.env` file:

```bash
# Application
APP_NAME={{ project_name | title }}
VERSION={{ version }}
DEBUG=true
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/{{ project_name }}_dev

# Security
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Monitoring
ENABLE_METRICS=true
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=json
```

## 📚 API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Example API Usage

```python
import httpx

# Create a user
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/users/",
        json={
            "email": "user@example.com",
            "password": "securepassword",
            "full_name": "John Doe"
        }
    )
    user_data = response.json()

# Authenticate
response = await client.post(
    "http://localhost:8000/api/v1/auth/login",
    data={
        "username": "user@example.com",
        "password": "securepassword"
    }
)
token_data = response.json()
access_token = token_data["access_token"]

# Use authenticated endpoint
headers = {"Authorization": f"Bearer {access_token}"}
response = await client.get(
    "http://localhost:8000/api/v1/users/me",
    headers=headers
)
current_user = response.json()
```

## 🗄️ Database

### Models

```python
# app/models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.models.base import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

### Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Add user table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Show migration history
alembic history
```

## 🔐 Authentication

The service uses JWT tokens for authentication:

```python
# app/core/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
```

## 📊 Monitoring

### Prometheus Metrics

The service exposes Prometheus metrics at `/metrics`:

```python
# app/middleware.py
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Structured Logging

```python
# app/utils/logging.py
import structlog
import logging

def setup_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

## 🚀 Deployment

### Docker Production

```dockerfile
# docker/Dockerfile.prod
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as production

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ project_name }}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {{ project_name }}
  template:
    metadata:
      labels:
        app: {{ project_name }}
    spec:
      containers:
      - name: {{ project_name }}
        image: {{ project_name }}:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: {{ project_name }}-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: {{ project_name }}-secrets
              key: secret-key
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 🧪 Testing

### Unit Tests

```python
# tests/unit/test_models.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.user import User
from app.models.base import Base

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def test_create_user(db_session):
    user = User(
        email="test@example.com",
        hashed_password="hashedpassword",
        full_name="Test User"
    )
    db_session.add(user)
    db_session.commit()
    
    assert user.id is not None
    assert user.email == "test@example.com"
```

### Integration Tests

```python
# tests/integration/test_api.py
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_user():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/users/",
            json={
                "email": "test@example.com",
                "password": "testpassword",
                "full_name": "Test User"
            }
        )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `pre-commit run --all-files`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Standards

- **Type hints** required for all functions
- **Docstrings** for all public functions and classes
- **Tests** for all new functionality
- **Database migrations** for schema changes
- **API documentation** updates for endpoint changes

## 📄 License

This project is licensed under the {{ license }} License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI Team** for the excellent web framework
- **SQLAlchemy Team** for the powerful ORM
- **Pytest Team** for the testing framework
- **Template Heaven** for the microservice template

---

**Built with ❤️ using Template Heaven FastAPI Microservice Template**

*This template provides a solid foundation for building scalable, production-ready microservices with FastAPI and modern Python practices.*
