# {{ project_name }}

{{ project_description }}

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone {{ repository_url }}
   cd {{ project_name }}
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   ```

4. **Setup environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Setup database**
   ```bash
   make db-upgrade
   ```

6. **Start development server**
   ```bash
   make dev
   ```

The API will be available at `http://localhost:8000`

## 📚 Documentation

- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health/live

## 🏗️ Architecture

### Technology Stack

- **Backend**: FastAPI with async support
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Cache**: Redis for caching and sessions
- **Authentication**: JWT with refresh tokens
- **Testing**: Pytest with comprehensive coverage
- **Documentation**: Sphinx with auto-generated API docs
- **Deployment**: Docker with Kubernetes manifests
- **Monitoring**: Prometheus metrics with structured logging

### Project Structure

```
{{ project_name }}/
├── app/                          # Main application code
│   ├── main.py                   # FastAPI application entry point
│   ├── core/                     # Core business logic
│   │   ├── config.py             # Configuration management
│   │   ├── database.py           # Database connection and session
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── security.py           # Authentication and security
│   │   └── exceptions.py         # Custom exceptions
│   ├── api/                      # API layer
│   │   ├── dependencies.py       # API dependencies
│   │   └── v1/                   # API version 1
│   │       ├── auth.py           # Authentication endpoints
│   │       ├── users.py          # User management endpoints
│   │       └── posts.py          # Post management endpoints
│   ├── core/services/            # Business logic services
│   └── utils/                    # Utility functions
│       └── logging.py            # Logging configuration
├── tests/                        # Comprehensive test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── e2e/                      # End-to-end tests
│   └── performance/              # Performance tests
├── docs/                         # Complete documentation
├── .github/workflows/            # CI/CD pipeline
├── .cursor/rules/                # AI coding agent rules
├── k8s/                          # Kubernetes manifests
├── alembic/                      # Database migrations
├── pyproject.toml                # Python project configuration
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Development environment
├── Makefile                      # Development commands
└── README.md                     # This file
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
make test

# Run specific test types
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-e2e          # End-to-end tests only
make test-performance  # Performance tests only

# Run tests with coverage
make test-coverage
```

### Test Structure

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test API endpoints and database interactions
- **End-to-End Tests**: Test complete user workflows
- **Performance Tests**: Test system performance and benchmarks

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security checks
make security

# Run all checks
make check
```

### Database Management

```bash
# Create new migration
make db-revision message="Add new table"

# Apply migrations
make db-upgrade

# Rollback migration
make db-downgrade

# Reset database (WARNING: Deletes all data)
make db-reset
```

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
make pre-commit
```

## 🐳 Docker

### Development with Docker

```bash
# Start development environment
make docker-dev

# Build and run production container
make docker-build
make docker-run
```

### Docker Compose Services

- **{{ project_name }}**: Main application
- **postgres**: PostgreSQL database
- **redis**: Redis cache
- **celery-worker**: Background task worker
- **celery-beat**: Scheduled task scheduler
- **flower**: Celery monitoring
- **nginx**: Reverse proxy (optional)

## 🚀 Deployment

### Production Deployment

1. **Build Docker image**
   ```bash
   make docker-build
   ```

2. **Deploy with Docker Compose**
   ```bash
   docker-compose up -d
   ```

3. **Deploy with Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

### Environment Variables

Required environment variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
ALLOWED_HOSTS=localhost,127.0.0.1
```

## 📊 Monitoring

### Health Checks

- **Liveness**: `/health/live`
- **Readiness**: `/health/ready`
- **Metrics**: `/metrics`

### Logging

The application uses structured JSON logging with:
- Request correlation IDs
- Performance metrics
- Security events
- Business events

### Metrics

Prometheus metrics are available at `/metrics`:
- HTTP request metrics
- Database query metrics
- Custom business metrics

## 🔒 Security

### Security Features

- **JWT Authentication** with refresh tokens
- **Password Hashing** using bcrypt
- **Input Validation** with Pydantic models
- **SQL Injection Protection** with SQLAlchemy ORM
- **CORS Configuration** for cross-origin requests
- **Rate Limiting** to prevent abuse
- **Security Headers** for enhanced protection

### Security Scanning

```bash
# Run security checks
make security

# Comprehensive security scan
make security-full
```

## 🤖 AI Coding Agent Support

This project includes comprehensive AI coding agent support:

- **Cursor AI Rules**: Located in `.cursor/rules/`
- **AI Configuration**: Project-specific AI settings
- **Development Workflow**: AI-optimized development process
- **Code Generation**: AI-assisted code generation with best practices

### AI Setup

```bash
# Setup AI context
python scripts/setup_ai_context.py

# Verify AI integration
python scripts/verify_ai_setup.py
```

## 📈 Performance

### Benchmarking

```bash
# Run performance benchmarks
make benchmark

# Compare with previous benchmark
make benchmark-compare
```

### Performance Features

- **Async/Await** for I/O operations
- **Connection Pooling** for database
- **Redis Caching** for frequently accessed data
- **Prometheus Metrics** for monitoring
- **Performance Logging** for optimization

## 🛠️ Available Commands

Run `make help` to see all available commands:

```bash
make help
```

### Common Commands

- `make dev` - Start development server
- `make test` - Run all tests
- `make lint` - Run linting
- `make format` - Format code
- `make security` - Run security checks
- `make docker-dev` - Start development with Docker
- `make docs` - Generate documentation

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests and checks**
   ```bash
   make check
   ```
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Workflow

1. **Setup development environment**
   ```bash
   make setup
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make changes and test**
   ```bash
   make test
   make check
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin feature/your-feature
   ```

## 📄 License

This project is licensed under the {{ license }} License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** for the excellent web framework
- **SQLAlchemy** for the powerful ORM
- **Pytest** for the testing framework
- **Sphinx** for documentation generation
- **Docker** for containerization

## 📞 Support

- **Documentation**: Check this README and the docs/ directory
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact {{ author_email }} for support

---

**Built with ❤️ by {{ author }}**

For more information, visit our [GitHub repository]({{ repository_url }}).