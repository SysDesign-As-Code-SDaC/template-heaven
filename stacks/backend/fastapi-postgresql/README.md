# FastAPI PostgreSQL Backend Template

A comprehensive FastAPI backend template with PostgreSQL, JWT authentication, comprehensive tooling, and production-ready deployment. This template follows all gold standard practices for automated repo management, testing, documentation, and deployment.

## 🚀 Features

### Core Features
- **FastAPI** with async support and automatic OpenAPI documentation
- **PostgreSQL** with SQLAlchemy ORM and async drivers
- **JWT Authentication** with refresh tokens and secure password hashing
- **User Management** with registration, login, profile management
- **Role-based Access Control** with admin and user permissions
- **Email Verification** and password reset functionality
- **Session Management** with secure token handling

### Development & Quality Assurance
- **Comprehensive Testing**: Unit, integration, and end-to-end tests
- **Code Quality**: Black, isort, flake8, mypy, pylint
- **Security Scanning**: Bandit, Safety, pip-audit
- **Type Checking**: Full mypy configuration
- **Pre-commit Hooks**: Automated code quality checks

### Deployment & Infrastructure
- **Docker & Docker Compose**: Multi-stage builds for all environments
- **Production Ready**: Gunicorn, health checks, monitoring
- **Database Migrations**: Alembic for schema management
- **Environment Configuration**: Pydantic settings management
- **Monitoring**: Prometheus metrics and structured logging

### Documentation & Tooling
- **Auto-generated API Docs**: Swagger UI and ReDoc
- **Comprehensive Makefile**: 50+ development commands
- **Automation Ready**: GitHub Actions examples (disabled)
- **Development Scripts**: Database seeding, user management
- **Performance Benchmarking**: pytest-benchmark integration

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Security](#-security)
- [Monitoring](#-monitoring)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis 6+ (optional, for caching)
- Docker & Docker Compose (recommended)

### Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd fastapi-postgresql-backend
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
   make env  # Creates .env from template
   # Edit .env with your database credentials
   ```

5. **Setup database**
   ```bash
   make db-init
   make db-upgrade
   ```

6. **Start development server**
   ```bash
   make dev
   ```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

### Docker Setup (Recommended)

```bash
# Start complete development environment
make docker-dev

# Or use docker-compose directly
docker-compose -f docker/docker-compose.dev.yml up --build
```

## 📁 Project Structure

```
fastapi-postgresql-backend/
├── src/app/                          # Main application code
│   ├── __init__.py                   # Package initialization
│   ├── main.py                       # FastAPI application entry point
│   ├── core/                         # Core functionality
│   │   ├── config.py                 # Configuration management
│   │   ├── database.py               # Database connection and session
│   │   ├── security.py               # Authentication and security
│   │   └── exceptions.py             # Custom exceptions
│   ├── models/                       # SQLAlchemy models
│   │   └── user.py                   # User database model
│   ├── schemas/                      # Pydantic schemas
│   │   └── user.py                   # User request/response schemas
│   ├── services/                     # Business logic services
│   │   └── user_service.py           # User management service
│   ├── api/                          # API layer
│   │   └── v1/                       # API version 1
│   │       ├── api.py                # API router
│   │       ├── auth.py               # Authentication endpoints
│   │       └── users.py              # User management endpoints
│   └── utils/                        # Utility functions
│       ├── logging.py                # Logging configuration
│       └── metrics.py                # Prometheus metrics
├── tests/                            # Comprehensive test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── e2e/                          # End-to-end tests
│   └── performance/                  # Performance tests
├── docs/                             # Documentation
├── scripts/                          # Utility scripts
├── docker/                           # Docker configurations
│   ├── docker-compose.yml            # Production setup
│   ├── docker-compose.dev.yml        # Development setup
│   └── monitoring/                   # Monitoring configurations
├── .github/workflows/                # Automation workflows
├── alembic/                          # Database migrations
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── pyproject.toml                    # Python project configuration
├── Makefile                          # Development commands
├── Dockerfile                        # Multi-stage container build
└── README.md                         # This file
```

## 📚 API Documentation

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | User login |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| POST | `/api/v1/auth/logout` | User logout |
| GET | `/api/v1/auth/me` | Get current user profile |
| PUT | `/api/v1/auth/me` | Update user profile |
| POST | `/api/v1/auth/change-password` | Change password |

### User Management Endpoints (Admin)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/users/` | List users (paginated) |
| GET | `/api/v1/users/{user_id}` | Get user by ID |
| PUT | `/api/v1/users/{user_id}` | Update user |
| DELETE | `/api/v1/users/{user_id}` | Delete user |
| POST | `/api/v1/users/bulk-deactivate` | Bulk deactivate users |
| GET | `/api/v1/users/stats` | User statistics |

### Health & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/metrics` | Prometheus metrics |
| GET | `/api/v1/health` | API health check |

## 🧪 Testing

### Test Structure

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

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test API endpoints and database interactions
- **End-to-End Tests**: Test complete user workflows
- **Performance Tests**: Benchmark critical operations

### Test Coverage

The template maintains >80% code coverage with comprehensive testing for:
- Authentication flows
- User management operations
- API endpoint validation
- Database operations
- Error handling scenarios

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
# Create migration
make db-revision message="Add new table"

# Apply migrations
make db-upgrade

# Rollback migration
make db-downgrade

# Reset database (WARNING: Deletes all data)
make db-reset

# Connect to database shell
make db-shell
```

### Development Commands

```bash
# Start development server
make dev

# Start with debug logging
make dev-debug

# Create superuser
make create-superuser

# Seed database with sample data
make seed

# Generate API documentation
make api-docs

# View all available commands
make help
```

## 🐳 Docker Development

### Development Environment

```bash
# Start full development stack
make docker-dev

# View logs
make logs

# Run tests in container
make docker-test

# Connect to database
make db-shell
```

### Services Included

- **FastAPI App**: Main application with hot reload
- **PostgreSQL**: Database with persistent storage
- **Redis**: Cache and session storage
- **PgAdmin**: Database administration interface
- **Redis Commander**: Redis management interface
- **MailHog**: Email testing tool (for development)

## 🚀 Deployment

### Production Deployment

```bash
# Build production image
make docker-build

# Deploy with docker-compose
docker-compose up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

### Environment Configuration

Required environment variables for production:

```bash
# Database
POSTGRES_SERVER=your-db-host
POSTGRES_USER=your-db-user
POSTGRES_PASSWORD=your-db-password
POSTGRES_DB=your-db-name

# Security
SECRET_KEY=your-secret-key-here
DEBUG=false

# Email (optional)
SMTP_HOST=your-smtp-host
SMTP_USER=your-smtp-user
SMTP_PASSWORD=your-smtp-password
EMAILS_FROM_EMAIL=noreply@yourdomain.com
```

### Production Checklist

- [ ] Set strong SECRET_KEY
- [ ] Configure production database
- [ ] Set DEBUG=false
- [ ] Configure ALLOWED_HOSTS
- [ ] Set up monitoring and logging
- [ ] Configure HTTPS certificates
- [ ] Set up database backups
- [ ] Configure rate limiting
- [ ] Set up health checks

## 📊 Monitoring

### Health Checks

- **Liveness**: `/health/live` - Application is running
- **Readiness**: `/health/ready` - Application can serve requests
- **Database**: Checks database connectivity

### Metrics

Prometheus metrics available at `/metrics`:
- HTTP request metrics
- Database query metrics
- Authentication metrics
- Error rates and response times

### Logging

Structured JSON logging with:
- Request correlation IDs
- Performance metrics
- Security events
- Business events
- Error tracking

## 🔒 Security

### Authentication & Authorization

- **JWT Tokens**: Secure token-based authentication
- **Password Hashing**: bcrypt with salt rounds
- **Refresh Tokens**: Secure token rotation
- **Session Management**: Automatic cleanup

### Security Features

- **Input Validation**: Pydantic model validation
- **SQL Injection Protection**: SQLAlchemy ORM
- **XSS Protection**: Content Security Policy headers
- **CSRF Protection**: Token-based protection
- **Rate Limiting**: Request rate limiting
- **Security Headers**: OWASP recommended headers

### Security Scanning

```bash
# Run security checks
make security

# Comprehensive security audit
make security-full
```

## 🤖 AI Coding Agent Support

This template includes comprehensive AI coding agent support:

- **Cursor Rules**: Located in `.cursor/rules/`
- **Development Guidelines**: AI-assisted development patterns
- **Code Generation**: Best practices for AI-generated code
- **Review Checklists**: Automated code review guidelines

### AI Development Setup

```bash
# Setup AI context (if available)
python scripts/setup_ai_context.py

# Verify AI integration
python scripts/verify_ai_setup.py
```

## 📈 Performance

### Benchmarking

```bash
# Run performance benchmarks
make benchmark

# Compare with previous results
make benchmark-compare
```

### Performance Features

- **Async/Await**: Non-blocking I/O operations
- **Connection Pooling**: Database connection optimization
- **Redis Caching**: Frequently accessed data caching
- **Query Optimization**: Efficient database queries
- **Response Compression**: Gzip compression

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
- `make db-upgrade` - Run database migrations
- `make create-superuser` - Create admin user
- `make health` - Check application health

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
   make test
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** for the excellent web framework
- **SQLAlchemy** for the powerful ORM
- **Pydantic** for data validation
- **PostgreSQL** for the robust database
- **Docker** for containerization
- **Prometheus** for monitoring

---

**Built with ❤️ using Template Heaven's gold standard practices**

For more information, visit our [GitHub repository](https://github.com/template-heaven/templateheaven).
