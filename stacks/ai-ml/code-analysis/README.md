# AI-Powered Code Analysis Template

A comprehensive containerized application for AI-driven code analysis, providing automated code review, bug detection, security vulnerability scanning, performance optimization suggestions, and intelligent code completion using multiple AI models and analysis engines.

## 🚀 Features

- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, Go, Rust, C++, and more
- **AI-Powered Analysis**: GPT-4, Claude, CodeLlama, and custom fine-tuned models
- **Security Scanning**: Automated vulnerability detection and security best practices
- **Performance Analysis**: Code optimization suggestions and performance bottleneck detection
- **Bug Detection**: Static analysis for common bugs and anti-patterns
- **Code Review Automation**: Intelligent pull request reviews and suggestions
- **Real-time Analysis**: Live code analysis as you type
- **Batch Processing**: Analyze entire codebases and generate comprehensive reports
- **Integration APIs**: RESTful APIs for automation pipeline integration
- **Custom Rules**: Extensible rule engine for custom analysis patterns
- **Containerized**: Full Docker deployment with scalable architecture

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for development)
- 8GB+ RAM (for AI model inference)
- API keys for AI services (optional, can use local models)

## 🛠️ Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone and navigate to the template
git clone <repository-url> code-analysis
cd code-analysis

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the analysis service
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# Analyze a code file
curl -X POST http://localhost:8000/api/analyze/file \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello_world():\n    print(\"Hello, World!\")\n\nhello_world()",
    "language": "python",
    "filename": "hello.py"
  }'

# Analyze a GitHub repository
curl -X POST http://localhost:8000/api/analyze/repository \
  -H "Content-Type: application/json" \
  -d '{
    "repository_url": "https://github.com/user/repo",
    "branch": "main",
    "analysis_types": ["security", "performance", "bugs"]
  }'
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Start the service
python -m uvicorn code_analysis.main:app --host 0.0.0.0 --port 8000 --reload

# Access web interface
open http://localhost:8501
```

## 📁 Project Structure

```
code-analysis/
├── src/
│   ├── code_analysis/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── core/
│   │   │   ├── analyzer.py      # Main analysis orchestration
│   │   │   ├── ai_engine.py     # AI model integration
│   │   │   ├── static_analyzer.py # Static analysis engine
│   │   │   ├── security_scanner.py # Security vulnerability scanner
│   │   │   ├── performance_analyzer.py # Performance analysis
│   │   │   ├── bug_detector.py  # Bug pattern detection
│   │   │   └── rule_engine.py   # Custom rule engine
│   │   ├── analyzers/
│   │   │   ├── python_analyzer.py # Python-specific analysis
│   │   │   ├── javascript_analyzer.py # JS/TS analysis
│   │   │   ├── java_analyzer.py # Java analysis
│   │   │   └── go_analyzer.py   # Go analysis
│   │   ├── ai/
│   │   │   ├── openai_client.py # OpenAI integration
│   │   │   ├── anthropic_client.py # Anthropic Claude
│   │   │   ├── local_llm.py     # Local model support
│   │   │   └── prompt_templates.py # Analysis prompts
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── analysis.py   # Analysis endpoints
│   │   │   │   ├── reports.py    # Report endpoints
│   │   │   │   ├── rules.py      # Rule management
│   │   │   │   └── webhooks.py   # automation pipeline integration
│   │   │   └── models.py         # Pydantic models
│   │   └── utils/
│   │       ├── config.py         # Configuration management
│   │       ├── caching.py        # Result caching
│   │       └── metrics.py        # Performance metrics
│   ├── tests/
│   │   ├── test_analyzers.py
│   │   ├── test_ai_integration.py
│   │   ├── test_api_endpoints.py
│   │   └── test_security_scanning.py
│   └── scripts/
│       ├── setup_models.py       # AI model setup
│       ├── train_custom_rules.py # Rule training
│       └── benchmark_analysis.py # Performance benchmarking
├── models/
│   ├── custom-rules/             # Custom analysis rules
│   ├── fine-tuned/               # Fine-tuned AI models
│   └── cache/                    # Analysis result cache
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── docker-compose.gpu.yml
│   └── nginx.conf
├── docs/
│   ├── api.md                    # API documentation
│   ├── analyzers.md              # Analyzer documentation
│   ├── rules.md                  # Custom rules guide
│   └── integration.md            # automation pipeline integration
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

### Environment Variables

```bash
# Server Configuration
ANALYSIS_HOST=0.0.0.0
ANALYSIS_PORT=8000
ANALYSIS_WORKERS=4

# AI Model Configuration
AI_PROVIDER=openai  # openai, anthropic, local, mixed
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
LOCAL_MODEL_PATH=./models/fine-tuned

# Analysis Configuration
SUPPORTED_LANGUAGES=python,javascript,typescript,java,go,rust,cpp
MAX_FILE_SIZE=1MB
ANALYSIS_TIMEOUT=300
CACHE_TTL=3600

# Security Configuration
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
API_KEY_REQUIRED=true
ANALYSIS_API_KEY=your-analysis-key

# Performance Configuration
ENABLE_GPU=true
BATCH_SIZE=10
MAX_CONCURRENT_ANALYSES=5

# Logging and Monitoring
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/test_api_endpoints.py -v

# Test AI model integration
pytest tests/test_ai_integration.py -v

# Performance benchmarking
python scripts/benchmark_analysis.py --language python --iterations 100

# Security testing
pytest tests/test_security_scanning.py -v
```

## 🚀 Deployment

### Production Deployment

```bash
# Build production image
docker build -f docker/Dockerfile.prod -t code-analysis:prod .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# GPU-enabled deployment
docker-compose -f docker-compose.gpu.yml up -d
```

### Scaling

```bash
# Horizontal scaling
docker-compose up -d --scale code-analysis=5

# Load balancing
docker-compose -f docker-compose.lb.yml up -d
```

## 📚 API Reference

### Analysis Endpoints

```bash
# Analyze single file
POST /api/analyze/file
{
  "code": "def hello():\n    print('Hello')",
  "language": "python",
  "filename": "hello.py",
  "analysis_types": ["security", "performance", "bugs"]
}

# Analyze repository
POST /api/analyze/repository
{
  "repository_url": "https://github.com/user/repo",
  "branch": "main",
  "path_filter": "*.py",
  "analysis_types": ["security", "performance"]
}

# Real-time analysis
POST /api/analyze/realtime
{
  "code": "current code content",
  "cursor_position": {"line": 10, "column": 5},
  "language": "python"
}
```

### Report Endpoints

```bash
# Get analysis report
GET /api/reports/{analysis_id}

# List recent analyses
GET /api/reports?limit=10&status=completed

# Export report
GET /api/reports/{analysis_id}/export?format=json

# Delete old reports
DELETE /api/reports/cleanup?older_than_days=30
```

## 🔒 Security Features

### Code Security
- **Vulnerability Detection**: Automated CVE and security issue detection
- **Secret Scanning**: Hardcoded credentials and API keys detection
- **Injection Prevention**: SQL injection and XSS vulnerability detection
- **Access Control**: Proper file permission and access pattern analysis

### API Security
- **Authentication**: API key and JWT token authentication
- **Rate Limiting**: Configurable request rate limiting
- **Input Validation**: Comprehensive input sanitization and validation
- **Audit Logging**: Complete audit trail of all analysis operations

## 📊 Analysis Types

### Security Analysis
- **Vulnerability Scanning**: Known security vulnerabilities
- **Code Injection**: SQL injection, XSS, command injection detection
- **Authentication Issues**: Weak authentication, hardcoded credentials
- **Access Control**: Insecure direct object references, privilege escalation

### Performance Analysis
- **Algorithm Complexity**: Time and space complexity analysis
- **Resource Usage**: Memory leaks, inefficient data structures
- **Database Queries**: N+1 queries, missing indexes
- **Caching Opportunities**: Cache implementation suggestions

### Bug Detection
- **Logic Errors**: Null pointer exceptions, division by zero
- **Type Errors**: Type mismatches and conversion issues
- **Concurrency Issues**: Race conditions, deadlocks
- **Resource Management**: File handles, database connections

### Code Quality
- **Best Practices**: Language-specific best practices
- **Code Smells**: Maintainability and readability issues
- **Documentation**: Missing docstrings, comments
- **Testing**: Test coverage and test quality analysis

## 🤖 AI Integration

### Supported AI Models

#### OpenAI GPT
- **GPT-4**: Advanced reasoning and analysis
- **GPT-3.5-turbo**: Fast and cost-effective analysis

#### Anthropic Claude
- **Claude 2**: Balanced performance and reasoning
- **Claude Instant**: Fast responses for simple analysis

#### Local Models
- **CodeLlama**: Meta's code-specialized models
- **StarCoder**: Large code generation models
- **Custom Fine-tuned**: Domain-specific analysis models

### Custom AI Training

```bash
# Prepare training data
python scripts/prepare_training_data.py \
  --source-repos repos.json \
  --output training_data.jsonl

# Fine-tune model
python scripts/fine_tune_model.py \
  --training-data training_data.jsonl \
  --base-model codellama \
  --output-model custom-analyzer

# Evaluate model
python scripts/evaluate_model.py \
  --model custom-analyzer \
  --test-data test_cases.json
```

## 🔧 Custom Rules Engine

### Creating Custom Rules

```python
# custom_rules/security_rules.py
from code_analysis.core.rule_engine import AnalysisRule, RuleSeverity

class CustomSecurityRule(AnalysisRule):
    """Custom security rule example."""

    name = "custom_insecure_random"
    severity = RuleSeverity.HIGH
    description = "Use of insecure random number generation"

    def analyze(self, code: str, language: str) -> List[AnalysisIssue]:
        issues = []

        if language == "python":
            # Check for insecure random usage
            if "random.random()" in code or "random.randint(" in code:
                issues.append(AnalysisIssue(
                    rule=self.name,
                    severity=self.severity,
                    message="Use secrets module for cryptographic purposes",
                    line=self._find_line_number(code, "random."),
                    suggestion="Use secrets.randbelow() or secrets.token_bytes()"
                ))

        return issues
```

### Rule Management

```bash
# List available rules
curl http://localhost:8000/api/rules

# Add custom rule
curl -X POST http://localhost:8000/api/rules \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_rule",
    "code": "rule implementation",
    "language": "python",
    "severity": "medium"
  }'

# Test rule
curl -X POST http://localhost:8000/api/rules/test \
  -H "Content-Type: application/json" \
  -d '{
    "rule_name": "custom_rule",
    "test_code": "test code here"
  }'
```

## 📈 Performance Optimization

### Configuration Tuning

```python
# High-performance configuration
MAX_CONCURRENT_ANALYSES=20
BATCH_SIZE=50
CACHE_TTL=7200
ANALYSIS_TIMEOUT=180

# AI model optimization
AI_MODEL_CACHE_SIZE=10
AI_REQUEST_BATCH_SIZE=5
AI_MAX_RETRIES=3
```

### Hardware Acceleration

```bash
# GPU acceleration
docker run --gpus all \
  -e ENABLE_GPU=true \
  -e GPU_MEMORY_FRACTION=0.8 \
  code-analysis:latest

# CPU optimization
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

## 🔗 Automation Integration

### Automation Integration (GitHub Actions disabled)

```yaml
# .github/workflows/code-analysis.yml
name: Code Analysis

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Code Analysis
        run: |
          curl -X POST http://your-analysis-service.com/api/analyze/repository \
            -H "Authorization: Bearer ${{ secrets.ANALYSIS_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d '{
              "repository_url": "${{ github.repositoryUrl }}",
              "branch": "${{ github.head_ref || github.ref_name }}",
              "commit_sha": "${{ github.sha }}"
            }'

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            // Post analysis results as PR comment
```

### Other Automation Platforms

```bash
# Jenkins pipeline
pipeline {
    agent any
    stages {
        stage('Code Analysis') {
            steps {
                sh '''
                    curl -X POST ${ANALYSIS_SERVICE_URL}/api/analyze/repository \
                      -H "Authorization: Bearer ${ANALYSIS_API_KEY}" \
                      -H "Content-Type: application/json" \
                      -d "{
                        \\"repository_url\\": \\"${GIT_URL}\\"",
                        \\"branch\\": \\"${BRANCH_NAME}\\"",
                        \\"commit_sha\\": \\"${GIT_COMMIT}\\""
                      }"
                '''
            }
        }
    }
}
```

## 📊 Monitoring and Metrics

### Key Metrics
- **Analysis Performance**: Response times and throughput
- **AI Model Usage**: Token consumption and cost tracking
- **Rule Effectiveness**: True positive rates and false positives
- **System Health**: Resource usage and error rates

### Dashboard Access

```bash
# Metrics endpoint
curl http://localhost:9090/metrics

# Analysis statistics
curl http://localhost:8000/api/stats

# Performance dashboard
open http://localhost:3000/dashboards/code-analysis
```

## 🔧 Extension and Customization

### Adding New Language Support

```python
# src/code_analysis/analyzers/rust_analyzer.py
from code_analysis.core.analyzer import BaseAnalyzer

class RustAnalyzer(BaseAnalyzer):
    """Rust code analyzer."""

    language = "rust"
    file_extensions = [".rs"]

    def analyze_syntax(self, code: str) -> List[AnalysisIssue]:
        # Implement Rust-specific syntax analysis
        pass

    def analyze_security(self, code: str) -> List[AnalysisIssue]:
        # Implement Rust security analysis
        pass

    def analyze_performance(self, code: str) -> List[AnalysisIssue]:
        # Implement Rust performance analysis
        pass
```

### Custom AI Prompts

```python
# src/code_analysis/ai/prompt_templates.py
CUSTOM_ANALYSIS_PROMPT = """
Analyze the following {language} code for:

1. Security vulnerabilities
2. Performance issues
3. Code quality problems
4. Best practice violations

Code:
{code}

Provide detailed analysis with specific recommendations.
"""
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

- [MCP Middleware](../mcp-middleware/) - AI assistant integration
- [RAG System](../rag-system/) - Document analysis and Q&A
- [FastAPI Microservice](../fastapi-microservice/) - API foundation

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in `docs/`
- Join our community Discord
