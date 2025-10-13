# GitHub Actions CI/CD Templates

A collection of production-ready GitHub Actions workflows for various development stacks and deployment scenarios.

## 🚀 Features

- **Multi-Stack Support** (Node.js, Python, Go, Rust, Java, etc.)
- **Testing Workflows** with coverage reporting
- **Security Scanning** with CodeQL and dependency checks
- **Docker Build & Push** to container registries
- **Deployment Workflows** to various platforms
- **Release Automation** with semantic versioning
- **Matrix Testing** for multiple versions
- **Caching** for faster builds
- **Secrets Management** best practices

## 📋 Prerequisites

- GitHub repository
- GitHub Actions enabled
- Required secrets configured

## 🛠️ Quick Start

### 1. Copy Workflow Files

Copy the desired workflow files to `.github/workflows/` in your repository:

```bash
mkdir -p .github/workflows
cp stacks/devops/github-actions/workflows/* .github/workflows/
```

### 2. Configure Secrets

Add required secrets in your GitHub repository settings:

```bash
# For Docker deployments
DOCKER_USERNAME
DOCKER_PASSWORD
DOCKER_REGISTRY

# For cloud deployments
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AZURE_CREDENTIALS
GCP_SA_KEY

# For package registries
NPM_TOKEN
PYPI_TOKEN
```

### 3. Customize Workflows

Edit the workflow files to match your project requirements:

- Update Node.js/Python versions
- Modify test commands
- Configure deployment targets
- Adjust notification settings

## 📁 Available Workflows

### Node.js/TypeScript

```yaml
# .github/workflows/nodejs.yml
name: Node.js CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [18.x, 20.x]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run linter
      run: npm run lint
    
    - name: Run tests
      run: npm run test:coverage
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage/lcov.info
```

### Python

```yaml
# .github/workflows/python.yml
name: Python CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linter
      run: |
        flake8 src/ tests/
        black --check src/ tests/
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Docker Build & Push

```yaml
# .github/workflows/docker.yml
name: Docker Build & Push

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ secrets.DOCKER_USERNAME }}/${{ github.event.repository.name }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

### Security Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1' # Weekly on Monday at 2 AM

jobs:
  codeql:
    runs-on: ubuntu-latest
    
    strategy:
      fail-fast: false
      matrix:
        language: [ 'javascript', 'python' ]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v3
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3

  dependency-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'
```

### Deployment to AWS

```yaml
# .github/workflows/deploy-aws.yml
name: Deploy to AWS

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service \
          --cluster ${{ secrets.ECS_CLUSTER }} \
          --service ${{ secrets.ECS_SERVICE }} \
          --force-new-deployment
    
    - name: Deploy to Lambda
      run: |
        zip -r function.zip .
        aws lambda update-function-code \
          --function-name ${{ secrets.LAMBDA_FUNCTION }} \
          --zip-file fileb://function.zip
```

## 🔧 Workflow Configuration

### Environment Variables

```yaml
env:
  NODE_ENV: production
  NPM_CONFIG_CACHE: ~/.npm
  PYTHON_VERSION: 3.11
  DOCKER_BUILDKIT: 1
```

### Caching

```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.npm
      node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

### Matrix Strategies

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    node-version: [18.x, 20.x]
    exclude:
      - os: windows-latest
        node-version: 18.x
```

## 🚀 Deployment Strategies

### Blue-Green Deployment

```yaml
- name: Blue-Green Deployment
  run: |
    # Deploy to staging (green)
    kubectl apply -f k8s/green/
    
    # Run health checks
    kubectl wait --for=condition=ready pod -l app=myapp-green
    
    # Switch traffic to green
    kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'
    
    # Clean up blue
    kubectl delete -f k8s/blue/
```

### Rolling Deployment

```yaml
- name: Rolling Deployment
  run: |
    kubectl set image deployment/myapp myapp=${{ secrets.DOCKER_USERNAME }}/myapp:${{ github.sha }}
    kubectl rollout status deployment/myapp
    kubectl rollout history deployment/myapp
```

## 📊 Monitoring & Notifications

### Slack Notifications

```yaml
- name: Notify Slack
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#deployments'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
  if: always()
```

### Discord Notifications

```yaml
- name: Notify Discord
  uses: Ilshidur/action-discord@master
  with:
    webhook: ${{ secrets.DISCORD_WEBHOOK }}
    args: 'Deployment completed successfully! 🚀'
```

## 📚 Learning Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Actions](https://github.com/docker/actions)
- [AWS Actions](https://github.com/aws-actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

## 🔗 Upstream Source

- **Repository**: [actions/starter-workflows](https://github.com/actions/starter-workflows)
- **Documentation**: [docs.github.com/actions](https://docs.github.com/en/actions)
- **License**: MIT
