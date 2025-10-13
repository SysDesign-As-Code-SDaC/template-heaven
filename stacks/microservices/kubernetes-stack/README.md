# Kubernetes Microservices Stack

A production-ready microservices architecture template using Kubernetes, featuring modern container orchestration, service mesh, and observability for 2025.

## 🚀 Features

- **Kubernetes** - Container orchestration platform
- **Helm Charts** - Package management for Kubernetes
- **ArgoCD** - GitOps continuous deployment
- **Istio Service Mesh** - Traffic management and security
- **Prometheus & Grafana** - Monitoring and observability
- **Jaeger** - Distributed tracing
- **ELK Stack** - Centralized logging
- **Redis** - Caching and session storage
- **PostgreSQL** - Primary database
- **MongoDB** - Document database
- **Nginx Ingress** - Load balancing and SSL termination
- **Cert-Manager** - Automatic SSL certificate management

## 📋 Prerequisites

- Kubernetes cluster (v1.25+)
- kubectl configured
- Helm 3.x
- Docker
- Git

## 🛠️ Quick Start

### 1. Clone Repository

```bash
git clone <this-repo> microservices-stack
cd microservices-stack
```

### 2. Install Prerequisites

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install ArgoCD CLI
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
rm argocd-linux-amd64
```

### 3. Deploy Infrastructure

```bash
# Deploy ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Deploy Istio
istioctl install --set values.defaultRevision=default

# Deploy monitoring stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace
```

### 4. Deploy Applications

```bash
# Deploy applications via ArgoCD
kubectl apply -f argocd-applications/
```

## 📁 Project Structure

```
├── applications/              # Microservice applications
│   ├── user-service/         # User management service
│   ├── order-service/        # Order processing service
│   ├── payment-service/      # Payment processing service
│   ├── notification-service/ # Notification service
│   └── api-gateway/          # API Gateway
├── infrastructure/           # Infrastructure components
│   ├── databases/            # Database configurations
│   ├── messaging/            # Message queue configurations
│   ├── monitoring/           # Monitoring stack
│   └── security/             # Security configurations
├── helm-charts/              # Custom Helm charts
│   ├── microservice/         # Generic microservice chart
│   └── database/             # Database chart
├── k8s-manifests/            # Kubernetes manifests
│   ├── namespaces/           # Namespace definitions
│   ├── configmaps/           # Configuration maps
│   ├── secrets/              # Secret definitions
│   └── ingress/              # Ingress configurations
├── argocd-applications/      # ArgoCD application definitions
├── istio/                    # Istio configurations
│   ├── gateway/              # Gateway configurations
│   ├── virtual-services/     # Virtual service definitions
│   ├── destination-rules/    # Destination rule definitions
│   └── authorization/        # Authorization policies
└── scripts/                  # Deployment and utility scripts
```

## 🔧 Available Scripts

```bash
# Infrastructure Management
./scripts/setup-cluster.sh           # Setup Kubernetes cluster
./scripts/install-istio.sh           # Install Istio service mesh
./scripts/install-monitoring.sh      # Install monitoring stack
./scripts/install-argocd.sh          # Install ArgoCD

# Application Deployment
./scripts/deploy-apps.sh             # Deploy all applications
./scripts/deploy-service.sh user-service # Deploy specific service
./scripts/rollback-service.sh user-service # Rollback service

# Monitoring and Debugging
./scripts/port-forward.sh            # Port forward services
./scripts/logs.sh user-service       # View service logs
./scripts/describe-pod.sh user-service # Describe pod details

# Database Management
./scripts/backup-database.sh         # Backup databases
./scripts/restore-database.sh        # Restore databases
./scripts/migrate-database.sh        # Run database migrations
```

## 🏗️ Microservice Architecture

### User Service

```yaml
# applications/user-service/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  namespace: microservices
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
        version: v1
    spec:
      containers:
      - name: user-service
        image: user-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: user-service-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: user-service-secrets
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Configuration

```yaml
# applications/user-service/k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: user-service
  namespace: microservices
spec:
  selector:
    app: user-service
  ports:
  - name: http
    port: 80
    targetPort: 8080
  type: ClusterIP
```

### Istio Virtual Service

```yaml
# istio/virtual-services/user-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: user-service
  namespace: microservices
spec:
  hosts:
  - user-service
  http:
  - match:
    - headers:
        version:
          exact: v2
    route:
    - destination:
        host: user-service
        subset: v2
  - route:
    - destination:
        host: user-service
        subset: v1
      weight: 90
    - destination:
        host: user-service
        subset: v2
      weight: 10
```

## 📊 Monitoring Configuration

### Prometheus ServiceMonitor

```yaml
# infrastructure/monitoring/prometheus-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: user-service-monitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: user-service
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "User Service Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{service=\"user-service\"}[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{service=\"user-service\"}[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## 🔍 Distributed Tracing

### Jaeger Configuration

```yaml
# infrastructure/monitoring/jaeger-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-config
  namespace: monitoring
data:
  jaeger.yaml: |
    service:
      name: user-service
    sampling:
      type: const
      param: 1
    reporter:
      logSpans: true
      localAgentHostPort: jaeger-agent:6831
```

### Application Tracing

```go
// Example Go service with tracing
package main

import (
    "context"
    "github.com/opentracing/opentracing-go"
    "github.com/uber/jaeger-client-go"
    "github.com/uber/jaeger-client-go/config"
)

func initTracing() {
    cfg := config.Configuration{
        ServiceName: "user-service",
        Sampler: &config.SamplerConfig{
            Type:  jaeger.SamplerTypeConst,
            Param: 1,
        },
        Reporter: &config.ReporterConfig{
            LogSpans: true,
        },
    }
    
    tracer, _, _ := cfg.NewTracer()
    opentracing.SetGlobalTracer(tracer)
}

func getUser(ctx context.Context, userID string) (*User, error) {
    span, ctx := opentracing.StartSpanFromContext(ctx, "getUser")
    defer span.Finish()
    
    span.SetTag("user.id", userID)
    
    // Database query
    span.SetTag("db.statement", "SELECT * FROM users WHERE id = ?")
    user, err := db.GetUser(ctx, userID)
    if err != nil {
        span.SetTag("error", true)
        span.LogFields(log.Error(err))
        return nil, err
    }
    
    span.SetTag("user.found", user != nil)
    return user, nil
}
```

## 🔐 Security Configuration

### Network Policies

```yaml
# k8s-manifests/security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: user-service-netpol
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      app: user-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### RBAC Configuration

```yaml
# k8s-manifests/security/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: microservices
  name: user-service-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: user-service-binding
  namespace: microservices
subjects:
- kind: ServiceAccount
  name: user-service
  namespace: microservices
roleRef:
  kind: Role
  name: user-service-role
  apiGroup: rbac.authorization.k8s.io
```

## 🚀 CI/CD with ArgoCD

### Application Definition

```yaml
# argocd-applications/user-service.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: user-service
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/microservices-stack
    targetRevision: HEAD
    path: applications/user-service/k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: microservices
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - PruneLast=true
```

### GitOps Workflow

```bash
# 1. Make changes to application code
git add .
git commit -m "feat: add new user endpoint"
git push origin main

# 2. ArgoCD automatically detects changes and syncs
# 3. Monitor deployment in ArgoCD UI
argocd app get user-service

# 4. Check deployment status
kubectl get pods -n microservices -l app=user-service
```

## 📚 Learning Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Istio Documentation](https://istio.io/latest/docs/)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)

## 🔗 Upstream Source

- **Repository**: [kubernetes/examples](https://github.com/kubernetes/examples)
- **Istio**: [istio/istio](https://github.com/istio/istio)
- **ArgoCD**: [argoproj/argo-cd](https://github.com/argoproj/argo-cd)
- **License**: Apache-2.0
