# 🏛️ Template Heaven

> Interactive system design creation tool powered by LLMs — Generate architecture diagrams, specs, and code from natural language

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Part of](https://img.shields.io/badge/Part_of-SysDesign--As--Code-purple)](https://github.com/SysDesign-As-Code-SDaC)

---

## 🎯 Vision

Turn natural language descriptions into production-ready system designs:

```
"Build a microservice that handles user authentication with JWT, 
stores sessions in Redis, and integrates with OAuth providers"
```

**↓ Generates ↓**

- Architecture diagram (Mermaid/PlantUML)
- OpenAPI specification
- Database schema
- Docker Compose configuration
- Code scaffolding

---

## 🚀 Features

### 📐 Design Templates

| Template | Description |
|----------|-------------|
| `microservice` | REST API with DB, caching, auth |
| `event-driven` | Message queues, event sourcing |
| `monolith` | Traditional layered architecture |
| `serverless` | AWS Lambda / GCP Functions |
| `multi-agent` | AI agent orchestration patterns |

### 🤖 LLM Integration

- **Interactive prompting** — Refine designs through conversation
- **Context-aware** — Understands existing codebase patterns
- **Multi-model** — Works with OpenAI, Anthropic, local models

### 📝 Output Formats

- **Diagrams**: Mermaid, PlantUML, D2
- **Specs**: OpenAPI 3.0, AsyncAPI, GraphQL SDL
- **Code**: Python, TypeScript, Go scaffolding
- **Infra**: Docker, Kubernetes, Terraform

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/SysDesign-As-Code-SDaC/template-heaven.git
cd template-heaven

# Install dependencies
pip install -e .

# Or with npm for web interface
npm install && npm run dev
```

---

## 🔧 Usage

### CLI Mode

```bash
# Generate from prompt
template-heaven generate "user authentication service with OAuth2"

# From existing spec
template-heaven scaffold --input openapi.yaml --output ./src

# Interactive mode
template-heaven interactive
```

### Python API

```python
from template_heaven import Designer

designer = Designer(model="gpt-4")
design = designer.generate(
    prompt="Build a payment processing service",
    templates=["microservice", "event-driven"],
    output_formats=["mermaid", "openapi", "dockerfile"]
)

print(design.diagram)
design.save("./output")
```

---

## 🗂️ Template Structure

```
templates/
├── microservice/
│   ├── manifest.yaml      # Template metadata
│   ├── prompts/           # LLM prompt templates
│   ├── schemas/           # JSON Schema definitions
│   └── scaffolds/         # Code generation templates
├── event-driven/
├── multi-agent/
└── ...
```

---

## 🔗 Integration with SDaC Ecosystem

- **[aiagentsuite](https://github.com/jimmyjdejesus-cmyk/aiagentsuite)** — Use VDE protocols for design workflows
- **openspec** — Generate OpenAPI specs (coming soon)
- **GitHub Actions** — Automated design validation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-template`)
3. Follow VDE methodology
4. Submit a pull request

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<div align="center">

**Part of the [SysDesign-As-Code](https://github.com/SysDesign-As-Code-SDaC) ecosystem**

</div>
