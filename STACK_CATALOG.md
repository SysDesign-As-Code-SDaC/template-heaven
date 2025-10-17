# 📚 Template-Heaven Stack Catalog

This catalog provides an overview of all available technology stacks in the template-heaven multi-branch architecture.

## 🌿 Architecture Overview

Template-heaven uses a **multi-branch architecture** where each technology stack has its own dedicated branch. This provides better organization, parallel development, and cleaner history.

- **Main/Dev Branch**: Core infrastructure, documentation, and tools (you are here)
- **Stack Branches**: Dedicated branches for each technology stack
- **Automated Workflows**: Daily trend detection and upstream syncing

## 📂 Available Stack Branches

### 🚀 Core Development Stacks

| Stack | Branch | Description | Templates |
|-------|--------|-------------|-----------|
| **Fullstack** | [`stack/fullstack`](../../tree/stack/fullstack) | Full-stack applications | Next.js, T3 Stack, Remix |
| **Frontend** | [`stack/frontend`](../../tree/stack/frontend) | Frontend frameworks | React, Vue, Svelte, Vite |
| **Backend** | [`stack/backend`](../../tree/stack/backend) | Backend services | Express, FastAPI, Django, Go |
| **Mobile** | [`stack/mobile`](../../tree/stack/mobile) | Mobile development | React Native, Flutter, Electron |

### 🤖 AI/ML Stacks

| Stack | Branch | Description | Templates |
|-------|--------|-------------|-----------|
| **AI/ML** | [`stack/ai-ml`](../../tree/stack/ai-ml) | Traditional ML and data science | MCP SDK, AI Benchmarks, Quantum ML |
| **Advanced AI** | [`stack/advanced-ai`](../../tree/stack/advanced-ai) | LLMs, RAG, vector databases | LangChain, LlamaIndex, ChromaDB |
| **Agentic AI** | [`stack/agentic-ai`](../../tree/stack/agentic-ai) | Autonomous systems and agents | LangGraph, CrewAI, AutoGen |
| **Generative AI** | [`stack/generative-ai`](../../tree/stack/generative-ai) | Content creation and generation | DALL-E, GPT, Stable Diffusion |

### 🏗️ Infrastructure Stacks

| Stack | Branch | Description | Templates |
|-------|--------|-------------|-----------|
| **DevOps** | [`stack/devops`](../../tree/stack/devops) | CI/CD, infrastructure, Docker, K8s | GitHub Actions, Terraform, Helm |
| **Microservices** | [`stack/microservices`](../../tree/stack/microservices) | Microservices architecture | Kubernetes, Istio, Event-driven |
| **Monorepo** | [`stack/monorepo`](../../tree/stack/monorepo) | Monorepo build systems | Turborepo, Nx, pnpm workspaces |
| **Serverless** | [`stack/serverless`](../../tree/stack/serverless) | Serverless and edge computing | Vercel, Cloudflare Workers, AWS Lambda |

### 🌐 Specialized Stacks

| Stack | Branch | Description | Templates |
|-------|--------|-------------|-----------|
| **Web3** | [`stack/web3`](../../tree/stack/web3) | Blockchain and smart contracts | Hardhat, Foundry, Solidity |
| **Quantum Computing** | [`stack/quantum-computing`](../../tree/stack/quantum-computing) | Quantum frameworks | Qiskit, Cirq, PennyLane |
| **Computational Biology** | [`stack/computational-biology`](../../tree/stack/computational-biology) | Bioinformatics pipelines | BWA, GATK, Biopython |
| **Scientific Computing** | [`stack/scientific-computing`](../../tree/stack/scientific-computing) | HPC, CUDA, molecular dynamics | LAMMPS, GROMACS, OpenFOAM |

### 🚀 Emerging Technology Stacks

| Stack | Branch | Description | Templates |
|-------|--------|-------------|-----------|
| **Space Technologies** | [`stack/space-technologies`](../../tree/stack/space-technologies) | Satellite systems, orbital computing | Satellite constellations, Ground stations |
| **6G Wireless** | [`stack/6g-wireless`](../../tree/stack/6g-wireless) | Next-gen communication | Ultra-low latency, Holographic communication |
| **Structural Batteries** | [`stack/structural-batteries`](../../tree/stack/structural-batteries) | Energy storage integration | Carbon fiber composites, Lightweight materials |
| **Polyfunctional Robots** | [`stack/polyfunctional-robots`](../../tree/stack/polyfunctional-robots) | Multi-task robotic systems | Task switching, Adaptive systems |

### 🛠️ Development Tools

| Stack | Branch | Description | Templates |
|-------|--------|-------------|-----------|
| **Modern Languages** | [`stack/modern-languages`](../../tree/stack/modern-languages) | Rust, Zig, Mojo, Julia | Systems programming, AI acceleration |
| **VSCode Extensions** | [`stack/vscode-extensions`](../../tree/stack/vscode-extensions) | VSCode extension development | TypeScript, JavaScript, Custom extensions |
| **Documentation** | [`stack/docs`](../../tree/stack/docs) | Documentation templates | README, CONTRIBUTING, Best practices |
| **Workflows** | [`stack/workflows`](../../tree/stack/workflows) | General workflows, software engineering best practices | GitHub Actions, CI/CD, Best practices |

## 🔄 How to Use Stack Branches

### For Template Users

1. **Navigate to Stack Branch**
   ```bash
   git checkout stack/frontend
   ```

2. **Browse Templates**
   ```bash
   ls stacks/frontend/
   cat stacks/frontend/react-vite/README.md
   ```

3. **Use Template**
   ```bash
   cp -r stacks/frontend/react-vite ../my-new-project
   cd ../my-new-project
   npm install && npm run dev
   ```

### For Contributors

1. **Choose Your Stack**
   ```bash
   git checkout stack/your-stack
   ```

2. **Add Templates**
   ```bash
   ./scripts/sync_to_branch.sh template-name upstream-url your-stack
   ```

3. **Create PR**
   ```bash
   git push origin stack/your-stack
   # Create PR to merge into dev
   ```

### For Maintainers

1. **Monitor Trends**: Review daily trend detection reports
2. **Sync Upstream**: Automated workflows keep templates updated
3. **Review PRs**: Approve template additions and updates
4. **Manage Stacks**: Use provided scripts for stack management

## 🤖 Automated Features

### Daily Trend Detection
- Monitors GitHub for trending repositories
- Analyzes repository metrics and growth patterns
- Creates issues for high-priority templates
- Auto-generates PRs for approved templates

### Upstream Syncing
- Checks upstream repositories for updates
- Creates PRs to relevant stack branches
- Maintains template freshness and security

### Branch Synchronization
- Syncs core tools to all stack branches
- Propagates documentation updates
- Maintains configuration consistency

## 📊 Stack Statistics

| Metric | Value |
|--------|-------|
| **Total Stacks** | 24 |
| **Active Branches** | 3 (fullstack, frontend, workflows) |
| **Templates Available** | 50+ |
| **Automated Workflows** | 3 (sync, trend detection, branch sync) |
| **Documentation Pages** | 4 comprehensive guides |

## 🔗 Related Resources

- **[Branch Strategy](./docs/BRANCH_STRATEGY.md)** - Complete architecture overview
- **[Stack Branch Guide](./docs/STACK_BRANCH_GUIDE.md)** - Detailed usage guide
- **[Trend Detection Integration](./docs/TREND_DETECTION_INTEGRATION.md)** - Automated discovery
- **[Contributing to Stacks](./docs/CONTRIBUTING_TO_STACKS.md)** - Contribution guidelines
- **[Main README](./README.md)** - Quick start and navigation

## 📈 Roadmap

### Phase 1: Core Stacks (Completed)
- ✅ Fullstack, Frontend, Workflows branches created
- ✅ Core infrastructure and documentation
- ✅ Automated workflows configured

### Phase 2: Expansion (In Progress)
- 🔄 Create remaining 21 stack branches
- 🔄 Add templates to each stack
- 🔄 Configure stack-specific settings

### Phase 3: Optimization (Planned)
- ⏳ Advanced trend detection algorithms
- ⏳ Cross-stack template relationships
- ⏳ Performance monitoring and optimization

---

**Last Updated**: 2024-01-15  
**Version**: 1.0  
**Maintainer**: Template Team

*This catalog is automatically updated when new stacks are added or existing stacks are modified.*
