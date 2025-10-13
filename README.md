# 🌐 Organization Universal Template Repository

Welcome! This is your all-in-one **private** repository for production-ready, best-practice templates across all major modern software development stacks and workflows. Use this as a starting point for any project within your organization—whether web, backend, DevOps, AI/ML, data, mobile, or VSCode extension development.

## 🌿 Multi-Branch Architecture

This repository uses a **hybrid multi-branch architecture** to organize templates by technology stack:

- **`dev` branch**: Core infrastructure, documentation, scripts, and tools (you are here)
- **Stack branches**: Dedicated branches for each technology stack (e.g., `stack/frontend`, `stack/backend`)
- **Automated workflows**: Daily trend detection and upstream syncing

### 📂 Browse Stack Branches

| Stack | Branch | Description |
|-------|--------|-------------|
| **Fullstack** | [`stack/fullstack`](../../tree/stack/fullstack) | Full-stack applications (Next.js, T3 Stack, Remix) |
| **Frontend** | [`stack/frontend`](../../tree/stack/frontend) | Frontend frameworks (React, Vue, Svelte) |
| **Backend** | [`stack/backend`](../../tree/stack/backend) | Backend services (Express, FastAPI, Django) |
| **AI/ML** | [`stack/ai-ml`](../../tree/stack/ai-ml) | Traditional ML and data science |
| **Advanced AI** | [`stack/advanced-ai`](../../tree/stack/advanced-ai) | LLMs, RAG, vector databases |
| **Agentic AI** | [`stack/agentic-ai`](../../tree/stack/agentic-ai) | Autonomous systems and agents |
| **Mobile** | [`stack/mobile`](../../tree/stack/mobile) | Mobile and desktop applications |
| **DevOps** | [`stack/devops`](../../tree/stack/devops) | CI/CD, infrastructure, Docker, K8s |
| **Web3** | [`stack/web3`](../../tree/stack/web3) | Blockchain and smart contracts |
| **Microservices** | [`stack/microservices`](../../tree/stack/microservices) | Microservices architecture |
| **Monorepo** | [`stack/monorepo`](../../tree/stack/monorepo) | Monorepo build systems |
| **Quantum Computing** | [`stack/quantum-computing`](../../tree/stack/quantum-computing) | Quantum frameworks |
| **Computational Biology** | [`stack/computational-biology`](../../tree/stack/computational-biology) | Bioinformatics pipelines |
| **Scientific Computing** | [`stack/scientific-computing`](../../tree/stack/scientific-computing) | HPC, CUDA, molecular dynamics |
| **Space Technologies** | [`stack/space-technologies`](../../tree/stack/space-technologies) | Satellite systems, orbital computing |
| **6G Wireless** | [`stack/6g-wireless`](../../tree/stack/6g-wireless) | Next-gen communication |
| **Structural Batteries** | [`stack/structural-batteries`](../../tree/stack/structural-batteries) | Energy storage integration |
| **Polyfunctional Robots** | [`stack/polyfunctional-robots`](../../tree/stack/polyfunctional-robots) | Multi-task robotic systems |
| **Generative AI** | [`stack/generative-ai`](../../tree/stack/generative-ai) | Content creation and generation |
| **Modern Languages** | [`stack/modern-languages`](../../tree/stack/modern-languages) | Rust, Zig, Mojo, Julia |
| **Serverless** | [`stack/serverless`](../../tree/stack/serverless) | Serverless and edge computing |
| **VSCode Extensions** | [`stack/vscode-extensions`](../../tree/stack/vscode-extensions) | VSCode extension development |
| **Documentation** | [`stack/docs`](../../tree/stack/docs) | Documentation templates |
| **Workflows** | [`stack/workflows`](../../tree/stack/workflows) | General workflows, software engineering best practices |

### 📖 Architecture Documentation

- **[Branch Strategy](./docs/BRANCH_STRATEGY.md)** - Complete architecture overview
- **[Stack Branch Guide](./docs/STACK_BRANCH_GUIDE.md)** - How to work with stack branches
- **[Trend Detection Integration](./docs/TREND_DETECTION_INTEGRATION.md)** - Automated template discovery
- **[Contributing to Stacks](./docs/CONTRIBUTING_TO_STACKS.md)** - Contribution guidelines

---

## 🛡️ Privacy and Bootstrapping Instructions

This repository is **private** and intended for internal use only.  
**Do not fork public upstream templates directly.**  
Instead, use the instructions below to pull in, update, and manage the latest best practices from the open source community without exposing your organization’s work.

### How to Bootstrap and Update Templates

1. **Cloning or Updating a Template from Upstream (No Fork)**
   - Add the original template repo as a remote:
     ```sh
     git remote add upstream https://github.com/original-owner/original-template.git
     git fetch upstream
     git checkout upstream/main -- path/to/template
     ```
   - Or use the provided `scripts/sync_template.sh` to automate this task.

2. **Regularly Sync with Upstream**
   - To update a template, simply repeat the above process or run the sync script.

3. **Bootstrapping a New Project**
   - Browse to the desired template in `/stacks/`
   - Copy or scaffold it into your new project with included scripts or manually
   - Customize as needed and keep your project private

4. **Trending Template Detection**
   - The `trending-flagger` tool is available for automatically detecting trending templates.
   - For detailed instructions on how to set up and run the tool, please refer to the `README.md` file in the `tools/trending-flagger/trend-detector` directory.

5. **Template Provenance**
   - Each template’s README notes its original upstream source.
   - Respect upstream licenses and attributions in each template subfolder.

---

## 📚 Structure Overview

### 🚀 **Core Development Stacks**
- **Fullstack**
  - Node.js + React (Vite, Next.js, T3 Stack)
  - Python (Django/Flask) + Vue, FastAPI + React
  - Go + Svelte, Rust + Yew
  - Remix, Nuxt.js
- **Frontend**
  - React (Vite, CRA, Next.js, Remix)
  - Vue (Vite, Nuxt)
  - SvelteKit, Angular, Astro, Qwik, SolidJS, Lit
- **Backend / API**
  - Node.js (Express, Fastify, NestJS)
  - Python (Flask, FastAPI, Django REST, Starlette)
  - Go (Gin, Fiber), Rust (Rocket, Axum), Java (Spring Boot, Quarkus)
  - GraphQL (Apollo, Hasura)

### 🏗️ **Advanced Architecture & Enterprise**
- **Monorepo & Build Systems**
  - Turborepo, Nx, pnpm workspaces
  - Lerna, Rush, Bazel
- **Microservices & Enterprise**
  - Kubernetes, Helm, ArgoCD, Istio
  - Event-Driven Architecture (Kafka, NATS)
  - Service Mesh, API Gateway (Kong, Ambassador, Traefik)
  - Event Sourcing, CQRS patterns

### 🤖 **AI/ML & Data Science (2025 Ready)**
- **Traditional ML**
  - Cookiecutter Data Science, MLOps (Azure ML, MLflow, Kedro)
  - Deep Learning (PyTorch Lightning + Hydra, TensorFlow, JAX)
  - Experiment Tracking (Weights & Biases, Sacred, Neptune)
- **Advanced AI & LLMs**
  - LLM/GenAI (LangChain, LlamaIndex, Haystack, HuggingFace)
  - RAG Applications, Vector Databases (Pinecone, Weaviate, Qdrant)
  - AI Agent Systems (LangGraph, CrewAI, AutoGen)
  - Multi-modal AI, Computer Vision, Federated Learning
  - Explainable AI (SHAP, LIME, Captum)
- **Data Engineering**
  - Notebook Workflows (Jupyter, Papermill, nbdev)
  - Data Versioning (DVC, LakeFS)
  - Data Engineering (Airflow, dbt, Dagster)
  - Data Warehousing (Snowflake, BigQuery, Redshift)

### ⚛️ **Quantum Computing & Advanced Physics**
- **Quantum Computing**
  - IBM Qiskit, Google Cirq, PennyLane
  - Quantum algorithms (Shor's, Grover's, VQE, QAOA)
  - Quantum machine learning and neural networks
  - Quantum cryptography (BB84, E91 protocols)
- **Scientific Computing & HPC**
  - CUDA programming, GPU acceleration
  - High-performance computing clusters
  - Molecular dynamics simulations (LAMMPS, GROMACS)
  - Parallel computing (MPI, OpenMP)
- **Physics Simulation**
  - Computational fluid dynamics (OpenFOAM, ANSYS)
  - Finite element analysis (FEniCS, deal.II)
  - Particle physics simulation (ROOT, Geant4)
  - Astrophysical modeling and simulation

### 🧬 **Computational Biology & Life Sciences**
- **Bioinformatics & Genomics**
  - Genomic data analysis pipelines
  - RNA-seq, ChIP-seq, variant calling
  - Machine learning for genomics
  - Large-scale genomics analysis
- **Protein Science**
  - Protein structure prediction (AlphaFold)
  - Molecular dynamics simulations
  - Drug discovery and design
  - Structural biology tools
- **Quantum Chemistry & Materials**
  - Density functional theory (DFT)
  - Ab initio calculations
  - Materials discovery and screening
  - Quantum chemistry simulations

### 🌐 **Web3 & Blockchain**
- **Smart Contract Development**
  - Hardhat, Foundry, Truffle
  - Solidity, Vyper, Move
- **DApp Development**
  - Full-stack Web3 applications
  - DeFi protocols, NFT marketplaces
  - Cross-chain applications

### ☁️ **Cloud & Infrastructure**
- **DevOps / CI-CD / Infrastructure**
  - GitHub Actions, GitLab CI, CircleCI, Jenkins Pipelines
  - Docker & Compose, Podman
  - Terraform, Pulumi, Ansible, Helm, AWS CDK
  - Kubernetes Manifests, Kustomize, ArgoCD
- **Serverless & Edge Computing**
  - AWS Lambda, Vercel Functions, Netlify Functions
  - Cloudflare Workers, Edge Runtime
  - WebAssembly (WASM), Deno
- **Observability & Monitoring**
  - Prometheus, Grafana, ELK, Jaeger
  - OpenTelemetry, Distributed Tracing
  - APM, Log Aggregation, Alerting

### 📱 **Mobile & Desktop**
- **Mobile Development**
  - React Native, Expo, Flutter
  - Kotlin Multiplatform, Swift
- **Desktop Applications**
  - Electron, Tauri, Capacitor
  - Native desktop frameworks

### 🎮 **Game Development**
- **Game Engines**
  - Unity, Unreal Engine, Godot
  - Phaser, Three.js, Babylon.js
- **AR/VR Development**
  - Unity XR, WebXR, ARCore, ARKit

### 🌱 **Green Computing & Sustainability**
- **Energy-Efficient Computing**
  - Low-power algorithms and systems
  - ARM, RISC-V architectures
  - Carbon footprint monitoring
- **Sustainable AI**
  - Model compression and quantization
  - Energy-efficient training
  - Green software development

### 🏗️ **Platform Engineering & DevSecOps**
- **Internal Developer Platforms**
  - Backstage, DevSpace, Tilt
  - Self-service developer tools
  - Developer experience optimization
- **Advanced Security**
  - DevSecOps integration
  - Zero trust architecture
  - Post-quantum cryptography
  - AI-powered threat detection

### 🚀 **Emerging Technologies**
- **Extended Reality (XR)**
  - AR/VR development (Unity XR, WebXR)
  - Mixed reality applications
  - Spatial computing
- **Brain-Computer Interfaces**
  - Neural interface development
  - EEG analysis and processing
  - BCI applications
- **Neuromorphic Computing**
  - Brain-inspired computing
  - SpiNNaker, Loihi processors
  - Event-driven architectures

### 🔧 **Development Tools**
- **VSCode Extensions**
  - TypeScript Starter, JavaScript Starter
  - Custom extension development
- **IoT & Edge Computing**
  - Arduino, Raspberry Pi, MQTT
  - ROS, Balena, Zephyr
  - Edge AI, Computer Vision

### 📚 **Documentation & Community**
- **Documentation**
  - Best README template, CONTRIBUTING.md
  - Docs (Docusaurus, MkDocs, Sphinx, Jupyter Book)
  - ADRs (Architectural Decision Records)
- **Project Management**
  - Agile, Scrum, Kanban templates
  - Issue/PR templates, Project roadmaps

---

## 🚀 Quickstart

### For Template Users

1. **Browse Stack Branches**: Navigate to the relevant stack branch (e.g., `stack/frontend`)
   ```bash
   # Switch to a stack branch
   git checkout stack/frontend
   
   # Or view online
   # Visit https://github.com/your-org/template-heaven/tree/stack/frontend
   ```

2. **Explore Templates**: Browse available templates in the stack
   ```bash
   # List templates in the current stack
   ls stacks/frontend/
   
   # Read template documentation
   cat stacks/frontend/react-vite/README.md
   ```

3. **Use a Template**: Copy the template to your new project
   ```bash
   # Copy template to new project
   cp -r stacks/frontend/react-vite ../my-new-project
   cd ../my-new-project
   
   # Follow template setup instructions
   npm install
   npm run dev
   ```

### For Contributors

1. **Choose Your Stack**: Identify which stack you're contributing to
2. **Switch to Stack Branch**: `git checkout stack/your-stack`
3. **Make Changes**: Add or update templates
4. **Create PR**: Submit changes for review
5. **Read Guidelines**: See [Contributing to Stacks](./docs/CONTRIBUTING_TO_STACKS.md)

### For Maintainers

1. **Monitor Trends**: Review daily trend detection reports (automated)
2. **Sync Upstream**: Automated workflows keep templates up-to-date
3. **Review PRs**: Approve template additions and updates
4. **Manage Stacks**: Use provided scripts for stack management

### Automated Features

- **Daily Trend Detection**: Automatically discovers trending templates
- **Upstream Syncing**: Keeps templates updated with upstream changes
- **Branch Synchronization**: Propagates core tools to all stack branches
- **Automated PRs**: Creates pull requests for high-priority templates

---

## 🏗️ Gold-Standard Templates & Upstream References

### Fullstack
- [T3 Stack (Next.js/TypeScript/tRPC/Prisma)](https://github.com/t3-oss/create-t3-app)
- [Next.js Examples](https://github.com/vercel/next.js/tree/canary/examples)
- [Remix Indie Stack](https://github.com/remix-run/indie-stack)
- [Django + Vue](https://github.com/gtalarico/django-vue-template)
- [FastAPI + React](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [Go + Svelte](https://github.com/janpfeifer/go-sveltekit-template)
- [Rust + Yew](https://github.com/jetli/create-yew-app)

### Frontend
- [Vite React Template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react)
- [Vue Vite Template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-vue)
- [SvelteKit](https://github.com/sveltejs/kit)
- [Astro](https://github.com/withastro/astro)
- [Qwik](https://github.com/BuilderIO/qwik)
- [SolidJS](https://github.com/solidjs/templates)
- [Lit](https://github.com/lit/lit/tree/main/packages/create-lit-app)

### Backend / API
- [Express API Starter](https://github.com/sahat/hackathon-starter)
- [NestJS Starter](https://github.com/nestjs/typescript-starter)
- [FastAPI Template](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [Django REST Starter](https://github.com/encode/django-rest-framework)
- [Go Fiber Starter](https://github.com/gofiber/boilerplate)
- [Rust Rocket Starter](https://github.com/SergioBenitez/Rocket/tree/master/examples)
- [Spring Boot Starter](https://github.com/spring-projects/spring-boot/tree/main/spring-boot-samples)
- [GraphQL Node Starter](https://github.com/graphql-boilerplates/node-graphql-server)

### AI / ML / Data Science
- [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science)
- [Kedro Template](https://github.com/kedro-org/kedro-template)
- [MLflow](https://github.com/mlflow/mlflow)
- [PyTorch Lightning + Hydra](https://github.com/ashleve/lightning-hydra-template)
- [Azure ML MLOps](https://github.com/Azure/mlops)
- [Kubeflow Pipelines](https://github.com/kubeflow/pipelines)
- [Metaflow](https://github.com/Netflix/metaflow)
- [Flyte](https://github.com/flyteorg/flyte)
- [Weights & Biases Examples](https://github.com/wandb/examples)
- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [Haystack LLM](https://github.com/deepset-ai/haystack)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Jupyter Nbdev](https://github.com/fastai/nbdev)
- [DVC Data Versioning](https://github.com/iterative/dvc)
- [LakeFS](https://github.com/treeverse/lakeFS)

### Mobile / Desktop
- [React Native Template](https://github.com/react-native-community/react-native-template-typescript)
- [Expo Starter](https://github.com/expo/examples)
- [Flutter Starter](https://github.com/flutter/samples)
- [Electron Forge](https://github.com/electron/forge-template)
- [Tauri App](https://github.com/tauri-apps/tauri)

### DevOps / Infrastructure
- [actions/starter-workflows](https://github.com/actions/starter-workflows)
- [GitLab CI Templates](https://gitlab.com/gitlab-org/gitlab/-/tree/master/lib/gitlab/ci/templates)
- [Docker Compose Samples](https://github.com/docker/awesome-compose)
- [Terraform AWS Modules](https://github.com/terraform-aws-modules)
- [Pulumi Examples](https://github.com/pulumi/examples)
- [Ansible Best Practices](https://github.com/bertvv/ansible-best-practices)
- [Kubernetes Manifests](https://github.com/kubernetes/examples)
- [Helm Charts](https://github.com/helm/charts)
- [Kustomize](https://github.com/kubernetes-sigs/kustomize)
- [ArgoCD](https://github.com/argoproj/argo-cd)
- [Serverless Framework](https://github.com/serverless/examples)
- [AWS SAM](https://github.com/aws/aws-sam-cli-app-templates)

### VSCode Extensions
- [VSCode Extension TypeScript Starter](https://github.com/microsoft/vscode-extension-samples)
- [Yeoman Generator for VSCode Extensions](https://github.com/microsoft/vscode-generator-code)

### Docs & Community
- [Best README Template](https://github.com/othneildrew/Best-README-Template)
- [Github Issue/PR Templates](https://github.com/stevemao/github-issue-templates)
- [Docusaurus Docs Starter](https://github.com/facebook/docusaurus)
- [MkDocs Material](https://github.com/squidfunk/mkdocs-material)
- [Sphinx Quickstart](https://github.com/sphinx-doc/sphinx)
- [Jupyter Book](https://github.com/executablebooks/jupyter-book)
- [ADR Tools](https://github.com/npryce/adr-tools)

### Monorepo & Static Sites
- [Turborepo Starter](https://github.com/vercel/turbo)
- [Nx Monorepo](https://github.com/nrwl/nx)
- [pnpm Monorepo](https://github.com/pnpm/pnpm-examples)
- [Lerna Monorepo](https://github.com/lerna/lerna)
- [Gatsby Starter](https://github.com/gatsbyjs/gatsby-starter-default)
- [Jekyll Starter](https://github.com/barryclark/jekyll-now)
- [Eleventy Starter](https://github.com/11ty/eleventy-base-blog)

### Microservices, Auth, Observability, Serverless, etc.
- [Microservices Patterns (Go)](https://github.com/microservices-demo/microservices-demo)
- [Kong Gateway](https://github.com/Kong/kong)
- [Keycloak](https://github.com/keycloak/keycloak)
- [Prometheus Monitoring](https://github.com/prometheus/prometheus)
- [Grafana Dashboards](https://github.com/grafana/grafana)
- [ELK Stack](https://github.com/deviantony/docker-elk)
- [Jaeger Tracing](https://github.com/jaegertracing/jaeger)
- [Serverless Examples](https://github.com/serverless/examples)

### Data Engineering & Warehousing
- [Airflow Example DAGs](https://github.com/apache/airflow/tree/main/airflow/example_dags)
- [dbt Starter](https://github.com/dbt-labs/dbt-starter-project)
- [Dagster Example Pipelines](https://github.com/dagster-io/dagster/tree/master/examples)
- [Great Expectations](https://github.com/great-expectations/great_expectations)

---

## 🛠️ Using Templates

- **Copy** the desired folder from `stacks/` (or use the script).
- **Customize** your stack as needed.
- **Check** each stack README for setup, testing, CI, and deployment instructions.

---

## 🤝 Contributing

Want to add a new stack or improve a template? See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines, review process, and contact info.

---

## ⭐ Credits

Many templates here are forks or adaptations of open-source community best practices.  
Please see individual template folders for original authors and license notes.

---

Happy building!
