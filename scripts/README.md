# Scripts Directory

This directory contains utility scripts for managing the Organization Universal Template Repository's multi-branch stack architecture.

## 🌿 Multi-Branch Architecture

This repository uses a hybrid multi-branch architecture where each technology stack has its own dedicated branch. The scripts automatically handle branch detection and switching.

## Available Scripts

### Template Synchronization

#### `sync_template.sh` (Bash/Linux/macOS)
Enhanced Bash script for syncing templates from upstream repositories with automatic branch detection and switching.

**Usage:**
```bash
./scripts/sync_template.sh [OPTIONS] TEMPLATE_NAME UPSTREAM_URL [TARGET_CATEGORY]
```

**Features:**
- **Automatic branch detection**: Detects the target stack branch based on template category
- **Auto-checkout**: Automatically switches to the correct stack branch
- **Branch creation**: Creates local branches from remote if needed
- **Enhanced error handling**: Better error messages and recovery

**Examples:**
```bash
# Sync T3 Stack template (auto-detects fullstack branch)
./scripts/sync_template.sh t3-stack https://github.com/t3-oss/create-t3-app

# Sync specific subdirectory
./scripts/sync_template.sh nextjs-app https://github.com/vercel/next.js --subdir examples/hello-world

# Force overwrite existing template
./scripts/sync_template.sh express-api https://github.com/sahat/hackathon-starter backend --force
```

#### `sync_template.ps1` (PowerShell/Windows)
Enhanced PowerShell script for syncing templates from upstream repositories with automatic branch detection and switching.

**Usage:**
```powershell
.\scripts\sync_template.ps1 [OPTIONS] TEMPLATE_NAME UPSTREAM_URL [TARGET_CATEGORY]
```

**Features:**
- **Automatic branch detection**: Detects the target stack branch based on template category
- **Auto-checkout**: Automatically switches to the correct stack branch
- **Branch creation**: Creates local branches from remote if needed
- **Enhanced error handling**: Better error messages and recovery

**Examples:**
```powershell
# Sync T3 Stack template (auto-detects fullstack branch)
.\scripts\sync_template.ps1 t3-stack https://github.com/t3-oss/create-t3-app

# Sync specific subdirectory
.\scripts\sync_template.ps1 nextjs-app https://github.com/vercel/next.js -Subdir examples/hello-world

# Force overwrite existing template
.\scripts\sync_template.ps1 express-api https://github.com/sahat/hackathon-starter backend -Force
```

### Branch Management

#### `create_stack_branch.sh` (Bash/Linux/macOS)
Script for creating new stack branches with proper structure and configuration.

**Usage:**
```bash
./scripts/create_stack_branch.sh <stack-name>
```

**Examples:**
```bash
# Create a new backend stack branch
./scripts/create_stack_branch.sh backend

# Create a new ai-ml stack branch
./scripts/create_stack_branch.sh ai-ml
```

#### `create_stack_branch.ps1` (PowerShell/Windows)
PowerShell script for creating new stack branches with proper structure and configuration.

**Usage:**
```powershell
.\scripts\create_stack_branch.ps1 <stack-name>
```

**Examples:**
```powershell
# Create a new backend stack branch
.\scripts\create_stack_branch.ps1 backend

# Create a new ai-ml stack branch
.\scripts\create_stack_branch.ps1 ai-ml
```

#### `branch_manager.py` (Python)
Advanced Python script for managing stack branches with comprehensive functionality.

**Usage:**
```bash
python scripts/branch_manager.py [COMMAND] [OPTIONS]
```

**Commands:**
- `list`: List all available stack branches
- `create <name>`: Create a new stack branch
- `validate <name>`: Validate a stack branch structure
- `sync <name>`: Sync core tools to a stack branch

**Examples:**
```bash
# List all stack branches
python scripts/branch_manager.py list

# Create a new stack branch
python scripts/branch_manager.py create backend

# Validate stack branch structure
python scripts/branch_manager.py validate frontend

# Sync core tools to stack branch
python scripts/branch_manager.py sync fullstack
```

## Script Options

### Common Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-f, --force` | Force overwrite existing template |
| `-d, --dry-run` | Show what would be done without making changes |
| `-v, --verbose` | Enable verbose output |
| `--branch BRANCH` | Specify upstream branch (default: main) |
| `--subdir PATH` | Sync only a subdirectory from upstream |

### PowerShell-Specific Options

| Option | Description |
|--------|-------------|
| `-Branch BRANCH` | Specify upstream branch (default: main) |
| `-Subdir PATH` | Sync only a subdirectory from upstream |
| `-Force` | Force overwrite existing template |
| `-DryRun` | Show what would be done without making changes |
| `-Verbose` | Enable verbose output |
| `-Help` | Show help message |

## Stack Categories

The scripts automatically detect the appropriate stack branch based on the upstream URL or template name. Available stack branches:

### Core Development Stacks
- **fullstack**: Complete application stacks (Next.js, T3 Stack, Remix)
- **frontend**: Frontend-only frameworks and libraries (React, Vue, Svelte)
- **backend**: Backend services, APIs, and server applications (Express, FastAPI, Django)
- **mobile**: Mobile and desktop application frameworks (React Native, Flutter, Electron)

### AI/ML Stacks
- **ai-ml**: Traditional machine learning and data science (PyTorch, TensorFlow, Scikit-learn)
- **advanced-ai**: LLMs, RAG, and vector databases (LangChain, LlamaIndex, ChromaDB)
- **agentic-ai**: Autonomous systems and agents (LangGraph, CrewAI, AutoGen)
- **generative-ai**: Content creation and generation (DALL-E, GPT, Stable Diffusion)

### Infrastructure Stacks
- **devops**: automation, infrastructure, and deployment tools (Docker, Kubernetes, Terraform)
- **microservices**: Microservices architecture (Kubernetes, Istio, Event-driven)
- **monorepo**: Monorepo build systems (Turborepo, Nx, pnpm workspaces)
- **serverless**: Serverless and edge computing (Vercel, Cloudflare Workers, AWS Lambda)

### Specialized Stacks
- **web3**: Blockchain and smart contracts (Hardhat, Foundry, Solidity)
- **quantum-computing**: Quantum frameworks (Qiskit, Cirq, PennyLane)
- **computational-biology**: Bioinformatics pipelines (BWA, GATK, Biopython)
- **scientific-computing**: HPC, CUDA, and molecular dynamics (LAMMPS, GROMACS, OpenFOAM)

### Emerging Technology Stacks
- **space-technologies**: Satellite systems and orbital computing
- **6g-wireless**: Next-generation communication systems
- **structural-batteries**: Energy storage integration
- **polyfunctional-robots**: Multi-task robotic systems

### Development Tools
- **modern-languages**: Rust, Zig, Mojo, Julia
- **vscode-extensions**: VSCode extension development
- **docs**: Documentation templates
- **workflows**: General workflows and software engineering best practices

## How It Works

### Template Synchronization Process

1. **Validation**: The script validates the template name and upstream URL
2. **Stack Detection**: Automatically detects the appropriate stack branch based on keywords
3. **Branch Management**: 
   - Checks if target stack branch exists locally
   - Creates local branch from remote if needed
   - Automatically switches to the target stack branch
4. **Repository Cloning**: Clones the upstream repository to a temporary directory
5. **File Copying**: Copies the relevant files to the target template directory
6. **Metadata Creation**: Creates a `.upstream-info` file with sync metadata
7. **Cleanup**: Removes temporary files and directories

### Branch Management Process

1. **Validation**: Validates the stack name and checks for conflicts
2. **Branch Creation**: Creates a new stack branch from the current dev branch
3. **Structure Setup**: Creates the necessary directory structure
4. **Configuration**: Adds stack-specific configuration files
5. **Documentation**: Creates initial documentation and templates
6. **Commit**: Commits the initial stack structure

## Generated Files

### `.upstream-info`
Each synced template includes a `.upstream-info` file containing:

- Upstream repository URL
- Branch used for sync
- Subdirectory (if applicable)
- Last sync timestamp
- Sync script and command used
- License and attribution information

## Best Practices

### Before Syncing
1. **Research**: Verify the upstream template is actively maintained
2. **Test**: Use `--dry-run` to preview changes
3. **Backup**: Ensure existing templates are backed up if using `--force`

### After Syncing
1. **Review**: Check all copied files for completeness
2. **Customize**: Update README.md with organization-specific information
3. **Test**: Create a test project to verify the template works
4. **Document**: Update any organization-specific documentation

### Security Considerations
1. **Review**: Always review synced code for security issues
2. **Dependencies**: Check and update dependency versions
3. **Secrets**: Remove any hardcoded secrets or API keys
4. **Licenses**: Ensure compliance with upstream licenses

## Troubleshooting

### Common Issues

**Permission Denied (Linux/macOS)**
```bash
chmod +x scripts/sync_template.sh
```

**Git Not Found**
- Ensure Git is installed and available in PATH
- On Windows, install Git for Windows or use GitHub Desktop

**Repository Not Found**
- Verify the upstream URL is correct and accessible
- Check if the repository is private and you have access

**Branch Not Found**
- Verify the branch exists in the upstream repository
- Use `--branch` to specify a different branch

**Subdirectory Not Found**
- Verify the subdirectory path is correct
- Check the repository structure using the web interface

### Getting Help

1. **Script Help**: Use `--help` or `-Help` for detailed usage information
2. **Dry Run**: Use `--dry-run` or `-DryRun` to preview operations
3. **Verbose Output**: Use `--verbose` or `-Verbose` for detailed logging
4. **Issues**: Create an issue in this repository for bugs or feature requests

## Contributing

To improve these scripts:

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly on your platform
5. **Submit** a pull request

Please ensure:
- Scripts work on both Linux/macOS (Bash) and Windows (PowerShell)
- Error handling is comprehensive
- Documentation is updated
- Tests are added for new features
