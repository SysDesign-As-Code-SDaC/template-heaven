# AI-ML Stack Templates

Traditional ML and data science workflows

## ðŸ“‚ Available Templates

This stack branch contains production-ready templates for ai-ml development.

### Template Categories

- **MCP Middleware** - Model Context Protocol servers and clients
- **AI Benchmarks** - Performance testing and evaluation suites
- **Generic AI Models** - Flexible ML model templates
- **Quantum Computing** - Quantum algorithms and ML integration
- **Voice Processing** - NVIDIA Maverick voice AI integration
- **Advanced Libraries** - Specialized AI/ML libraries and tools

## ðŸš€ Quick Start

### Using a Template

1. **Browse Templates**
   ```bash
   ls stacks/ai-ml/
   ```

2. **Copy Template to Your Project**
   ```bash
   cp -r stacks/ai-ml/your-template ../my-new-project
   cd ../my-new-project
   ```

3. **Follow Template Setup Instructions**
   Each template has its own README.md with specific setup instructions.

### Adding a New Template

Use the sync script to add a new template from upstream:

```powershell
.\scripts\sync_template.ps1 template-name https://github.com/owner/repo ai-ml
```

Or see [Contributing to Stacks](../../docs/CONTRIBUTING_TO_STACKS.md) for detailed instructions.

## ðŸ“– Documentation

- **[Main README](../../README.md)** - Repository overview
- **[Stack Catalog](../../STACK_CATALOG.md)** - All available stacks
- **[Branch Strategy](../../docs/BRANCH_STRATEGY.md)** - Architecture details
- **[Contributing Guide](../../docs/CONTRIBUTING_TO_STACKS.md)** - How to contribute

## ðŸ”„ Keeping Templates Updated

Templates in this stack are synchronized with their upstream sources regularly.

### Manual Update

To update a specific template:

```powershell
.\scripts\sync_template.ps1 template-name upstream-url ai-ml -Force
```

### Automated Updates

This repository uses automated workflows to:
- Monitor upstream repositories for updates
- Detect trending new templates
- Create pull requests for updates

## ðŸ“‹ Template Index

<!-- This section will be populated as templates are added -->

| Template | Description | Upstream | Status |
|----------|-------------|----------|--------|
| [python-mcp-sdk](mcp-middleware/python-mcp-sdk/) | Production-ready MCP server and client template | [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | ✅ Active |
| [ai-benchmarks-suite](ai-benchmarks-suite/) | Comprehensive AI performance testing suite | Custom | ✅ Active |
| [generic-ai-model](generic-ai-model/) | Flexible ML model template with multiple algorithms | Custom | ✅ Active |
| [quantum-computing-starter](quantum-computing-starter/) | Quantum algorithms and ML integration | Custom | ✅ Active |
| [nvidia-maverick-llama-voice](nvidia-maverick-llama-voice/) | Voice AI integration with NVIDIA Maverick | Custom | ✅ Active |

## ðŸ¤ Contributing

Contributions are welcome! Please see our [Contributing Guide](../../docs/CONTRIBUTING_TO_STACKS.md).

### What to Contribute

- New templates from popular upstream repositories
- Improvements to existing templates
- Documentation updates
- Bug fixes

### Contribution Process

1. Switch to this stack branch
2. Add or update templates
3. Test thoroughly
4. Submit a pull request
5. Wait for review and merge

## ðŸ“œ License

Templates in this repository maintain their original licenses. Please check each template's directory for specific license information.

---

**Last Updated**: 2025-10-13  
**Branch**: stack/ai-ml  
**Maintainer**: Template Team
