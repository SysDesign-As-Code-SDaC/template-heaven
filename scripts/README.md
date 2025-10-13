# Scripts Directory

This directory contains utility scripts for managing the Organization Universal Template Repository.

## Available Scripts

### Template Synchronization

#### `sync_template.sh` (Bash/Linux/macOS)
Bash script for syncing templates from upstream repositories.

**Usage:**
```bash
./scripts/sync_template.sh [OPTIONS] TEMPLATE_NAME UPSTREAM_URL [TARGET_CATEGORY]
```

**Examples:**
```bash
# Sync T3 Stack template
./scripts/sync_template.sh t3-stack https://github.com/t3-oss/create-t3-app

# Sync specific subdirectory
./scripts/sync_template.sh nextjs-app https://github.com/vercel/next.js --subdir examples/hello-world

# Force overwrite existing template
./scripts/sync_template.sh express-api https://github.com/sahat/hackathon-starter backend --force
```

#### `sync_template.ps1` (PowerShell/Windows)
PowerShell script for syncing templates from upstream repositories.

**Usage:**
```powershell
.\scripts\sync_template.ps1 [OPTIONS] TEMPLATE_NAME UPSTREAM_URL [TARGET_CATEGORY]
```

**Examples:**
```powershell
# Sync T3 Stack template
.\scripts\sync_template.ps1 t3-stack https://github.com/t3-oss/create-t3-app

# Sync specific subdirectory
.\scripts\sync_template.ps1 nextjs-app https://github.com/vercel/next.js -Subdir examples/hello-world

# Force overwrite existing template
.\scripts\sync_template.ps1 express-api https://github.com/sahat/hackathon-starter backend -Force
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

## Template Categories

The scripts automatically detect the appropriate category based on the upstream URL or template name. Available categories:

- **fullstack**: Complete application stacks (frontend + backend)
- **frontend**: Frontend-only frameworks and libraries
- **backend**: Backend services, APIs, and server applications
- **ai-ml**: Machine learning, data science, and AI frameworks
- **mobile**: Mobile and desktop application frameworks
- **devops**: CI/CD, infrastructure, and deployment tools
- **vscode-extensions**: VSCode extension development templates
- **docs**: Documentation and community templates
- **other**: Specialized templates (monorepo, microservices, etc.)

## How It Works

1. **Validation**: The script validates the template name and upstream URL
2. **Category Detection**: Automatically detects the appropriate category or uses the provided one
3. **Repository Cloning**: Clones the upstream repository to a temporary directory
4. **File Copying**: Copies the relevant files to the target template directory
5. **Metadata Creation**: Creates a `.upstream-info` file with sync metadata
6. **Cleanup**: Removes temporary files and directories

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
