#!/bin/bash

# create_stack_branch.sh - Create a new stack branch with proper structure and configuration
# Usage: ./scripts/create_stack_branch.sh <stack-name> [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STACK_NAME=""
DESCRIPTION=""
MAINTAINERS=""
FORCE=false
DRY_RUN=false
VERBOSE=false

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 <stack-name> [options]

Create a new stack branch with proper structure and configuration.

Arguments:
    stack-name          Name of the stack to create (e.g., frontend, backend, ai-ml)

Options:
    -d, --description   Description of the stack
    -m, --maintainers   Comma-separated list of maintainers (e.g., "@team1,@team2")
    -f, --force         Force creation even if branch exists
    --dry-run           Show what would be done without making changes
    -v, --verbose       Enable verbose output
    -h, --help          Show this help message

Examples:
    $0 frontend --description "Frontend framework templates"
    $0 backend -d "Backend service templates" -m "@backend-team"
    $0 ai-ml --dry-run

Available stack categories:
    Core Development: fullstack, frontend, backend, mobile
    AI/ML: ai-ml, advanced-ai, agentic-ai, generative-ai
    Infrastructure: devops, microservices, monorepo, serverless
    Specialized: web3, quantum-computing, computational-biology, scientific-computing
    Emerging: space-technologies, 6g-wireless, structural-batteries, polyfunctional-robots
    Tools: modern-languages, vscode-extensions, docs, workflows
EOF
}

# Function to validate stack name
validate_stack_name() {
    local stack=$1
    
    # Check if stack name is provided
    if [[ -z "$stack" ]]; then
        print_status $RED "Error: Stack name is required"
        show_usage
        exit 1
    fi
    
    # Check if stack name contains valid characters
    if [[ ! "$stack" =~ ^[a-z0-9-]+$ ]]; then
        print_status $RED "Error: Stack name must contain only lowercase letters, numbers, and hyphens"
        exit 1
    fi
    
    # Check if stack name is too long
    if [[ ${#stack} -gt 50 ]]; then
        print_status $RED "Error: Stack name is too long (max 50 characters)"
        exit 1
    fi
}

# Function to check if branch exists
check_branch_exists() {
    local branch_name="stack/$STACK_NAME"
    
    if git show-ref --verify --quiet "refs/heads/$branch_name" 2>/dev/null; then
        return 0  # Branch exists locally
    fi
    
    if git show-ref --verify --quiet "refs/remotes/origin/$branch_name" 2>/dev/null; then
        return 0  # Branch exists remotely
    fi
    
    return 1  # Branch doesn't exist
}

# Function to create stack configuration
create_stack_config() {
    local config_file="stacks/$STACK_NAME/.stack-config.yml"
    
    cat > "$config_file" << EOF
stack_name: "$(echo $STACK_NAME | sed 's/-/ /g' | sed 's/\b\w/\U&/g')"
category: "$STACK_NAME"
description: "$DESCRIPTION"
maintainers: [$MAINTAINERS]
upstream_sources: []
trend_keywords:
  - "$STACK_NAME"
auto_sync: true
created_at: "$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
last_updated: "$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
version: "1.0.0"
EOF
    
    print_status $GREEN "Created stack configuration: $config_file"
}

# Function to create trend detection configuration
create_trend_config() {
    local config_file="stacks/$STACK_NAME/.trend-detection-config.yml"
    
    cat > "$config_file" << EOF
stack_name: "$STACK_NAME"
enabled: true
keywords:
  - "$STACK_NAME"
  - "template"
  - "boilerplate"
  - "starter"
characteristics:
  - has_readme: true
  - has_license: true
  - has_ci: true
thresholds:
  stars:
    minimum: 100
    trending: 1000
    critical: 5000
  forks:
    minimum: 10
    trending: 100
    critical: 500
  growth_rate:
    minimum: 0.1
    trending: 0.5
    critical: 1.0
auto_sync:
  enabled: true
  require_approval: true
notifications:
  enabled: true
  channels:
    - "slack"
    - "email"
  priority_threshold: 0.7
EOF
    
    print_status $GREEN "Created trend detection configuration: $config_file"
}

# Function to create stack README
create_stack_readme() {
    local readme_file="stacks/$STACK_NAME/README.md"
    
    cat > "$readme_file" << EOF
# $(echo $STACK_NAME | sed 's/-/ /g' | sed 's/\b\w/\U&/g') Stack

$DESCRIPTION

## 📋 Available Templates

This stack contains templates for $STACK_NAME development. See [TEMPLATES.md](./TEMPLATES.md) for a complete list.

## 🚀 Quick Start

1. **Browse Templates**
   \`\`\`bash
   ls stacks/$STACK_NAME/
   \`\`\`

2. **Use a Template**
   \`\`\`bash
   cp -r stacks/$STACK_NAME/template-name ../my-new-project
   cd ../my-new-project
   # Follow template-specific setup instructions
   \`\`\`

## 📚 Documentation

- [Template List](./TEMPLATES.md) - Complete list of available templates
- [Stack Configuration](./.stack-config.yml) - Stack configuration
- [Trend Detection](./.trend-detection-config.yml) - Trend detection settings

## 🤝 Contributing

To add new templates to this stack:

1. Use the sync script:
   \`\`\`bash
   ./scripts/sync_to_branch.sh template-name upstream-url $STACK_NAME
   \`\`\`

2. Or manually add templates following the [contribution guidelines](../../docs/CONTRIBUTING_TO_STACKS.md)

## 📊 Stack Statistics

- **Templates**: 0 (initial)
- **Last Updated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
- **Maintainers**: $MAINTAINERS

---

**Stack**: $STACK_NAME  
**Version**: 1.0.0  
**Last Updated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF
    
    print_status $GREEN "Created stack README: $readme_file"
}

# Function to create templates index
create_templates_index() {
    local templates_file="stacks/$STACK_NAME/TEMPLATES.md"
    
    cat > "$templates_file" << EOF
# $(echo $STACK_NAME | sed 's/-/ /g' | sed 's/\b\w/\U&/g') Templates

This document lists all available templates in the $STACK_NAME stack.

## 📋 Template List

| Template | Description | Upstream | Last Updated | Status |
|----------|-------------|----------|--------------|--------|
| *No templates yet* | *Add templates using the sync script* | - | - | - |

## 🚀 Adding Templates

To add a new template to this stack:

\`\`\`bash
# Use the automated sync script
./scripts/sync_to_branch.sh template-name upstream-url $STACK_NAME

# Example
./scripts/sync_to_branch.sh react-vite https://github.com/vitejs/vite $STACK_NAME
\`\`\`

## 📊 Template Statistics

- **Total Templates**: 0
- **Active Templates**: 0
- **Deprecated Templates**: 0
- **Last Updated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

---

**Stack**: $STACK_NAME  
**Last Updated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF
    
    print_status $GREEN "Created templates index: $templates_file"
}

# Function to create .gitkeep file
create_gitkeep() {
    local gitkeep_file="stacks/$STACK_NAME/.gitkeep"
    touch "$gitkeep_file"
    print_status $GREEN "Created .gitkeep file: $gitkeep_file"
}

# Function to create stack branch
create_stack_branch() {
    local branch_name="stack/$STACK_NAME"
    
    print_status $BLUE "Creating stack branch: $branch_name"
    
    # Check if we're on dev branch
    local current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "dev" ]]; then
        print_status $YELLOW "Warning: Not on dev branch (current: $current_branch)"
        if [[ "$FORCE" != true ]]; then
            print_status $RED "Error: Must be on dev branch to create stack branches"
            exit 1
        fi
    fi
    
    # Create and switch to new branch
    if [[ "$DRY_RUN" == true ]]; then
        print_status $YELLOW "[DRY RUN] Would create branch: $branch_name"
    else
        git checkout -b "$branch_name"
        print_status $GREEN "Created and switched to branch: $branch_name"
    fi
    
    # Create stack directory structure
    local stack_dir="stacks/$STACK_NAME"
    if [[ "$DRY_RUN" == true ]]; then
        print_status $YELLOW "[DRY RUN] Would create directory: $stack_dir"
    else
        mkdir -p "$stack_dir"
        print_status $GREEN "Created directory: $stack_dir"
    fi
    
    # Create stack files
    if [[ "$DRY_RUN" == true ]]; then
        print_status $YELLOW "[DRY RUN] Would create stack configuration files"
    else
        create_stack_config
        create_trend_config
        create_stack_readme
        create_templates_index
        create_gitkeep
    fi
    
    # Add and commit files
    if [[ "$DRY_RUN" == true ]]; then
        print_status $YELLOW "[DRY RUN] Would add and commit files"
    else
        git add "stacks/$STACK_NAME/"
        git commit -m "feat: create $STACK_NAME stack branch

- Add stack configuration and documentation
- Initialize template directory structure
- Configure trend detection settings

Stack: $STACK_NAME
Maintainers: $MAINTAINERS"
        print_status $GREEN "Committed initial stack structure"
    fi
    
    # Push branch to remote
    if [[ "$DRY_RUN" == true ]]; then
        print_status $YELLOW "[DRY RUN] Would push branch to remote"
    else
        git push -u origin "$branch_name"
        print_status $GREEN "Pushed branch to remote: origin/$branch_name"
    fi
}

# Function to show summary
show_summary() {
    print_status $GREEN "✅ Stack branch creation completed!"
    echo
    print_status $BLUE "Summary:"
    echo "  Stack Name: $STACK_NAME"
    echo "  Branch: stack/$STACK_NAME"
    echo "  Description: $DESCRIPTION"
    echo "  Maintainers: $MAINTAINERS"
    echo
    print_status $BLUE "Next Steps:"
    echo "  1. Switch to the new branch: git checkout stack/$STACK_NAME"
    echo "  2. Add templates: ./scripts/sync_to_branch.sh template-name upstream-url $STACK_NAME"
    echo "  3. Update documentation as needed"
    echo "  4. Create pull request to merge back to dev"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--description)
            DESCRIPTION="$2"
            shift 2
            ;;
        -m|--maintainers)
            MAINTAINERS="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_status $RED "Error: Unknown option $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$STACK_NAME" ]]; then
                STACK_NAME="$1"
            else
                print_status $RED "Error: Multiple stack names provided"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Set default values
if [[ -z "$DESCRIPTION" ]]; then
    DESCRIPTION="Templates for $STACK_NAME development"
fi

if [[ -z "$MAINTAINERS" ]]; then
    MAINTAINERS="\"@$STACK_NAME-team\""
fi

# Validate inputs
validate_stack_name "$STACK_NAME"

# Check if branch already exists
if check_branch_exists; then
    if [[ "$FORCE" != true ]]; then
        print_status $RED "Error: Branch stack/$STACK_NAME already exists"
        print_status $YELLOW "Use --force to overwrite or choose a different name"
        exit 1
    else
        print_status $YELLOW "Warning: Branch stack/$STACK_NAME already exists, will overwrite"
    fi
fi

# Create the stack branch
print_status $BLUE "Creating stack branch for: $STACK_NAME"
create_stack_branch

# Show summary
if [[ "$DRY_RUN" != true ]]; then
    show_summary
else
    print_status $YELLOW "Dry run completed. Use without --dry-run to create the branch."
fi
