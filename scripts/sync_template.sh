#!/bin/bash

# Organization Universal Template Repository - Template Sync Script
# This script helps sync templates from upstream repositories without forking

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STACKS_DIR="$REPO_ROOT/stacks"
TEMP_DIR="/tmp/template-sync-$$"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] TEMPLATE_NAME UPSTREAM_URL [TARGET_CATEGORY]

Sync a template from an upstream repository to the local template collection.

ARGUMENTS:
    TEMPLATE_NAME     Name for the template (e.g., 't3-stack', 'nextjs-app')
    UPSTREAM_URL      Git URL of the upstream repository
    TARGET_CATEGORY   Category directory (default: auto-detect from URL)

OPTIONS:
    -h, --help        Show this help message
    -f, --force       Force overwrite existing template
    -d, --dry-run     Show what would be done without making changes
    -v, --verbose     Enable verbose output
    --branch BRANCH   Specify upstream branch (default: main/master)
    --subdir PATH     Sync only a subdirectory from upstream

EXAMPLES:
    $0 t3-stack https://github.com/t3-oss/create-t3-app
    $0 nextjs-app https://github.com/vercel/next.js --subdir examples/hello-world
    $0 express-api https://github.com/sahat/hackathon-starter backend --force

EOF
}

# Function to detect category from URL or template name
detect_category() {
    local template_name="$1"
    local upstream_url="$2"
    
    # Check if category is explicitly provided
    if [[ $# -ge 3 ]]; then
        echo "$3"
        return
    fi
    
    # Auto-detect based on URL patterns
    case "$upstream_url" in
        *"full-stack"*|*"fullstack"*|*"t3"*|*"remix"*|*"next.js"*)
            echo "fullstack"
            ;;
        *"react"*|*"vue"*|*"svelte"*|*"frontend"*|*"vite"*)
            echo "frontend"
            ;;
        *"api"*|*"backend"*|*"server"*|*"express"*|*"fastapi"*|*"django"*)
            echo "backend"
            ;;
        *"ml"*|*"ai"*|*"data-science"*|*"pytorch"*|*"tensorflow"*)
            echo "ai-ml"
            ;;
        *"mobile"*|*"react-native"*|*"flutter"*|*"electron"*)
            echo "mobile"
            ;;
        *"devops"*|*"ci"*|*"docker"*|*"kubernetes"*|*"terraform"*)
            echo "devops"
            ;;
        *"vscode"*|*"extension"*)
            echo "vscode-extensions"
            ;;
        *"docs"*|*"documentation"*|*"readme"*)
            echo "docs"
            ;;
        *"workflow"*|*"github"*|*"action"*|*"template"*)
            echo "workflows"
            ;;
        *)
            # Fallback: detect from template name
            case "$template_name" in
                *"fullstack"*|*"full-stack"*)
                    echo "fullstack"
                    ;;
                *"frontend"*|*"react"*|*"vue"*|*"svelte"*)
                    echo "frontend"
                    ;;
                *"backend"*|*"api"*|*"server"*)
                    echo "backend"
                    ;;
                *"ml"*|*"ai"*|*"data"*)
                    echo "ai-ml"
                    ;;
                *"mobile"*|*"native"*)
                    echo "mobile"
                    ;;
                *"devops"*|*"ci"*|*"infra"*)
                    echo "devops"
                    ;;
                *"vscode"*|*"extension"*)
                    echo "vscode-extensions"
                    ;;
                *"docs"*|*"documentation"*)
                    echo "docs"
                    ;;
                *"workflow"*|*"github"*|*"action"*)
                    echo "workflows"
                    ;;
                *)
                    echo "other"
                    ;;
            esac
            ;;
    esac
}

# Function to get target stack branch name
get_stack_branch() {
    local category="$1"
    echo "stack/$category"
}

# Function to check if we're on the correct branch
check_branch() {
    local target_branch="$1"
    local current_branch
    current_branch=$(git branch --show-current)
    
    if [[ "$current_branch" != "$target_branch" ]]; then
        print_info "Current branch: $current_branch"
        print_info "Target branch: $target_branch"
        print_warning "Not on target branch. Auto-checking out to $target_branch..."
        
        # Check if target branch exists locally
        if git show-ref --verify --quiet "refs/heads/$target_branch"; then
            git checkout "$target_branch"
        else
            # Check if target branch exists remotely
            if git show-ref --verify --quiet "refs/remotes/origin/$target_branch"; then
                print_info "Creating local branch from remote $target_branch..."
                git checkout -b "$target_branch" "origin/$target_branch"
            else
                print_error "Target branch '$target_branch' does not exist locally or remotely."
                print_info "Available branches:"
                git branch -a | grep -E "(stack/|dev)" | sed 's/^/  /'
                print_info "Please create the stack branch first using:"
                print_info "  ./scripts/create_stack_branch.sh $category"
                exit 1
            fi
        fi
        
        print_success "Switched to branch: $target_branch"
    else
        print_info "Already on target branch: $target_branch"
    fi
}

# Function to validate template name
validate_template_name() {
    local name="$1"
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        print_error "Invalid template name: '$name'. Use only alphanumeric characters, hyphens, and underscores."
        exit 1
    fi
    
    # Check length
    if [[ ${#name} -gt 50 ]]; then
        print_error "Template name too long: '$name'. Maximum 50 characters."
        exit 1
    fi
}

# Function to validate upstream URL
validate_upstream_url() {
    local url="$1"
    
    # Basic URL validation
    if [[ ! "$url" =~ ^https?:// ]]; then
        print_error "Invalid upstream URL: '$url'. Must be a valid HTTP/HTTPS URL."
        exit 1
    fi
    
    # Check if it's a GitHub/GitLab URL
    if [[ ! "$url" =~ (github\.com|gitlab\.com|bitbucket\.org) ]]; then
        print_warning "URL doesn't appear to be from GitHub, GitLab, or Bitbucket: '$url'"
    fi
}

# Function to check if template already exists
check_existing_template() {
    local template_path="$1"
    local force="$2"
    
    if [[ -d "$template_path" ]]; then
        if [[ "$force" == "true" ]]; then
            print_warning "Template already exists at '$template_path'. Force overwrite enabled."
            return 0
        else
            print_error "Template already exists at '$template_path'. Use --force to overwrite."
            exit 1
        fi
    fi
}

# Function to create upstream info file
create_upstream_info() {
    local template_path="$1"
    local upstream_url="$2"
    local branch="$3"
    local subdir="$4"
    
    local upstream_info_file="$template_path/.upstream-info"
    
    cat > "$upstream_info_file" << EOF
# Upstream Template Information
# This file tracks the source of this template

Upstream URL: $upstream_url
Branch: $branch
Subdirectory: ${subdir:-/}
Last Sync: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Sync Script: scripts/sync_template.sh
Sync Command: $0 $@

# License Information
# Please check the upstream repository for license details
# and ensure compliance with the original license terms.

# Attribution
# This template is based on work from the upstream repository.
# Please maintain proper attribution when using this template.
EOF
}

# Function to perform the sync
sync_template() {
    local template_name="$1"
    local upstream_url="$2"
    local category="$3"
    local branch="$4"
    local subdir="$5"
    local force="$6"
    local dry_run="$7"
    local verbose="$8"
    
    local template_path="$STACKS_DIR/$category/$template_name"
    
    print_info "Syncing template '$template_name' from '$upstream_url'"
    print_info "Target category: $category"
    print_info "Target path: $template_path"
    
    if [[ "$dry_run" == "true" ]]; then
        print_info "DRY RUN: Would sync template to '$template_path'"
        return 0
    fi
    
    # Check if template already exists
    check_existing_template "$template_path" "$force"
    
    # Create temporary directory
    mkdir -p "$TEMP_DIR"
    
    # Clone upstream repository
    print_info "Cloning upstream repository..."
    if [[ "$verbose" == "true" ]]; then
        git clone --depth 1 --branch "$branch" "$upstream_url" "$TEMP_DIR/upstream"
    else
        git clone --depth 1 --branch "$branch" "$upstream_url" "$TEMP_DIR/upstream" > /dev/null 2>&1
    fi
    
    # Create target directory
    mkdir -p "$template_path"
    
    # Copy files
    if [[ -n "$subdir" ]]; then
        local source_path="$TEMP_DIR/upstream/$subdir"
        if [[ ! -d "$source_path" ]]; then
            print_error "Subdirectory '$subdir' not found in upstream repository"
            rm -rf "$TEMP_DIR"
            exit 1
        fi
        print_info "Copying files from subdirectory '$subdir'..."
        cp -r "$source_path"/* "$template_path/"
    else
        print_info "Copying all files from upstream repository..."
        cp -r "$TEMP_DIR/upstream"/* "$template_path/"
    fi
    
    # Create upstream info file
    create_upstream_info "$template_path" "$upstream_url" "$branch" "$subdir"
    
    # Clean up
    rm -rf "$TEMP_DIR"
    
    print_success "Template '$template_name' synced successfully to '$template_path'"
    print_info "Next steps:"
    print_info "  1. Review the template files in '$template_path'"
    print_info "  2. Update the template README.md with organization-specific information"
    print_info "  3. Test the template by creating a new project"
    print_info "  4. Commit the changes to version control"
}

# Main function
main() {
    local template_name=""
    local upstream_url=""
    local category=""
    local branch="main"
    local subdir=""
    local force="false"
    local dry_run="false"
    local verbose="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -f|--force)
                force="true"
                shift
                ;;
            -d|--dry-run)
                dry_run="true"
                shift
                ;;
            -v|--verbose)
                verbose="true"
                shift
                ;;
            --branch)
                branch="$2"
                shift 2
                ;;
            --subdir)
                subdir="$2"
                shift 2
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$template_name" ]]; then
                    template_name="$1"
                elif [[ -z "$upstream_url" ]]; then
                    upstream_url="$1"
                elif [[ -z "$category" ]]; then
                    category="$1"
                else
                    print_error "Too many arguments"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$template_name" || -z "$upstream_url" ]]; then
        print_error "Template name and upstream URL are required"
        show_usage
        exit 1
    fi
    
    # Validate inputs
    validate_template_name "$template_name"
    validate_upstream_url "$upstream_url"
    
    # Auto-detect category if not provided
    if [[ -z "$category" ]]; then
        category=$(detect_category "$template_name" "$upstream_url")
        print_info "Auto-detected category: $category"
    fi
    
    # Get target stack branch
    local target_branch
    target_branch=$(get_stack_branch "$category")
    print_info "Target stack branch: $target_branch"
    
    # Check and switch to target branch
    check_branch "$target_branch"
    
    # Update STACKS_DIR to point to the current branch's stacks directory
    STACKS_DIR="$REPO_ROOT/stacks"
    
    # Validate category directory exists in current branch
    if [[ ! -d "$STACKS_DIR/$category" ]]; then
        print_error "Invalid category: '$category'. Available categories in current branch:"
        if [[ -d "$STACKS_DIR" ]]; then
            ls -1 "$STACKS_DIR" | sed 's/^/  - /'
        else
            print_error "No stacks directory found in current branch."
            print_info "This might be the main/dev branch. Please switch to a stack branch first."
        fi
        exit 1
    fi
    
    # Perform the sync
    sync_template "$template_name" "$upstream_url" "$category" "$branch" "$subdir" "$force" "$dry_run" "$verbose"
}

# Run main function with all arguments
main "$@"
