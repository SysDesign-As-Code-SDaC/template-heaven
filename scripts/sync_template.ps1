# Organization Universal Template Repository - Template Sync Script (PowerShell)
# This script helps sync templates from upstream repositories without forking

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$TemplateName,
    
    [Parameter(Mandatory=$true, Position=1)]
    [string]$UpstreamUrl,
    
    [Parameter(Position=2)]
    [string]$TargetCategory,
    
    [string]$Branch = "main",
    [string]$Subdir = "",
    [switch]$Force,
    [switch]$DryRun,
    [switch]$Verbose,
    [switch]$Help
)

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Function to print colored output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

# Function to show usage
function Show-Usage {
    Write-Host @"
Usage: .\sync_template.ps1 [OPTIONS] TEMPLATE_NAME UPSTREAM_URL [TARGET_CATEGORY]

Sync a template from an upstream repository to the local template collection.

ARGUMENTS:
    TEMPLATE_NAME     Name for the template (e.g., 't3-stack', 'nextjs-app')
    UPSTREAM_URL      Git URL of the upstream repository
    TARGET_CATEGORY   Category directory (default: auto-detect from URL)

OPTIONS:
    -Branch BRANCH    Specify upstream branch (default: main)
    -Subdir PATH      Sync only a subdirectory from upstream
    -Force            Force overwrite existing template
    -DryRun           Show what would be done without making changes
    -Verbose          Enable verbose output
    -Help             Show this help message

EXAMPLES:
    .\sync_template.ps1 t3-stack https://github.com/t3-oss/create-t3-app
    .\sync_template.ps1 nextjs-app https://github.com/vercel/next.js -Subdir examples/hello-world
    .\sync_template.ps1 express-api https://github.com/sahat/hackathon-starter backend -Force

"@
}

# Function to detect category from URL or template name
function Get-Category {
    param(
        [string]$TemplateName,
        [string]$UpstreamUrl,
        [string]$ProvidedCategory
    )
    
    if ($ProvidedCategory) {
        return $ProvidedCategory
    }
    
    # Auto-detect based on URL patterns
    $urlLower = $UpstreamUrl.ToLower()
    $nameLower = $TemplateName.ToLower()
    
    if ($urlLower -match "full-stack|fullstack|t3|remix|next\.js") {
        return "fullstack"
    }
    elseif ($urlLower -match "react|vue|svelte|frontend|vite") {
        return "frontend"
    }
    elseif ($urlLower -match "api|backend|server|express|fastapi|django") {
        return "backend"
    }
    elseif ($urlLower -match "ml|ai|data-science|pytorch|tensorflow") {
        return "ai-ml"
    }
    elseif ($urlLower -match "mobile|react-native|flutter|electron") {
        return "mobile"
    }
    elseif ($urlLower -match "devops|ci|docker|kubernetes|terraform") {
        return "devops"
    }
    elseif ($urlLower -match "vscode|extension") {
        return "vscode-extensions"
    }
    elseif ($urlLower -match "docs|documentation|readme") {
        return "docs"
    }
    elseif ($urlLower -match "workflow|github|action|template") {
        return "workflows"
    }
    else {
        # Fallback: detect from template name
        if ($nameLower -match "fullstack|full-stack") { return "fullstack" }
        elseif ($nameLower -match "frontend|react|vue|svelte") { return "frontend" }
        elseif ($nameLower -match "backend|api|server") { return "backend" }
        elseif ($nameLower -match "ml|ai|data") { return "ai-ml" }
        elseif ($nameLower -match "mobile|native") { return "mobile" }
        elseif ($nameLower -match "devops|ci|infra") { return "devops" }
        elseif ($nameLower -match "vscode|extension") { return "vscode-extensions" }
        elseif ($nameLower -match "docs|documentation") { return "docs" }
        elseif ($nameLower -match "workflow|github|action") { return "workflows" }
        else { return "other" }
    }
}

# Function to get target stack branch name
function Get-StackBranch {
    param([string]$Category)
    return "stack/$Category"
}

# Function to check if we're on the correct branch
function Test-Branch {
    param([string]$TargetBranch)
    
    $currentBranch = git branch --show-current
    
    if ($currentBranch -ne $TargetBranch) {
        Write-Info "Current branch: $currentBranch"
        Write-Info "Target branch: $TargetBranch"
        Write-Warning "Not on target branch. Auto-checking out to $TargetBranch..."
        
        # Check if target branch exists locally
        $localBranchExists = git show-ref --verify --quiet "refs/heads/$TargetBranch"
        if ($LASTEXITCODE -eq 0) {
            git checkout $TargetBranch
        }
        else {
            # Check if target branch exists remotely
            $remoteBranchExists = git show-ref --verify --quiet "refs/remotes/origin/$TargetBranch"
            if ($LASTEXITCODE -eq 0) {
                Write-Info "Creating local branch from remote $TargetBranch..."
                git checkout -b $TargetBranch "origin/$TargetBranch"
            }
            else {
                Write-Error "Target branch '$TargetBranch' does not exist locally or remotely."
                Write-Info "Available branches:"
                git branch -a | Select-String -Pattern "(stack/|dev)" | ForEach-Object { Write-Host "  $($_.Line)" }
                Write-Info "Please create the stack branch first using:"
                Write-Info "  .\scripts\create_stack_branch.ps1 $Category"
                exit 1
            }
        }
        
        Write-Success "Switched to branch: $TargetBranch"
    }
    else {
        Write-Info "Already on target branch: $TargetBranch"
    }
}

# Function to validate template name
function Test-TemplateName {
    param([string]$Name)
    
    if ($Name -notmatch '^[a-zA-Z0-9_-]+$') {
        Write-Error "Invalid template name: '$Name'. Use only alphanumeric characters, hyphens, and underscores."
        exit 1
    }
    
    if ($Name.Length -gt 50) {
        Write-Error "Template name too long: '$Name'. Maximum 50 characters."
        exit 1
    }
}

# Function to validate upstream URL
function Test-UpstreamUrl {
    param([string]$Url)
    
    if ($Url -notmatch '^https?://') {
        Write-Error "Invalid upstream URL: '$Url'. Must be a valid HTTP/HTTPS URL."
        exit 1
    }
    
    if ($Url -notmatch '(github\.com|gitlab\.com|bitbucket\.org)') {
        Write-Warning "URL doesn't appear to be from GitHub, GitLab, or Bitbucket: '$Url'"
    }
}

# Function to check if template already exists
function Test-ExistingTemplate {
    param(
        [string]$TemplatePath,
        [bool]$Force
    )
    
    if (Test-Path $TemplatePath) {
        if ($Force) {
            Write-Warning "Template already exists at '$TemplatePath'. Force overwrite enabled."
        }
        else {
            Write-Error "Template already exists at '$TemplatePath'. Use -Force to overwrite."
            exit 1
        }
    }
}

# Function to create upstream info file
function New-UpstreamInfo {
    param(
        [string]$TemplatePath,
        [string]$UpstreamUrl,
        [string]$Branch,
        [string]$Subdir
    )
    
    $upstreamInfoFile = Join-Path $TemplatePath ".upstream-info"
    $syncDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss UTC"
    
    $content = @"
# Upstream Template Information
# This file tracks the source of this template

Upstream URL: $UpstreamUrl
Branch: $Branch
Subdirectory: $(if ($Subdir) { $Subdir } else { "/" })
Last Sync: $syncDate
Sync Script: scripts/sync_template.ps1
Sync Command: .\sync_template.ps1 $TemplateName $UpstreamUrl $(if ($TargetCategory) { $TargetCategory })

# License Information
# Please check the upstream repository for license details
# and ensure compliance with the original license terms.

# Attribution
# This template is based on work from the upstream repository.
# Please maintain proper attribution when using this template.
"@
    
    Set-Content -Path $upstreamInfoFile -Value $content
}

# Function to perform the sync
function Sync-Template {
    param(
        [string]$TemplateName,
        [string]$UpstreamUrl,
        [string]$Category,
        [string]$Branch,
        [string]$Subdir,
        [bool]$Force,
        [bool]$DryRun,
        [bool]$Verbose
    )
    
    $ScriptDir = Split-Path -Parent $MyInvocation.PSCommandPath
    $RepoRoot = Split-Path -Parent $ScriptDir
    $StacksDir = Join-Path $RepoRoot "stacks"
    $TemplatePath = Join-Path $StacksDir "$Category\$TemplateName"
    $TempDir = Join-Path $env:TEMP "template-sync-$(Get-Random)"
    
    Write-Info "Syncing template '$TemplateName' from '$UpstreamUrl'"
    Write-Info "Target category: $Category"
    Write-Info "Target path: $TemplatePath"
    
    if ($DryRun) {
        Write-Info "DRY RUN: Would sync template to '$TemplatePath'"
        return
    }
    
    # Check if template already exists
    Test-ExistingTemplate $TemplatePath $Force
    
    # Create temporary directory
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    
    try {
        # Clone upstream repository
        Write-Info "Cloning upstream repository..."
        $cloneArgs = @("clone", "--depth", "1", "--branch", $Branch, $UpstreamUrl, (Join-Path $TempDir "upstream"))
        
        if ($Verbose) {
            & git @cloneArgs
        }
        else {
            & git @cloneArgs 2>$null
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to clone repository"
        }
        
        # Create target directory
        New-Item -ItemType Directory -Path $TemplatePath -Force | Out-Null
        
        # Copy files
        $upstreamPath = Join-Path $TempDir "upstream"
        if ($Subdir) {
            $sourcePath = Join-Path $upstreamPath $Subdir
            if (-not (Test-Path $sourcePath)) {
                throw "Subdirectory '$Subdir' not found in upstream repository"
            }
            Write-Info "Copying files from subdirectory '$Subdir'..."
            Copy-Item -Path "$sourcePath\*" -Destination $TemplatePath -Recurse -Force
        }
        else {
            Write-Info "Copying all files from upstream repository..."
            Copy-Item -Path "$upstreamPath\*" -Destination $TemplatePath -Recurse -Force
        }
        
        # Create upstream info file
        New-UpstreamInfo $TemplatePath $UpstreamUrl $Branch $Subdir
        
        Write-Success "Template '$TemplateName' synced successfully to '$TemplatePath'"
        Write-Info "Next steps:"
        Write-Info "  1. Review the template files in '$TemplatePath'"
        Write-Info "  2. Update the template README.md with organization-specific information"
        Write-Info "  3. Test the template by creating a new project"
        Write-Info "  4. Commit the changes to version control"
    }
    finally {
        # Clean up
        if (Test-Path $TempDir) {
            Remove-Item -Path $TempDir -Recurse -Force
        }
    }
}

# Main execution
if ($Help) {
    Show-Usage
    exit 0
}

# Validate required arguments
if (-not $TemplateName -or -not $UpstreamUrl) {
    Write-Error "Template name and upstream URL are required"
    Show-Usage
    exit 1
}

# Validate inputs
Test-TemplateName $TemplateName
Test-UpstreamUrl $UpstreamUrl

# Auto-detect category if not provided
if (-not $TargetCategory) {
    $TargetCategory = Get-Category $TemplateName $UpstreamUrl
    Write-Info "Auto-detected category: $TargetCategory"
}

# Get target stack branch
$TargetBranch = Get-StackBranch $TargetCategory
Write-Info "Target stack branch: $TargetBranch"

# Check and switch to target branch
Test-Branch $TargetBranch

# Update StacksDir to point to the current branch's stacks directory
$ScriptDir = Split-Path -Parent $MyInvocation.PSCommandPath
$RepoRoot = Split-Path -Parent $ScriptDir
$StacksDir = Join-Path $RepoRoot "stacks"

# Validate category directory exists in current branch
$CategoryPath = Join-Path $StacksDir $TargetCategory
if (-not (Test-Path $CategoryPath)) {
    Write-Error "Invalid category: '$TargetCategory'. Available categories in current branch:"
    if (Test-Path $StacksDir) {
        Get-ChildItem $StacksDir -Directory | ForEach-Object { Write-Host "  - $($_.Name)" }
    }
    else {
        Write-Error "No stacks directory found in current branch."
        Write-Info "This might be the main/dev branch. Please switch to a stack branch first."
    }
    exit 1
}

# Perform the sync
Sync-Template $TemplateName $UpstreamUrl $TargetCategory $Branch $Subdir $Force $DryRun $Verbose
