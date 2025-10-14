# create_stack_branch.ps1 - Create a new stack branch with proper structure and configuration
# Usage: .\scripts\create_stack_branch.ps1 <stack-name> [options]

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$StackName,
    
    [Parameter(Mandatory=$false)]
    [string]$Description = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Maintainers = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$Force,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory=$false)]
    [switch]$Help
)

# Function to print colored output
function Write-Status {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colorMap = @{
        "Red" = "Red"
        "Green" = "Green"
        "Yellow" = "Yellow"
        "Blue" = "Cyan"
        "White" = "White"
    }
    
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

# Function to show usage
function Show-Usage {
    Write-Host @"
Usage: .\scripts\create_stack_branch.ps1 <stack-name> [options]

Create a new stack branch with proper structure and configuration.

Arguments:
    stack-name          Name of the stack to create (e.g., frontend, backend, ai-ml)

Options:
    -Description        Description of the stack
    -Maintainers        Comma-separated list of maintainers (e.g., "@team1,@team2")
    -Force              Force creation even if branch exists
    -DryRun             Show what would be done without making changes
    -Verbose            Enable verbose output
    -Help               Show this help message

Examples:
    .\scripts\create_stack_branch.ps1 frontend -Description "Frontend framework templates"
    .\scripts\create_stack_branch.ps1 backend -Description "Backend service templates" -Maintainers "@backend-team"
    .\scripts\create_stack_branch.ps1 ai-ml -DryRun

Available stack categories:
    Core Development: fullstack, frontend, backend, mobile
    AI/ML: ai-ml, advanced-ai, agentic-ai, generative-ai
    Infrastructure: devops, microservices, monorepo, serverless
    Specialized: web3, quantum-computing, computational-biology, scientific-computing
    Emerging: space-technologies, 6g-wireless, structural-batteries, polyfunctional-robots
    Tools: modern-languages, vscode-extensions, docs, workflows
"@
}

# Function to validate stack name
function Test-StackName {
    param([string]$Stack)
    
    # Check if stack name is provided
    if ([string]::IsNullOrEmpty($Stack)) {
        Write-Status "Error: Stack name is required" "Red"
        Show-Usage
        exit 1
    }
    
    # Check if stack name contains valid characters
    if ($Stack -notmatch '^[a-z0-9-]+$') {
        Write-Status "Error: Stack name must contain only lowercase letters, numbers, and hyphens" "Red"
        exit 1
    }
    
    # Check if stack name is too long
    if ($Stack.Length -gt 50) {
        Write-Status "Error: Stack name is too long (max 50 characters)" "Red"
        exit 1
    }
}

# Function to check if branch exists
function Test-BranchExists {
    param([string]$BranchName)
    
    try {
        $localBranch = git show-ref --verify --quiet "refs/heads/$BranchName" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        
        $remoteBranch = git show-ref --verify --quiet "refs/remotes/origin/$BranchName" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        
        return $false
    }
    catch {
        return $false
    }
}

# Function to create stack configuration
function New-StackConfig {
    param([string]$ConfigFile)
    
    $configContent = @"
stack_name: "$($StackName -replace '-', ' ' | ForEach-Object { $_.Split(' ') | ForEach-Object { $_.Substring(0,1).ToUpper() + $_.Substring(1) } } | Join-String ' ')"
category: "$StackName"
description: "$Description"
maintainers: [$Maintainers]
upstream_sources: []
trend_keywords:
  - "$StackName"
auto_sync: true
created_at: "$(Get-Date -Format "yyyy-MM-dd HH:mm:ss") UTC"
last_updated: "$(Get-Date -Format "yyyy-MM-dd HH:mm:ss") UTC"
version: "1.0.0"
"@
    
    $configContent | Out-File -FilePath $ConfigFile -Encoding UTF8
    Write-Status "Created stack configuration: $ConfigFile" "Green"
}

# Function to create trend detection configuration
function New-TrendConfig {
    param([string]$ConfigFile)
    
    $configContent = @"
stack_name: "$StackName"
enabled: true
keywords:
  - "$StackName"
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
"@
    
    $configContent | Out-File -FilePath $ConfigFile -Encoding UTF8
    Write-Status "Created trend detection configuration: $ConfigFile" "Green"
}

# Function to create stack README
function New-StackReadme {
    param([string]$ReadmeFile)
    
    $stackTitle = ($StackName -replace '-', ' ' | ForEach-Object { $_.Split(' ') | ForEach-Object { $_.Substring(0,1).ToUpper() + $_.Substring(1) } } | Join-String ' ')
    
    $readmeContent = @"
# $stackTitle Stack

$Description

## 📋 Available Templates

This stack contains templates for $StackName development. See [TEMPLATES.md](./TEMPLATES.md) for a complete list.

## 🚀 Quick Start

1. **Browse Templates**
   ```bash
   ls stacks/$StackName/
   ```

2. **Use a Template**
   ```bash
   cp -r stacks/$StackName/template-name ../my-new-project
   cd ../my-new-project
   # Follow template-specific setup instructions
   ```

## 📚 Documentation

- [Template List](./TEMPLATES.md) - Complete list of available templates
- [Stack Configuration](./.stack-config.yml) - Stack configuration
- [Trend Detection](./.trend-detection-config.yml) - Trend detection settings

## 🤝 Contributing

To add new templates to this stack:

1. Use the sync script:
   ```bash
   .\scripts\sync_template.ps1 template-name upstream-url $StackName
   ```

2. Or manually add templates following the [contribution guidelines](../../docs/CONTRIBUTING_TO_STACKS.md)

## 📊 Stack Statistics

- **Templates**: 0 (initial)
- **Last Updated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") UTC
- **Maintainers**: $Maintainers

---

**Stack**: $StackName  
**Version**: 1.0.0  
**Last Updated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") UTC
"@
    
    $readmeContent | Out-File -FilePath $ReadmeFile -Encoding UTF8
    Write-Status "Created stack README: $ReadmeFile" "Green"
}

# Function to create templates index
function New-TemplatesIndex {
    param([string]$TemplatesFile)
    
    $stackTitle = ($StackName -replace '-', ' ' | ForEach-Object { $_.Split(' ') | ForEach-Object { $_.Substring(0,1).ToUpper() + $_.Substring(1) } } | Join-String ' ')
    
    $templatesContent = @"
# $stackTitle Templates

This document lists all available templates in the $StackName stack.

## 📋 Template List

| Template | Description | Upstream | Last Updated | Status |
|----------|-------------|----------|--------------|--------|
| *No templates yet* | *Add templates using the sync script* | - | - | - |

## 🚀 Adding Templates

To add a new template to this stack:

```bash
# Use the automated sync script
.\scripts\sync_template.ps1 template-name upstream-url $StackName

# Example
.\scripts\sync_template.ps1 react-vite https://github.com/vitejs/vite $StackName
```

## 📊 Template Statistics

- **Total Templates**: 0
- **Active Templates**: 0
- **Deprecated Templates**: 0
- **Last Updated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") UTC

---

**Stack**: $StackName  
**Last Updated**: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") UTC
"@
    
    $templatesContent | Out-File -FilePath $TemplatesFile -Encoding UTF8
    Write-Status "Created templates index: $TemplatesFile" "Green"
}

# Function to create .gitkeep file
function New-Gitkeep {
    param([string]$GitkeepFile)
    
    New-Item -Path $GitkeepFile -ItemType File -Force | Out-Null
    Write-Status "Created .gitkeep file: $GitkeepFile" "Green"
}

# Function to create stack branch
function New-StackBranch {
    $branchName = "stack/$StackName"
    
    Write-Status "Creating stack branch: $branchName" "Blue"
    
    # Check if we're on dev branch
    $currentBranch = git branch --show-current
    if ($currentBranch -ne "dev") {
        Write-Status "Warning: Not on dev branch (current: $currentBranch)" "Yellow"
        if (-not $Force) {
            Write-Status "Error: Must be on dev branch to create stack branches" "Red"
            exit 1
        }
    }
    
    # Create and switch to new branch
    if ($DryRun) {
        Write-Status "[DRY RUN] Would create branch: $branchName" "Yellow"
    } else {
        git checkout -b $branchName
        Write-Status "Created and switched to branch: $branchName" "Green"
    }
    
    # Create stack directory structure
    $stackDir = "stacks/$StackName"
    if ($DryRun) {
        Write-Status "[DRY RUN] Would create directory: $stackDir" "Yellow"
    } else {
        New-Item -Path $stackDir -ItemType Directory -Force | Out-Null
        Write-Status "Created directory: $stackDir" "Green"
    }
    
    # Create stack files
    if ($DryRun) {
        Write-Status "[DRY RUN] Would create stack configuration files" "Yellow"
    } else {
        New-StackConfig "stacks/$StackName/.stack-config.yml"
        New-TrendConfig "stacks/$StackName/.trend-detection-config.yml"
        New-StackReadme "stacks/$StackName/README.md"
        New-TemplatesIndex "stacks/$StackName/TEMPLATES.md"
        New-Gitkeep "stacks/$StackName/.gitkeep"
    }
    
    # Add and commit files
    if ($DryRun) {
        Write-Status "[DRY RUN] Would add and commit files" "Yellow"
    } else {
        git add "stacks/$StackName/"
        $commitMessage = @"
feat: create $StackName stack branch

- Add stack configuration and documentation
- Initialize template directory structure
- Configure trend detection settings

Stack: $StackName
Maintainers: $Maintainers
"@
        git commit -m $commitMessage
        Write-Status "Committed initial stack structure" "Green"
    }
    
    # Push branch to remote
    if ($DryRun) {
        Write-Status "[DRY RUN] Would push branch to remote" "Yellow"
    } else {
        git push -u origin $branchName
        Write-Status "Pushed branch to remote: origin/$branchName" "Green"
    }
}

# Function to show summary
function Show-Summary {
    Write-Status "✅ Stack branch creation completed!" "Green"
    Write-Host ""
    Write-Status "Summary:" "Blue"
    Write-Host "  Stack Name: $StackName"
    Write-Host "  Branch: stack/$StackName"
    Write-Host "  Description: $Description"
    Write-Host "  Maintainers: $Maintainers"
    Write-Host ""
    Write-Status "Next Steps:" "Blue"
    Write-Host "  1. Switch to the new branch: git checkout stack/$StackName"
    Write-Host "  2. Add templates: .\scripts\sync_template.ps1 template-name upstream-url $StackName"
    Write-Host "  3. Update documentation as needed"
    Write-Host "  4. Create pull request to merge back to dev"
}

# Main execution
if ($Help) {
    Show-Usage
    exit 0
}

# Set default values
if ([string]::IsNullOrEmpty($Description)) {
    $Description = "Templates for $StackName development"
}

if ([string]::IsNullOrEmpty($Maintainers)) {
    $Maintainers = "`"@$StackName-team`""
}

# Validate inputs
Test-StackName $StackName

# Check if branch already exists
$branchName = "stack/$StackName"
if (Test-BranchExists $branchName) {
    if (-not $Force) {
        Write-Status "Error: Branch stack/$StackName already exists" "Red"
        Write-Status "Use -Force to overwrite or choose a different name" "Yellow"
        exit 1
    } else {
        Write-Status "Warning: Branch stack/$StackName already exists, will overwrite" "Yellow"
    }
}

# Create the stack branch
Write-Status "Creating stack branch for: $StackName" "Blue"
New-StackBranch

# Show summary
if (-not $DryRun) {
    Show-Summary
} else {
    Write-Status "Dry run completed. Use without -DryRun to create the branch." "Yellow"
}
