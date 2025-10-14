#!/usr/bin/env python3
"""
Branch Manager - Advanced Python script for managing stack branches

This script provides comprehensive functionality for managing stack branches
in the template-heaven multi-branch architecture.

Usage:
    python scripts/branch_manager.py [COMMAND] [OPTIONS]

Commands:
    list                    List all available stack branches
    create <name>           Create a new stack branch
    validate <name>         Validate a stack branch structure
    sync <name>             Sync core tools to a stack branch
    status <name>           Show status of a stack branch
    report                  Generate comprehensive stack report
    cleanup                 Clean up temporary files and branches
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BranchManager:
    """Advanced branch manager for template-heaven stack branches."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the branch manager.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.repo_root = Path(__file__).parent.parent
        self.stacks_dir = self.repo_root / "stacks"
        
        # Available stack categories
        self.stack_categories = {
            "core": ["fullstack", "frontend", "backend", "mobile"],
            "ai-ml": ["ai-ml", "advanced-ai", "agentic-ai", "generative-ai"],
            "infrastructure": ["devops", "microservices", "monorepo", "serverless"],
            "specialized": ["web3", "quantum-computing", "computational-biology", "scientific-computing"],
            "emerging": ["space-technologies", "6g-wireless", "structural-batteries", "polyfunctional-robots"],
            "tools": ["modern-languages", "vscode-extensions", "docs", "workflows"]
        }
        
        # All available stacks
        self.all_stacks = [stack for category in self.stack_categories.values() for stack in category]
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARN, ERROR)
        """
        if self.verbose or level in ["WARN", "ERROR"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a shell command and return the result.
        
        Args:
            command: Command to run as list of strings
            cwd: Working directory for the command
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_root,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            self.log(f"Error running command {' '.join(command)}: {e}", "ERROR")
            return 1, "", str(e)
    
    def get_current_branch(self) -> str:
        """Get the current git branch.
        
        Returns:
            Current branch name
        """
        return_code, stdout, stderr = self.run_command(["git", "branch", "--show-current"])
        if return_code == 0:
            return stdout.strip()
        else:
            self.log(f"Error getting current branch: {stderr}", "ERROR")
            return "unknown"
    
    def get_all_branches(self) -> List[str]:
        """Get all available branches.
        
        Returns:
            List of branch names
        """
        return_code, stdout, stderr = self.run_command(["git", "branch", "-r"])
        if return_code == 0:
            branches = []
            for line in stdout.split('\n'):
                if 'origin/stack/' in line:
                    branch = line.strip().replace('origin/', '')
                    branches.append(branch)
            return branches
        else:
            self.log(f"Error getting branches: {stderr}", "ERROR")
            return []
    
    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists.
        
        Args:
            branch_name: Name of the branch to check
            
        Returns:
            True if branch exists, False otherwise
        """
        return_code, stdout, stderr = self.run_command(["git", "show-ref", "--verify", f"refs/heads/{branch_name}"])
        if return_code == 0:
            return True
        
        return_code, stdout, stderr = self.run_command(["git", "show-ref", "--verify", f"refs/remotes/origin/{branch_name}"])
        return return_code == 0
    
    def list_branches(self) -> None:
        """List all available stack branches with their status."""
        print("Available Stack Branches")
        print("=" * 50)
        
        all_branches = self.get_all_branches()
        current_branch = self.get_current_branch()
        
        for category, stacks in self.stack_categories.items():
            print(f"\n{category.upper()}")
            print("-" * 30)
            
            for stack in stacks:
                branch_name = f"stack/{stack}"
                status = "[OK]" if branch_name in all_branches else "[MISSING]"
                current = " (current)" if branch_name == current_branch else ""
                
                # Get stack info if it exists
                stack_info = ""
                if branch_name in all_branches:
                    stack_path = self.stacks_dir / stack
                    if stack_path.exists():
                        config_file = stack_path / ".stack-config.yml"
                        if config_file.exists():
                            try:
                                with open(config_file, 'r') as f:
                                    config = yaml.safe_load(f)
                                    version = config.get('version', 'unknown')
                                    last_updated = config.get('last_updated', 'unknown')
                                    stack_info = f" (v{version}, updated: {last_updated})"
                            except Exception as e:
                                self.log(f"Error reading config for {stack}: {e}", "WARN")
                
                print(f"  {status} {stack:<25} {stack_info}{current}")
        
        print(f"\nSummary:")
        print(f"  Total stacks: {len(self.all_stacks)}")
        print(f"  Created branches: {len(all_branches)}")
        print(f"  Current branch: {current_branch}")
    
    def create_branch(self, stack_name: str, description: str = "", maintainers: str = "", force: bool = False) -> bool:
        """Create a new stack branch.
        
        Args:
            stack_name: Name of the stack to create
            description: Description of the stack
            maintainers: Comma-separated list of maintainers
            force: Force creation even if branch exists
            
        Returns:
            True if successful, False otherwise
        """
        if stack_name not in self.all_stacks:
            self.log(f"Invalid stack name: {stack_name}", "ERROR")
            self.log(f"Available stacks: {', '.join(self.all_stacks)}", "ERROR")
            return False
        
        branch_name = f"stack/{stack_name}"
        
        # Check if branch already exists
        if self.branch_exists(branch_name) and not force:
            self.log(f"Branch {branch_name} already exists. Use --force to overwrite.", "ERROR")
            return False
        
        # Check if we're on dev branch
        current_branch = self.get_current_branch()
        if current_branch != "dev" and not force:
            self.log(f"Must be on dev branch to create stack branches (current: {current_branch})", "ERROR")
            return False
        
        self.log(f"Creating stack branch: {branch_name}")
        
        try:
            # Create and switch to new branch
            if self.branch_exists(branch_name) and force:
                self.log(f"Deleting existing branch: {branch_name}")
                return_code, _, stderr = self.run_command(["git", "branch", "-D", branch_name])
                if return_code != 0:
                    self.log(f"Error deleting branch: {stderr}", "WARN")
            
            return_code, _, stderr = self.run_command(["git", "checkout", "-b", branch_name])
            if return_code != 0:
                self.log(f"Error creating branch: {stderr}", "ERROR")
                return False
            
            # Create stack directory
            stack_dir = self.stacks_dir / stack_name
            stack_dir.mkdir(parents=True, exist_ok=True)
            
            # Set default values
            if not description:
                description = f"Templates for {stack_name} development"
            if not maintainers:
                maintainers = f"\"@{stack_name}-team\""
            
            # Create stack configuration
            self._create_stack_config(stack_name, description, maintainers)
            
            # Create trend detection configuration
            self._create_trend_config(stack_name)
            
            # Create README
            self._create_stack_readme(stack_name, description, maintainers)
            
            # Create templates index
            self._create_templates_index(stack_name)
            
            # Create .gitkeep
            (stack_dir / ".gitkeep").touch()
            
            # Add and commit files
            return_code, _, stderr = self.run_command(["git", "add", f"stacks/{stack_name}/"])
            if return_code != 0:
                self.log(f"Error adding files: {stderr}", "ERROR")
                return False
            
            commit_message = f"""feat: create {stack_name} stack branch

- Add stack configuration and documentation
- Initialize template directory structure
- Configure trend detection settings

Stack: {stack_name}
Maintainers: {maintainers}"""
            
            return_code, _, stderr = self.run_command(["git", "commit", "-m", commit_message])
            if return_code != 0:
                self.log(f"Error committing files: {stderr}", "ERROR")
                return False
            
            # Push branch to remote
            return_code, _, stderr = self.run_command(["git", "push", "-u", "origin", branch_name])
            if return_code != 0:
                self.log(f"Error pushing branch: {stderr}", "ERROR")
                return False
            
            self.log(f"Successfully created stack branch: {branch_name}", "INFO")
            return True
            
        except Exception as e:
            self.log(f"Error creating branch: {e}", "ERROR")
            return False
    
    def _create_stack_config(self, stack_name: str, description: str, maintainers: str) -> None:
        """Create stack configuration file."""
        config = {
            "stack_name": stack_name.replace('-', ' ').title(),
            "category": stack_name,
            "description": description,
            "maintainers": [maintainers] if maintainers else [f"@{stack_name}-team"],
            "upstream_sources": [],
            "trend_keywords": [stack_name],
            "auto_sync": True,
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "version": "1.0.0"
        }
        
        config_file = self.stacks_dir / stack_name / ".stack-config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        self.log(f"Created stack configuration: {config_file}")
    
    def _create_trend_config(self, stack_name: str) -> None:
        """Create trend detection configuration file."""
        config = {
            "stack_name": stack_name,
            "enabled": True,
            "keywords": [stack_name, "template", "boilerplate", "starter"],
            "characteristics": {
                "has_readme": True,
                "has_license": True,
                "has_ci": True
            },
            "thresholds": {
                "stars": {"minimum": 100, "trending": 1000, "critical": 5000},
                "forks": {"minimum": 10, "trending": 100, "critical": 500},
                "growth_rate": {"minimum": 0.1, "trending": 0.5, "critical": 1.0}
            },
            "auto_sync": {"enabled": True, "require_approval": True},
            "notifications": {
                "enabled": True,
                "channels": ["slack", "email"],
                "priority_threshold": 0.7
            }
        }
        
        config_file = self.stacks_dir / stack_name / ".trend-detection-config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        self.log(f"Created trend detection configuration: {config_file}")
    
    def _create_stack_readme(self, stack_name: str, description: str, maintainers: str) -> None:
        """Create stack README file."""
        stack_title = stack_name.replace('-', ' ').title()
        
        readme_content = f"""# {stack_title} Stack

{description}

## Available Templates

This stack contains templates for {stack_name} development. See [TEMPLATES.md](./TEMPLATES.md) for a complete list.

## Quick Start

1. **Browse Templates**
   ```bash
   ls stacks/{stack_name}/
   ```

2. **Use a Template**
   ```bash
   cp -r stacks/{stack_name}/template-name ../my-new-project
   cd ../my-new-project
   # Follow template-specific setup instructions
   ```

## Documentation

- [Template List](./TEMPLATES.md) - Complete list of available templates
- [Stack Configuration](./.stack-config.yml) - Stack configuration
- [Trend Detection](./.trend-detection-config.yml) - Trend detection settings

## Contributing

To add new templates to this stack:

1. Use the sync script:
   ```bash
   ./scripts/sync_to_branch.sh template-name upstream-url {stack_name}
   ```

2. Or manually add templates following the [contribution guidelines](../../docs/CONTRIBUTING_TO_STACKS.md)

## Stack Statistics

- **Templates**: 0 (initial)
- **Last Updated**: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
- **Maintainers**: {maintainers}

---

**Stack**: {stack_name}  
**Version**: 1.0.0  
**Last Updated**: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
"""
        
        readme_file = self.stacks_dir / stack_name / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        self.log(f"Created stack README: {readme_file}")
    
    def _create_templates_index(self, stack_name: str) -> None:
        """Create templates index file."""
        stack_title = stack_name.replace('-', ' ').title()
        
        templates_content = f"""# {stack_title} Templates

This document lists all available templates in the {stack_name} stack.

## Template List

| Template | Description | Upstream | Last Updated | Status |
|----------|-------------|----------|--------------|--------|
| *No templates yet* | *Add templates using the sync script* | - | - | - |

## Adding Templates

To add a new template to this stack:

```bash
# Use the automated sync script
./scripts/sync_to_branch.sh template-name upstream-url {stack_name}

# Example
./scripts/sync_to_branch.sh react-vite https://github.com/vitejs/vite {stack_name}
```

## Template Statistics

- **Total Templates**: 0
- **Active Templates**: 0
- **Deprecated Templates**: 0
- **Last Updated**: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC

---

**Stack**: {stack_name}  
**Last Updated**: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
"""
        
        templates_file = self.stacks_dir / stack_name / "TEMPLATES.md"
        with open(templates_file, 'w') as f:
            f.write(templates_content)
        
        self.log(f"Created templates index: {templates_file}")
    
    def validate_branch(self, stack_name: str) -> bool:
        """Validate a stack branch structure.
        
        Args:
            stack_name: Name of the stack to validate
            
        Returns:
            True if valid, False otherwise
        """
        branch_name = f"stack/{stack_name}"
        
        if not self.branch_exists(branch_name):
            self.log(f"Branch {branch_name} does not exist", "ERROR")
            return False
        
        self.log(f"Validating stack branch: {branch_name}")
        
        stack_dir = self.stacks_dir / stack_name
        if not stack_dir.exists():
            self.log(f"Stack directory does not exist: {stack_dir}", "ERROR")
            return False
        
        # Check required files
        required_files = [
            ".stack-config.yml",
            ".trend-detection-config.yml",
            "README.md",
            "TEMPLATES.md"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = stack_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.log(f"Missing required files: {', '.join(missing_files)}", "ERROR")
            return False
        
        # Validate configuration files
        try:
            config_file = stack_dir / ".stack-config.yml"
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            required_config_keys = ["stack_name", "category", "description", "version"]
            missing_keys = [key for key in required_config_keys if key not in config]
            
            if missing_keys:
                self.log(f"Missing configuration keys: {', '.join(missing_keys)}", "ERROR")
                return False
            
        except Exception as e:
            self.log(f"Error validating configuration: {e}", "ERROR")
            return False
        
        self.log(f"Stack branch {branch_name} is valid", "INFO")
        return True
    
    def sync_core_tools(self, stack_name: str) -> bool:
        """Sync core tools to a stack branch.
        
        Args:
            stack_name: Name of the stack to sync
            
        Returns:
            True if successful, False otherwise
        """
        branch_name = f"stack/{stack_name}"
        
        if not self.branch_exists(branch_name):
            self.log(f"Branch {branch_name} does not exist", "ERROR")
            return False
        
        self.log(f"Syncing core tools to: {branch_name}")
        
        try:
            # Switch to the stack branch
            return_code, _, stderr = self.run_command(["git", "checkout", branch_name])
            if return_code != 0:
                self.log(f"Error switching to branch: {stderr}", "ERROR")
                return False
            
            # Copy core tools from dev branch
            core_dirs = ["scripts", "tools"]
            for core_dir in core_dirs:
                if (self.repo_root / core_dir).exists():
                    return_code, _, stderr = self.run_command(["git", "checkout", "dev", "--", core_dir])
                    if return_code != 0:
                        self.log(f"Error copying {core_dir}: {stderr}", "WARN")
            
            # Add and commit changes
            return_code, _, stderr = self.run_command(["git", "add", "."])
            if return_code != 0:
                self.log(f"Error adding files: {stderr}", "ERROR")
                return False
            
            return_code, _, stderr = self.run_command(["git", "commit", "-m", f"feat: sync core tools to {stack_name} stack"])
            if return_code != 0:
                self.log(f"Error committing changes: {stderr}", "ERROR")
                return False
            
            # Push changes
            return_code, _, stderr = self.run_command(["git", "push", "origin", branch_name])
            if return_code != 0:
                self.log(f"Error pushing changes: {stderr}", "ERROR")
                return False
            
            self.log(f"Successfully synced core tools to {branch_name}", "INFO")
            return True
            
        except Exception as e:
            self.log(f"Error syncing core tools: {e}", "ERROR")
            return False
    
    def get_branch_status(self, stack_name: str) -> Dict:
        """Get status information for a stack branch.
        
        Args:
            stack_name: Name of the stack
            
        Returns:
            Dictionary with status information
        """
        branch_name = f"stack/{stack_name}"
        stack_dir = self.stacks_dir / stack_name
        
        status = {
            "stack_name": stack_name,
            "branch_name": branch_name,
            "exists": self.branch_exists(branch_name),
            "directory_exists": stack_dir.exists(),
            "template_count": 0,
            "last_updated": None,
            "version": None,
            "maintainers": []
        }
        
        if stack_dir.exists():
            # Count templates (directories that aren't config files)
            template_dirs = [d for d in stack_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            status["template_count"] = len(template_dirs)
            
            # Read configuration
            config_file = stack_dir / ".stack-config.yml"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    status["last_updated"] = config.get("last_updated")
                    status["version"] = config.get("version")
                    status["maintainers"] = config.get("maintainers", [])
                except Exception as e:
                    self.log(f"Error reading config: {e}", "WARN")
        
        return status
    
    def show_branch_status(self, stack_name: str) -> None:
        """Show detailed status for a stack branch."""
        status = self.get_branch_status(stack_name)
        
        print(f"📊 Stack Branch Status: {stack_name}")
        print("=" * 50)
        print(f"Branch Name: {status['branch_name']}")
        print(f"Exists: {'✅' if status['exists'] else '❌'}")
        print(f"Directory: {'✅' if status['directory_exists'] else '❌'}")
        print(f"Templates: {status['template_count']}")
        print(f"Version: {status['version'] or 'unknown'}")
        print(f"Last Updated: {status['last_updated'] or 'unknown'}")
        print(f"Maintainers: {', '.join(status['maintainers']) if status['maintainers'] else 'none'}")
    
    def generate_report(self) -> None:
        """Generate a comprehensive stack report."""
        print("Template Heaven Stack Report")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        all_branches = self.get_all_branches()
        current_branch = self.get_current_branch()
        
        # Overall statistics
        total_stacks = len(self.all_stacks)
        created_branches = len(all_branches)
        completion_rate = (created_branches / total_stacks) * 100
        
        print("Overall Statistics")
        print("-" * 30)
        print(f"Total Stacks: {total_stacks}")
        print(f"Created Branches: {created_branches}")
        print(f"Completion Rate: {completion_rate:.1f}%")
        print(f"Current Branch: {current_branch}")
        print()
        
        # Category breakdown
        print("Category Breakdown")
        print("-" * 30)
        for category, stacks in self.stack_categories.items():
            created_in_category = sum(1 for stack in stacks if f"stack/{stack}" in all_branches)
            total_in_category = len(stacks)
            category_rate = (created_in_category / total_in_category) * 100
            
            print(f"{category.title():<15}: {created_in_category}/{total_in_category} ({category_rate:.1f}%)")
        print()
        
        # Detailed stack information
        print("Detailed Stack Information")
        print("-" * 30)
        for stack in self.all_stacks:
            status = self.get_branch_status(stack)
            status_icon = "[OK]" if status["exists"] else "[MISSING]"
            template_info = f"({status['template_count']} templates)" if status["exists"] else ""
            
            print(f"{status_icon} {stack:<25} {template_info}")
            if status["exists"] and status["version"]:
                print(f"    Version: {status['version']}, Updated: {status['last_updated']}")
        print()
        
        # Recommendations
        print("Recommendations")
        print("-" * 30)
        if completion_rate < 100:
            missing_stacks = [stack for stack in self.all_stacks if f"stack/{stack}" not in all_branches]
            print(f"- Create missing stack branches: {', '.join(missing_stacks[:5])}")
            if len(missing_stacks) > 5:
                print(f"  ... and {len(missing_stacks) - 5} more")
        
        # Check for stacks with no templates
        empty_stacks = []
        for stack in self.all_stacks:
            if f"stack/{stack}" in all_branches:
                status = self.get_branch_status(stack)
                if status["template_count"] == 0:
                    empty_stacks.append(stack)
        
        if empty_stacks:
            print(f"- Add templates to empty stacks: {', '.join(empty_stacks[:5])}")
            if len(empty_stacks) > 5:
                print(f"  ... and {len(empty_stacks) - 5} more")
        
        print("- Run trend detection to identify new templates")
        print("- Update documentation and configurations regularly")
    
    def cleanup(self) -> bool:
        """Clean up temporary files and branches."""
        self.log("Cleaning up temporary files and branches")
        
        try:
            # Clean up any temporary directories
            temp_dirs = ["temp", "tmp", ".temp"]
            for temp_dir in temp_dirs:
                temp_path = self.repo_root / temp_dir
                if temp_path.exists():
                    import shutil
                    shutil.rmtree(temp_path)
                    self.log(f"Removed temporary directory: {temp_path}")
            
            # Clean up any untracked files
            return_code, stdout, stderr = self.run_command(["git", "clean", "-fd"])
            if return_code == 0:
                self.log("Cleaned up untracked files")
            else:
                self.log(f"Error cleaning up files: {stderr}", "WARN")
            
            self.log("Cleanup completed successfully")
            return True
            
        except Exception as e:
            self.log(f"Error during cleanup: {e}", "ERROR")
            return False


def main():
    """Main entry point for the branch manager."""
    parser = argparse.ArgumentParser(
        description="Advanced branch manager for template-heaven stack branches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/branch_manager.py list
  python scripts/branch_manager.py create frontend --description "Frontend templates"
  python scripts/branch_manager.py validate backend
  python scripts/branch_manager.py sync fullstack
  python scripts/branch_manager.py status ai-ml
  python scripts/branch_manager.py report
  python scripts/branch_manager.py cleanup
        """
    )
    
    parser.add_argument("command", choices=["list", "create", "validate", "sync", "status", "report", "cleanup"],
                       help="Command to execute")
    parser.add_argument("stack_name", nargs="?", help="Stack name (required for create, validate, sync, status)")
    parser.add_argument("--description", help="Description for new stack")
    parser.add_argument("--maintainers", help="Comma-separated list of maintainers")
    parser.add_argument("--force", action="store_true", help="Force operation even if branch exists")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Initialize branch manager
    manager = BranchManager(verbose=args.verbose)
    
    # Execute command
    if args.command == "list":
        manager.list_branches()
    
    elif args.command == "create":
        if not args.stack_name:
            print("Error: Stack name is required for create command")
            sys.exit(1)
        success = manager.create_branch(
            args.stack_name,
            args.description or "",
            args.maintainers or "",
            args.force
        )
        if not success:
            sys.exit(1)
    
    elif args.command == "validate":
        if not args.stack_name:
            print("Error: Stack name is required for validate command")
            sys.exit(1)
        success = manager.validate_branch(args.stack_name)
        if not success:
            sys.exit(1)
    
    elif args.command == "sync":
        if not args.stack_name:
            print("Error: Stack name is required for sync command")
            sys.exit(1)
        success = manager.sync_core_tools(args.stack_name)
        if not success:
            sys.exit(1)
    
    elif args.command == "status":
        if not args.stack_name:
            print("Error: Stack name is required for status command")
            sys.exit(1)
        manager.show_branch_status(args.stack_name)
    
    elif args.command == "report":
        manager.generate_report()
    
    elif args.command == "cleanup":
        success = manager.cleanup()
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
