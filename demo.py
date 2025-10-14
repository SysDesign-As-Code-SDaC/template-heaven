#!/usr/bin/env python3
"""
Template Heaven Demo Script

This script demonstrates how Template Heaven works by showing:
1. Template discovery and listing
2. Template search functionality
3. Project configuration
4. Template customization

Run this script to see Template Heaven in action!
"""

import sys
from pathlib import Path

# Add the package to the path for demo purposes
sys.path.insert(0, str(Path(__file__).parent))

from templateheaven import TemplateManager, Config, Wizard
from templateheaven.core.models import StackCategory


def demo_template_discovery():
    """Demonstrate template discovery and listing."""
    print("🔍 Template Discovery Demo")
    print("=" * 50)
    
    # Initialize components
    config = Config()
    manager = TemplateManager(config)
    
    # List all templates
    print("\n📋 All Available Templates:")
    all_templates = manager.list_templates()
    for template in all_templates[:5]:  # Show first 5
        print(f"  • {template.name} ({template.stack.value})")
        print(f"    {template.description}")
    
    print(f"\nTotal templates available: {len(all_templates)}")
    
    # List templates by stack
    print("\n🎯 Frontend Templates:")
    frontend_templates = manager.list_templates(stack="frontend")
    for template in frontend_templates:
        print(f"  • {template.name}")
        print(f"    Tags: {', '.join(template.tags[:3])}")
    
    # Search templates
    print("\n🔍 Search Results for 'react':")
    search_results = manager.search_templates("react", limit=3)
    for result in search_results:
        print(f"  • {result.template.name} (score: {result.score:.2f})")
        print(f"    Match reason: {result.match_reason}")
    
    return manager, config


def demo_template_info():
    """Demonstrate getting detailed template information."""
    print("\n\n📖 Template Information Demo")
    print("=" * 50)
    
    config = Config()
    manager = TemplateManager(config)
    
    # Get a specific template
    template = manager.get_template("react-vite")
    if template:
        print(f"\n📦 Template: {template.name}")
        print(f"   Stack: {template.stack.value}")
        print(f"   Description: {template.description}")
        print(f"   Version: {template.version}")
        print(f"   Author: {template.author}")
        print(f"   License: {template.license}")
        print(f"   Tags: {', '.join(template.tags)}")
        print(f"   Features: {', '.join(template.features)}")
        print(f"   Dependencies: {template.dependencies}")
        print(f"   Upstream URL: {template.upstream_url}")
    else:
        print("❌ Template not found")


def demo_stack_categories():
    """Demonstrate stack category functionality."""
    print("\n\n🏗️ Stack Categories Demo")
    print("=" * 50)
    
    config = Config()
    manager = TemplateManager(config)
    
    # Get all stacks
    stacks = manager.get_stacks()
    print(f"\n📚 Available Technology Stacks ({len(stacks)}):")
    
    for stack in stacks:
        stack_info = manager.get_stack_info(stack)
        print(f"\n  🎯 {stack_info['name']}")
        print(f"     Description: {stack_info['description']}")
        print(f"     Templates: {stack_info['template_count']}")
        
        # Show first few templates in each stack
        templates = stack_info['templates'][:3]
        for template in templates:
            print(f"       • {template.name}")


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n\n⚙️ Configuration Demo")
    print("=" * 50)
    
    config = Config()
    
    # Show current configuration
    print("\n📋 Current Configuration:")
    print(f"   Cache Directory: {config.get('cache_dir')}")
    print(f"   Default Author: {config.get('default_author')}")
    print(f"   Default License: {config.get('default_license')}")
    print(f"   Log Level: {config.get('log_level')}")
    
    # Show package manager preferences
    print("\n📦 Package Manager Preferences:")
    python_pm = config.get('package_managers.python', 'pip')
    node_pm = config.get('package_managers.node', 'npm')
    print(f"   Python: {python_pm}")
    print(f"   Node.js: {node_pm}")


def demo_template_validation():
    """Demonstrate template validation."""
    print("\n\n✅ Template Validation Demo")
    print("=" * 50)
    
    config = Config()
    manager = TemplateManager(config)
    
    # Get a template and validate it
    template = manager.get_template("react-vite")
    if template:
        is_valid = manager.validate_template(template)
        print(f"\n🔍 Validating template: {template.name}")
        print(f"   Valid: {'✅ Yes' if is_valid else '❌ No'}")
        
        if is_valid:
            print("   ✅ Template passes all validation checks")
        else:
            print("   ❌ Template has validation issues")


def demo_statistics():
    """Demonstrate template statistics."""
    print("\n\n📊 Template Statistics Demo")
    print("=" * 50)
    
    config = Config()
    manager = TemplateManager(config)
    
    # Get template statistics
    stats = manager.get_template_stats()
    print(f"\n📈 Template Statistics:")
    print(f"   Total Templates: {stats['total_templates']}")
    print(f"   Stacks: {len(stats['stacks'])}")
    print(f"   Unique Tags: {len(stats['tags'])}")
    print(f"   Dependencies: {len(stats['dependencies'])}")
    
    # Show most common tags
    print(f"\n🏷️ Most Common Tags:")
    sorted_tags = sorted(stats['tags'].items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags[:10]:
        print(f"   {tag}: {count} templates")
    
    # Show stack distribution
    print(f"\n📚 Stack Distribution:")
    for stack, count in stats['stacks'].items():
        print(f"   {stack}: {count} templates")


def main():
    """Run the complete demo."""
    print("🚀 Template Heaven Demo")
    print("=" * 50)
    print("This demo shows how Template Heaven works!")
    print("=" * 50)
    
    try:
        # Run all demos
        manager, config = demo_template_discovery()
        demo_template_info()
        demo_stack_categories()
        demo_configuration()
        demo_template_validation()
        demo_statistics()
        
        print("\n\n🎉 Demo Complete!")
        print("=" * 50)
        print("Template Heaven is working perfectly!")
        print("\nNext steps:")
        print("1. Try the interactive wizard: templateheaven init")
        print("2. List templates: templateheaven list")
        print("3. Search templates: templateheaven search 'react'")
        print("4. Get help: templateheaven --help")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you're running this from the Template Heaven directory")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
