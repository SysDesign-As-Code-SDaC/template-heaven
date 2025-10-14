#!/usr/bin/env python3
"""
Template Heaven Simple Demo

This script demonstrates how Template Heaven works without Unicode characters.
"""

import sys
from pathlib import Path

# Add the package to the path for demo purposes
sys.path.insert(0, str(Path(__file__).parent))

from templateheaven import TemplateManager, Config


def main():
    """Run a simple demo."""
    print("Template Heaven Demo")
    print("=" * 50)
    
    try:
        # Initialize components
        config = Config()
        manager = TemplateManager(config)
        
        # List all templates
        print("\nAll Available Templates:")
        all_templates = manager.list_templates()
        for template in all_templates[:5]:  # Show first 5
            print(f"  - {template.name} ({template.stack.value})")
            print(f"    {template.description}")
        
        print(f"\nTotal templates available: {len(all_templates)}")
        
        # List templates by stack
        print("\nFrontend Templates:")
        frontend_templates = manager.list_templates(stack="frontend")
        for template in frontend_templates:
            print(f"  - {template.name}")
            print(f"    Tags: {', '.join(template.tags[:3])}")
        
        # Search templates
        print("\nSearch Results for 'react':")
        search_results = manager.search_templates("react", limit=3)
        for result in search_results:
            print(f"  - {result.template.name} (score: {result.score:.2f})")
            print(f"    Match reason: {result.match_reason}")
        
        # Get template statistics
        stats = manager.get_template_stats()
        print(f"\nTemplate Statistics:")
        print(f"  Total Templates: {stats['total_templates']}")
        print(f"  Stacks: {len(stats['stacks'])}")
        print(f"  Unique Tags: {len(stats['tags'])}")
        
        print("\nDemo Complete!")
        print("Template Heaven is working perfectly!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
