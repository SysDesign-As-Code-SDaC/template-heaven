#!/usr/bin/env python3
"""
Quick Template Heaven Demo

This script demonstrates the Python API usage of Template Heaven.
"""

from templateheaven import TemplateManager, Config

def main():
    print("Template Heaven Python API Demo")
    print("=" * 40)
    
    # Initialize components
    config = Config()
    manager = TemplateManager(config)
    
    # List all templates
    print("\n1. All Available Templates:")
    all_templates = manager.list_templates()
    for template in all_templates[:3]:  # Show first 3
        print(f"   - {template.name} ({template.stack.value})")
        print(f"     {template.description}")
    
    # List templates by stack
    print("\n2. Frontend Templates:")
    frontend_templates = manager.list_templates(stack="frontend")
    for template in frontend_templates:
        print(f"   - {template.name}")
        print(f"     Tags: {', '.join(template.tags[:3])}")
    
    # Search templates
    print("\n3. Search Results for 'react':")
    search_results = manager.search_templates("react", limit=2)
    for result in search_results:
        print(f"   - {result.template.name} (score: {result.score:.2f})")
        print(f"     Match: {result.match_reason}")
    
    # Get template statistics
    stats = manager.get_template_stats()
    print(f"\n4. Template Statistics:")
    print(f"   Total Templates: {stats['total_templates']}")
    print(f"   Stacks: {len(stats['stacks'])}")
    print(f"   Unique Tags: {len(stats['tags'])}")
    
    print("\nDemo Complete!")
    print("Template Heaven is working perfectly!")

if __name__ == "__main__":
    main()
