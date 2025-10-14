#!/usr/bin/env python3
"""
Simple test script to verify Template Heaven functionality.
"""

import asyncio
import sys
import os
sys.path.insert(0, '.')

from templateheaven.database.connection import init_database, close_database
from templateheaven.services.template_service import template_service


async def test_basic_functionality():
    """Test basic functionality."""
    print("Testing Template Heaven Basic Functionality")
    print("=" * 50)
    
    try:
        # Test database initialization
        print("1. Testing database initialization...")
        await init_database()
        print("   SUCCESS: Database initialized")
        
        # Test template service
        print("2. Testing template service...")
        template_data = {
            "name": "test-template",
            "stack": "frontend",
            "description": "A test template",
            "technologies": ["react", "typescript"],
            "tags": ["test"],
            "stars": 50,
            "forks": 10,
            "quality_score": 0.8
        }
        
        template = await template_service.create_template(template_data)
        print(f"   SUCCESS: Created template '{template.name}'")
        
        # Test listing templates
        templates, total = await template_service.list_templates(limit=5)
        print(f"   SUCCESS: Listed {len(templates)} templates (total: {total})")
        
        # Test search
        search_results = await template_service.search_templates("test", limit=5)
        print(f"   SUCCESS: Search found {len(search_results)} templates")
        
        print("\nAll tests passed! Template Heaven is working.")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            await close_database()
        except:
            pass


if __name__ == "__main__":
    try:
        success = asyncio.run(test_basic_functionality())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest runner crashed: {e}")
        sys.exit(1)
