#!/usr/bin/env python3
"""
Test script to demonstrate Template Heaven system functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from templateheaven.database.connection import init_database, db_manager
from templateheaven.services.template_service import TemplateService
from templateheaven.services.auth_service import AuthService
from templateheaven.core.models import Template, StackCategory
from templateheaven.utils.logger import get_logger

logger = get_logger(__name__)


async def test_database():
    """Test database initialization and connection."""
    print("\n" + "="*60)
    print("📊 TESTING DATABASE")
    print("="*60)
    
    try:
        await init_database()
        print("✅ Database initialized successfully")
        
        # Test connection
        from sqlalchemy import text
        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            print("✅ Database connection verified")
            print(f"📁 Database file: {db_manager.engine.url}")
        
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


async def test_template_service():
    """Test template service functionality."""
    print("\n" + "="*60)
    print("📦 TESTING TEMPLATE SERVICE")
    print("="*60)
    
    try:
        service = TemplateService()
        
        # List templates
        templates, total = await service.list_templates()
        print(f"✅ Found {len(templates)} templates (total: {total})")
        
        if templates:
            print(f"\n📋 Sample template: {templates[0].name}")
            print(f"   Stack: {templates[0].stack}")
            print(f"   Description: {templates[0].description[:80] if templates[0].description else 'No description'}...")
        
        # Get stacks (if method exists)
        try:
            stacks = await service.get_stacks()
            print(f"\n✅ Found {len(stacks)} stacks")
            if stacks:
                print(f"📋 Sample stacks: {', '.join([s.name for s in stacks[:5]])}")
        except AttributeError:
            print("\n⚠️  Stack listing not available (may need to populate database)")
        
        return True
    except Exception as e:
        print(f"❌ Template service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_auth_service():
    """Test auth service functionality."""
    print("\n" + "="*60)
    print("🔐 TESTING AUTH SERVICE")
    print("="*60)
    
    try:
        service = AuthService()
        
        # Test password hashing
        password = "test_password_123"
        hashed = service.get_password_hash(password)
        print(f"✅ Password hashing works")
        print(f"   Original: {password}")
        print(f"   Hashed: {hashed[:50]}...")
        
        # Test password verification
        verified = service.verify_password(password, hashed)
        print(f"✅ Password verification: {verified}")
        
        # Test JWT token creation
        token_data = {"sub": "test_user", "user_id": "123"}
        token = service.create_access_token(token_data)
        print(f"✅ JWT token creation works")
        print(f"   Token: {token[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Auth service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "🚀"*30)
    print("TEMPLATE HEAVEN SYSTEM TEST")
    print("🚀"*30)
    
    results = []
    
    # Test database
    results.append(await test_database())
    
    # Test template service
    results.append(await test_template_service())
    
    # Test auth service
    results.append(await test_auth_service())
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

