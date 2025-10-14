#!/usr/bin/env python3
"""
Test script to verify Template Heaven functionality.

This script tests the core functionality of the Template Heaven system
including database operations, services, and API endpoints.
"""

import asyncio
import sys
import os
sys.path.insert(0, '.')

from templateheaven.database.connection import init_database, close_database
from templateheaven.services.template_service import template_service
from templateheaven.services.auth_service import auth_service
from templateheaven.core.models import Template, StackCategory


async def test_database_connection():
    """Test database connection and table creation."""
    print("Testing database connection...")
    try:
        await init_database()
        print("SUCCESS: Database initialized successfully")
        return True
    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}")
        return False


async def test_template_service():
    """Test template service functionality."""
    print("\nTesting template service...")
    try:
        # Test creating a template
        template_data = {
            "name": "test-react-app",
            "stack": "frontend",
            "description": "A test React application template",
            "technologies": ["react", "typescript", "vite"],
            "tags": ["frontend", "react", "test"],
            "stars": 100,
            "forks": 20,
            "quality_score": 0.85
        }
        
        template = await template_service.create_template(template_data)
        print(f"SUCCESS: Created template: {template.name}")
        
        # Test listing templates
        templates, total = await template_service.list_templates(limit=10)
        print(f"SUCCESS: Listed {len(templates)} templates (total: {total})")
        
        # Test getting template
        retrieved_template = await template_service.get_template(template.id)
        if retrieved_template:
            print(f"SUCCESS: Retrieved template: {retrieved_template.name}")
        else:
            print("ERROR: Failed to retrieve template")
            return False
        
        # Test searching templates
        search_results = await template_service.search_templates("react", limit=5)
        print(f"SUCCESS: Search found {len(search_results)} templates")
        
        return True
    except Exception as e:
        print(f"ERROR: Template service test failed: {e}")
        return False


async def test_auth_service():
    """Test authentication service functionality."""
    print("\n🔍 Testing authentication service...")
    try:
        # Test creating a user
        user = await auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
            full_name="Test User",
            roles=["user"]
        )
        print(f"✅ Created user: {user.username}")
        
        # Test authenticating user
        authenticated_user = await auth_service.authenticate_user("testuser", "testpassword123")
        if authenticated_user:
            print(f"✅ Authenticated user: {authenticated_user.username}")
        else:
            print("❌ Authentication failed")
            return False
        
        # Test creating access token
        token_data = {
            "sub": user.username,
            "user_id": str(user.id),
            "roles": ["user"]
        }
        token = auth_service.create_access_token(token_data)
        print("✅ Created access token")
        
        # Test verifying token
        payload = auth_service.verify_token(token)
        if payload and payload.get("sub") == user.username:
            print("✅ Token verification successful")
        else:
            print("❌ Token verification failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Authentication service test failed: {e}")
        return False


async def test_api_endpoints():
    """Test API endpoints using TestClient."""
    print("\n🔍 Testing API endpoints...")
    try:
        from fastapi.testclient import TestClient
        from templateheaven.api.main_new import app
        
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
        
        # Test health endpoint
        response = client.get("/api/v1/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test templates endpoint
        response = client.get("/api/v1/templates")
        if response.status_code == 200:
            print("✅ Templates endpoint working")
        else:
            print(f"❌ Templates endpoint failed: {response.status_code}")
            return False
        
        # Test search endpoint
        response = client.post("/api/v1/search", json={"query": "react"})
        if response.status_code == 200:
            print("✅ Search endpoint working")
        else:
            print(f"❌ Search endpoint failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("Template Heaven Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Template Service", test_template_service),
        ("Authentication Service", test_auth_service),
        ("API Endpoints", test_api_endpoints),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Template Heaven is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner crashed: {e}")
        sys.exit(1)
    finally:
        # Clean up
        try:
            asyncio.run(close_database())
        except:
            pass
