#!/usr/bin/env python3
"""
Manual verification script for Template Heaven system.
"""

import sys
import os
sys.path.insert(0, '.')

def test_imports():
    """Test core module imports."""
    print("🔍 Testing Core Module Imports...")
    
    try:
        from templateheaven.core.models import Template, StackCategory, APIResponse
        print("✅ Core models imported successfully")
        
        from templateheaven.api.main import app
        print("✅ FastAPI app imported successfully")
        
        from templateheaven.api.dependencies import get_settings
        print("✅ API dependencies imported successfully")
        
        from templateheaven.api.routes import health, templates, search, stacks, populate, auth
        print("✅ API routes imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_models():
    """Test Pydantic model validation."""
    print("\n🔍 Testing Pydantic Model Validation...")
    
    try:
        from templateheaven.core.models import Template, StackCategory, APIResponse
        
        # Test Template model
        template = Template(
            name='test-template',
            description='A test template',
            stack=StackCategory.FRONTEND,
            path='/test/path',
            stars=100,
            forks=10,
            quality_score=0.85
        )
        print("✅ Template model validation passed")
        
        # Test APIResponse model
        response = APIResponse(
            success=True,
            message='Test response',
            data={'test': 'data'}
        )
        print("✅ APIResponse model validation passed")
        
        return True
    except Exception as e:
        print(f"❌ Model validation error: {e}")
        return False

def test_settings():
    """Test settings configuration."""
    print("\n🔍 Testing Settings Configuration...")
    
    try:
        from templateheaven.api.dependencies import get_settings
        
        settings = get_settings()
        print(f"✅ Settings loaded: {settings.app_name}")
        print(f"   - Version: {settings.app_version}")
        print(f"   - Debug: {settings.debug}")
        print(f"   - Host: {settings.host}:{settings.port}")
        
        return True
    except Exception as e:
        print(f"❌ Settings error: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI application."""
    print("\n🔍 Testing FastAPI Application...")
    
    try:
        from templateheaven.api.main import app
        
        # Check if app is properly configured
        if hasattr(app, 'routes'):
            print(f"✅ FastAPI app has {len(app.routes)} routes")
        
        # Check for key routes
        route_paths = [route.path for route in app.routes]
        expected_routes = ['/', '/api/v1/health', '/docs', '/openapi.json']
        
        for expected_route in expected_routes:
            if expected_route in route_paths:
                print(f"✅ Route {expected_route} found")
            else:
                print(f"❌ Route {expected_route} missing")
        
        return True
    except Exception as e:
        print(f"❌ FastAPI app error: {e}")
        return False

def test_docker_files():
    """Test Docker configuration files."""
    print("\n🔍 Testing Docker Configuration...")
    
    docker_files = [
        'Dockerfile',
        'docker-compose.yml',
        'nginx/nginx.conf',
        'monitoring/prometheus.yml'
    ]
    
    all_exist = True
    for file_path in docker_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_requirements():
    """Test requirements file."""
    print("\n🔍 Testing Requirements...")
    
    if os.path.exists('requirements.txt'):
        print("✅ requirements.txt exists")
        
        # Check for key dependencies
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        key_deps = ['fastapi', 'uvicorn', 'pydantic', 'redis', 'sqlite3']
        for dep in key_deps:
            if dep in content:
                print(f"✅ {dep} dependency found")
            else:
                print(f"❌ {dep} dependency missing")
        
        return True
    else:
        print("❌ requirements.txt missing")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Template Heaven System Manual Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_models,
        test_settings,
        test_fastapi_app,
        test_docker_files,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
