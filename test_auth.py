#!/usr/bin/env python3
"""
Test script to verify Template Heaven authentication functionality.
"""

import asyncio
import sys
import os
sys.path.insert(0, '.')

from templateheaven.database.connection import init_database, close_database
from templateheaven.services.auth_service import auth_service


async def test_auth_functionality():
    """Test authentication functionality."""
    print("Testing Template Heaven Authentication")
    print("=" * 40)
    
    try:
        # Test database initialization
        print("1. Testing database initialization...")
        await init_database()
        print("   SUCCESS: Database initialized")
        
        # Test user creation
        print("2. Testing user creation...")
        user = await auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="test123",
            full_name="Test User",
            roles=["user"]
        )
        print(f"   SUCCESS: Created user '{user.username}'")
        
        # Test user authentication
        print("3. Testing user authentication...")
        authenticated_user = await auth_service.authenticate_user("testuser", "test123")
        if authenticated_user:
            print(f"   SUCCESS: Authenticated user '{authenticated_user.username}'")
        else:
            print("   ERROR: Authentication failed")
            return False
        
        # Test token creation
        print("4. Testing token creation...")
        token_data = {
            "sub": user.username,
            "user_id": str(user.id),
            "roles": ["user"]
        }
        token = auth_service.create_access_token(token_data)
        print("   SUCCESS: Created access token")
        
        # Test token verification
        print("5. Testing token verification...")
        payload = auth_service.verify_token(token)
        if payload and payload.get("sub") == user.username:
            print("   SUCCESS: Token verification successful")
        else:
            print("   ERROR: Token verification failed")
            return False
        
        # Test user retrieval
        print("6. Testing user retrieval...")
        retrieved_user = await auth_service.get_user_by_username("testuser")
        if retrieved_user and retrieved_user.username == "testuser":
            print(f"   SUCCESS: Retrieved user '{retrieved_user.username}'")
        else:
            print("   ERROR: User retrieval failed")
            return False
        
        print("\nAll authentication tests passed!")
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
        success = asyncio.run(test_auth_functionality())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest runner crashed: {e}")
        sys.exit(1)
