"""
Authentication service for Template Heaven.

This module provides authentication, authorization, and user management
functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database.models import User, Role, UserRole, APIKey
from ..database.connection import get_db_session
from ..utils.logger import get_logger
from ..api.dependencies import get_settings

logger = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


class AuthService:
    """Service for authentication and authorization operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.secret_key
        self.algorithm = self.settings.algorithm
        self.access_token_expire_minutes = self.settings.access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        # Truncate password to 72 bytes to avoid bcrypt limitation
        password = password[:72]
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(User).options(
                selectinload(User.roles).selectinload(UserRole.role)
            ).where(
                and_(User.username == username, User.is_active == True)
            )
            
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            if not self.verify_password(password, user.hashed_password):
                return None
            
            return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(User).options(
                selectinload(User.roles).selectinload(UserRole.role)
            ).where(User.id == user_id)
            
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(User).options(
                selectinload(User.roles).selectinload(UserRole.role)
            ).where(User.username == username)
            
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(User).options(
                selectinload(User.roles).selectinload(UserRole.role)
            ).where(User.email == email)
            
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def create_user(
        self, 
        username: str, 
        email: str, 
        password: str, 
        full_name: Optional[str] = None,
        roles: Optional[List[str]] = None
    ) -> User:
        """Create a new user."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            # Check if user already exists
            existing_user = await self.get_user_by_username(username)
            if existing_user:
                raise ValueError("Username already exists")
            
            existing_email = await self.get_user_by_email(email)
            if existing_email:
                raise ValueError("Email already exists")
            
            # Create user
            hashed_password = self.get_password_hash(password)
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name
            )
            
            session.add(user)
            await session.flush()
            
            # Assign roles
            if roles:
                await self._assign_roles(session, user.id, roles)
            
            # Load user with roles
            await session.refresh(user, ["roles"])
            
            return user
    
    async def update_user(
        self, 
        user_id: str, 
        update_data: Dict[str, Any]
    ) -> Optional[User]:
        """Update a user."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(User).where(User.id == user_id)
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            # Update fields
            for field, value in update_data.items():
                if hasattr(user, field) and value is not None:
                    if field == "password":
                        user.hashed_password = self.get_password_hash(value)
                    else:
                        setattr(user, field, value)
            
            user.updated_at = datetime.utcnow()
            await session.flush()
            
            return user
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user (soft delete)."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(User).where(User.id == user_id)
            result = await session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                return False
            
            user.is_active = False
            user.updated_at = datetime.utcnow()
            
            await session.flush()
            return True
    
    async def assign_roles(self, user_id: str, roles: List[str]) -> bool:
        """Assign roles to a user."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            return await self._assign_roles(session, user_id, roles)
    
    async def _assign_roles(self, session: AsyncSession, user_id: str, roles: List[str]) -> bool:
        """Internal method to assign roles."""
        # Get role IDs
        role_query = select(Role).where(Role.name.in_(roles))
        role_result = await session.execute(role_query)
        role_models = role_result.scalars().all()
        
        if not role_models:
            return False
        
        # Remove existing roles
        delete_query = select(UserRole).where(UserRole.user_id == user_id)
        delete_result = await session.execute(delete_query)
        existing_roles = delete_result.scalars().all()
        
        for existing_role in existing_roles:
            await session.delete(existing_role)
        
        # Add new roles
        for role_model in role_models:
            user_role = UserRole(
                user_id=user_id,
                role_id=role_model.id
            )
            session.add(user_role)
        
        await session.flush()
        return True
    
    async def get_user_roles(self, user_id: str) -> List[str]:
        """Get user roles."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(Role.name).join(UserRole).where(UserRole.user_id == user_id)
            result = await session.execute(query)
            return [row[0] for row in result.fetchall()]
    
    async def has_role(self, user_id: str, role: str) -> bool:
        """Check if user has a specific role."""
        roles = await self.get_user_roles(user_id)
        return role in roles
    
    async def has_any_role(self, user_id: str, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        user_roles = await self.get_user_roles(user_id)
        return any(role in user_roles for role in roles)
    
    async def create_api_key(
        self, 
        user_id: str, 
        name: str, 
        expires_at: Optional[datetime] = None
    ) -> str:
        """Create an API key for a user."""
        import secrets
        
        # Generate API key
        api_key = f"th_{secrets.token_urlsafe(32)}"
        key_hash = self.get_password_hash(api_key)
        
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            api_key_model = APIKey(
                user_id=user_id,
                name=name,
                key_hash=key_hash,
                expires_at=expires_at
            )
            
            session.add(api_key_model)
            await session.flush()
            
            return api_key
    
    async def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify an API key and return the associated user."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(APIKey).options(
                selectinload(APIKey.user).selectinload(User.roles).selectinload(UserRole.role)
            ).where(
                and_(
                    APIKey.is_active == True,
                    APIKey.expires_at > datetime.utcnow()
                )
            )
            
            result = await session.execute(query)
            api_keys = result.scalars().all()
            
            for api_key_model in api_keys:
                if self.verify_password(api_key, api_key_model.key_hash):
                    # Update last used
                    api_key_model.last_used = datetime.utcnow()
                    await session.flush()
                    
                    return api_key_model.user
            
            return None
    
    async def revoke_api_key(self, api_key_id: str) -> bool:
        """Revoke an API key."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(APIKey).where(APIKey.id == api_key_id)
            result = await session.execute(query)
            api_key = result.scalar_one_or_none()
            
            if not api_key:
                return False
            
            api_key.is_active = False
            await session.flush()
            
            return True
    
    async def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        from ..database.connection import db_manager
        async with db_manager.get_session() as session:
            query = select(APIKey).where(
                and_(APIKey.user_id == user_id, APIKey.is_active == True)
            ).order_by(APIKey.created_at.desc())
            
            result = await session.execute(query)
            return result.scalars().all()
    
    def _model_to_pydantic(self, user_model: User) -> UserModel:
        """Convert SQLAlchemy model to Pydantic model."""
        roles = []
        for user_role in user_model.roles:
            role_name = user_role.role.name
            try:
                role_enum = UserRoleEnum(role_name)
                roles.append(role_enum)
            except ValueError:
                # Handle custom roles
                pass
        
        return UserModel(
            id=str(user_model.id),
            username=user_model.username,
            email=user_model.email,
            full_name=user_model.full_name,
            is_active=user_model.is_active,
            is_superuser=user_model.is_superuser,
            roles=roles,
            created_at=user_model.created_at,
            updated_at=user_model.updated_at
        )


# Global service instance
auth_service = AuthService()
