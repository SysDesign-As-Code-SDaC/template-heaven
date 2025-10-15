"""
User Service

Business logic for user management, authentication, and profile operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from ..core.config import get_settings
from ..core.security import (
    hash_password,
    verify_password,
    create_token_pair,
    generate_verification_token,
    generate_password_reset_token,
    generate_session_token
)
from ..models.user import User, UserSession, PasswordResetToken, EmailVerificationToken
from ..schemas.user import (
    UserCreate,
    UserUpdate,
    UserAdminUpdate,
    PasswordResetRequest,
    PasswordResetConfirm,
    ChangePasswordRequest,
    LoginRequest
)

logger = logging.getLogger(__name__)
settings = get_settings()


class UserService:
    """Service class for user-related operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        stmt = select(User).where(User.username == username)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_by_username_or_email(self, username_or_email: str) -> Optional[User]:
        """Get user by username or email."""
        stmt = select(User).where(
            or_(User.username == username_or_email, User.email == username_or_email)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise ValueError("User with this email already exists")

        existing_username = await self.get_user_by_username(user_data.username)
        if existing_username:
            raise ValueError("Username already taken")

        # Create user
        hashed_password = hash_password(user_data.password)
        user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            bio=user_data.bio,
            website=user_data.website,
            location=user_data.location,
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        logger.info(f"User created: {user.username} ({user.id})")

        # Create email verification token
        await self.create_email_verification_token(user.id)

        return user

    async def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None

        # Check email uniqueness if being changed
        if user_data.email and user_data.email != user.email:
            existing = await self.get_user_by_email(user_data.email)
            if existing:
                raise ValueError("Email already in use")

        # Update fields
        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        user.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(user)

        logger.info(f"User updated: {user.username} ({user.id})")
        return user

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        # Soft delete by deactivating
        user.is_active = False
        user.updated_at = datetime.utcnow()

        await self.db.commit()

        logger.info(f"User deactivated: {user.username} ({user.id})")
        return True

    async def authenticate_user(self, login_data: LoginRequest) -> Optional[User]:
        """Authenticate user with username/email and password."""
        user = await self.get_user_by_username_or_email(login_data.username_or_email)
        if not user:
            return None

        if not user.is_active:
            return None

        if not verify_password(login_data.password, user.hashed_password):
            return None

        # Update last login
        user.update_last_login()
        await self.db.commit()

        logger.info(f"User authenticated: {user.username} ({user.id})")
        return user

    async def change_password(self, user_id: int, password_data: ChangePasswordRequest) -> bool:
        """Change user password."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        # Verify current password
        if not verify_password(password_data.current_password, user.hashed_password):
            raise ValueError("Current password is incorrect")

        # Update password
        user.hashed_password = hash_password(password_data.new_password)
        user.updated_at = datetime.utcnow()

        await self.db.commit()

        logger.info(f"Password changed for user: {user.username} ({user.id})")
        return True

    async def reset_password_request(self, reset_data: PasswordResetRequest) -> bool:
        """Initiate password reset process."""
        user = await self.get_user_by_email(reset_data.email)
        if not user:
            # Don't reveal if email exists for security
            return True

        # Create password reset token
        token = generate_password_reset_token()
        expires_at = datetime.utcnow() + timedelta(hours=24)

        reset_token = PasswordResetToken(
            user_id=user.id,
            token=token,
            expires_at=expires_at
        )

        self.db.add(reset_token)
        await self.db.commit()

        # TODO: Send email with reset token
        logger.info(f"Password reset token created for user: {user.username} ({user.id})")

        return True

    async def reset_password_confirm(self, reset_data: PasswordResetConfirm) -> bool:
        """Confirm password reset with token."""
        # Find valid reset token
        stmt = select(PasswordResetToken).where(
            and_(
                PasswordResetToken.token == reset_data.token,
                PasswordResetToken.used == False,
                PasswordResetToken.expires_at > datetime.utcnow()
            )
        )
        result = await self.db.execute(stmt)
        token_record = result.scalar_one_or_none()

        if not token_record:
            raise ValueError("Invalid or expired reset token")

        # Get user
        user = await self.get_user_by_id(token_record.user_id)
        if not user:
            raise ValueError("User not found")

        # Update password
        user.hashed_password = hash_password(reset_data.new_password)
        user.updated_at = datetime.utcnow()

        # Mark token as used
        token_record.used = True

        await self.db.commit()

        logger.info(f"Password reset completed for user: {user.username} ({user.id})")
        return True

    async def create_email_verification_token(self, user_id: int) -> str:
        """Create email verification token."""
        token = generate_verification_token()
        expires_at = datetime.utcnow() + timedelta(days=7)

        verification_token = EmailVerificationToken(
            user_id=user_id,
            token=token,
            expires_at=expires_at
        )

        self.db.add(verification_token)
        await self.db.commit()

        return token

    async def verify_email(self, token: str) -> bool:
        """Verify user email with token."""
        # Find valid verification token
        stmt = select(EmailVerificationToken).where(
            and_(
                EmailVerificationToken.token == token,
                EmailVerificationToken.used == False,
                EmailVerificationToken.expires_at > datetime.utcnow()
            )
        )
        result = await self.db.execute(stmt)
        token_record = result.scalar_one_or_none()

        if not token_record:
            raise ValueError("Invalid or expired verification token")

        # Get user
        user = await self.get_user_by_id(token_record.user_id)
        if not user:
            raise ValueError("User not found")

        # Verify email
        user.verify_email()

        # Mark token as used
        token_record.used = True

        await self.db.commit()

        logger.info(f"Email verified for user: {user.username} ({user.id})")
        return True

    async def create_session(self, user_id: int, ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> str:
        """Create a new user session."""
        session_token = generate_session_token()
        expires_at = datetime.utcnow() + timedelta(days=30)  # 30 days

        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )

        self.db.add(session)
        await self.db.commit()

        return session_token

    async def get_session(self, session_token: str) -> Optional[UserSession]:
        """Get session by token."""
        stmt = select(UserSession).where(
            and_(
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a user session."""
        session = await self.get_session(session_token)
        if not session:
            return False

        session.deactivate()
        await self.db.commit()

        return True

    async def get_users_list(self, page: int = 1, per_page: int = 20,
                           search: Optional[str] = None) -> Dict[str, Any]:
        """Get paginated list of users."""
        offset = (page - 1) * per_page

        # Build query
        query = select(User)

        if search:
            search_filter = f"%{search}%"
            query = query.where(
                or_(
                    User.username.ilike(search_filter),
                    User.email.ilike(search_filter),
                    User.full_name.ilike(search_filter)
                )
            )

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()

        # Get paginated results
        query = query.offset(offset).limit(per_page).order_by(User.created_at.desc())
        result = await self.db.execute(query)
        users = result.scalars().all()

        return {
            "users": users,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }

    async def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        # Total users
        total_result = await self.db.execute(select(func.count(User.id)))
        total_users = total_result.scalar()

        # Active users
        active_result = await self.db.execute(
            select(func.count(User.id)).where(User.is_active == True)
        )
        active_users = active_result.scalar()

        # Verified users
        verified_result = await self.db.execute(
            select(func.count(User.id)).where(User.email_verified == True)
        )
        verified_users = verified_result.scalar()

        # Superusers
        superuser_result = await self.db.execute(
            select(func.count(User.id)).where(User.is_superuser == True)
        )
        superusers = superuser_result.scalar()

        # Recent registrations (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_result = await self.db.execute(
            select(func.count(User.id)).where(User.created_at >= thirty_days_ago)
        )
        recent_registrations = recent_result.scalar()

        # Login activity (last 24 hours)
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        login_result = await self.db.execute(
            select(func.count(User.id)).where(User.last_login >= one_day_ago)
        )
        login_activity = login_result.scalar()

        return {
            "total_users": total_users,
            "active_users": active_users,
            "verified_users": verified_users,
            "superusers": superusers,
            "recent_registrations": recent_registrations,
            "login_activity": login_activity
        }

    async def admin_update_user(self, user_id: int, update_data: UserAdminUpdate) -> Optional[User]:
        """Admin update of user (full access)."""
        user = await self.get_user_by_id(user_id)
        if not user:
            return None

        # Check uniqueness constraints
        if update_data.email and update_data.email != user.email:
            existing = await self.get_user_by_email(update_data.email)
            if existing:
                raise ValueError("Email already in use")

        if update_data.username and update_data.username != user.username:
            existing = await self.get_user_by_username(update_data.username)
            if existing:
                raise ValueError("Username already taken")

        # Update all fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(user, field, value)

        user.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(user)

        logger.info(f"Admin updated user: {user.username} ({user.id})")
        return user

    async def bulk_deactivate_users(self, user_ids: List[int]) -> int:
        """Bulk deactivate users."""
        stmt = (
            update(User)
            .where(User.id.in_(user_ids))
            .values(is_active=False, updated_at=datetime.utcnow())
        )

        result = await self.db.execute(stmt)
        await self.db.commit()

        deactivated_count = result.rowcount
        logger.info(f"Bulk deactivated {deactivated_count} users")

        return deactivated_count

    async def cleanup_expired_tokens(self) -> Dict[str, int]:
        """Clean up expired tokens and sessions."""
        now = datetime.utcnow()

        # Clean up expired sessions
        session_stmt = delete(UserSession).where(UserSession.expires_at < now)
        session_result = await self.db.execute(session_stmt)

        # Clean up expired password reset tokens
        reset_stmt = delete(PasswordResetToken).where(
            or_(PasswordResetToken.expires_at < now, PasswordResetToken.used == True)
        )
        reset_result = await self.db.execute(reset_stmt)

        # Clean up expired email verification tokens
        verification_stmt = delete(EmailVerificationToken).where(
            or_(EmailVerificationToken.expires_at < now, EmailVerificationToken.used == True)
        )
        verification_result = await self.db.execute(verification_stmt)

        await self.db.commit()

        return {
            "expired_sessions": session_result.rowcount,
            "expired_reset_tokens": reset_result.rowcount,
            "expired_verification_tokens": verification_result.rowcount
        }
