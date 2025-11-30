"""
Authentication and Authorization

JWT-based authentication system with role-based access control
and API key support for production deployment.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from dataclasses import dataclass
from enum import Enum

from .config import get_settings

logger = structlog.get_logger()
security = HTTPBearer()


class UserRole(Enum):
    """User role enumeration."""
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class User:
    """User data container."""
    user_id: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    api_quota: int = 100  # Default daily quota
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AuthenticationError(Exception):
    """Custom authentication exception."""
    pass


class AuthorizationError(Exception):
    """Custom authorization exception."""
    pass


class JWTManager:
    """
    JWT token management with role-based access control.
    
    Features:
    - Token generation and validation
    - Role-based permissions
    - Token refresh
    - Blacklist support
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.security.jwt_secret_key
        self.algorithm = self.settings.security.jwt_algorithm
        self.expiration_hours = self.settings.security.jwt_expiration_hours
        
        # In production, use Redis for blacklist
        self.blacklisted_tokens = set()
    
    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        
        payload = {
            "sub": user_id,
            "email": email,
            "role": role.value,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info("Created access token", user_id=user_id, expires=expire)
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        
        expire = datetime.utcnow() + timedelta(days=30)  # Longer expiration
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token."""
        
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise AuthenticationError("Token has been revoked")
            
            # Decode token
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        
        try:
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")
            
            user_id = payload.get("sub")
            
            # TODO: Get user details from database
            # For now, create token with basic role
            new_token = self.create_access_token(
                user_id=user_id,
                email="",  # Would fetch from DB
                role=UserRole.USER
            )
            
            return new_token
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid refresh token")
    
    def revoke_token(self, token: str):
        """Add token to blacklist."""
        self.blacklisted_tokens.add(token)
        logger.info("Token revoked")
    
    def is_token_valid(self, token: str) -> bool:
        """Check if token is valid without raising exceptions."""
        try:
            self.verify_token(token)
            return True
        except AuthenticationError:
            return False


class PasswordManager:
    """Password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            hashed.encode('utf-8')
        )


class PermissionManager:
    """Role-based permission management."""
    
    # Define permissions for each role
    ROLE_PERMISSIONS = {
        UserRole.USER: [
            "generate_image",
            "view_own_jobs",
            "cancel_own_jobs"
        ],
        UserRole.PREMIUM: [
            "generate_image",
            "view_own_jobs",
            "cancel_own_jobs",
            "high_quality_generation",
            "priority_processing",
            "extended_quota"
        ],
        UserRole.ADMIN: [
            "*"  # All permissions
        ]
    }
    
    @classmethod
    def has_permission(cls, role: UserRole, permission: str) -> bool:
        """Check if role has specific permission."""
        
        role_perms = cls.ROLE_PERMISSIONS.get(role, [])
        
        # Admin has all permissions
        if "*" in role_perms:
            return True
        
        return permission in role_perms
    
    @classmethod
    def require_permission(cls, permission: str):
        """Decorator to require specific permission."""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be implemented with FastAPI dependencies
                # For now, just a placeholder
                return func(*args, **kwargs)
            return wrapper
        return decorator


class APIKeyManager:
    """API key management for programmatic access."""
    
    def __init__(self):
        # In production, store in database
        self.api_keys = {}  # api_key -> user_info
    
    def generate_api_key(self, user_id: str, name: str) -> str:
        """Generate new API key for user."""
        
        import secrets
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow(),
            "last_used": None,
            "is_active": True
        }
        
        logger.info("Generated API key", user_id=user_id, name=name)
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verify API key and return user info."""
        
        key_info = self.api_keys.get(api_key)
        
        if not key_info or not key_info["is_active"]:
            return None
        
        # Update last used
        key_info["last_used"] = datetime.utcnow()
        
        return key_info
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False
            logger.info("Revoked API key", api_key=api_key[:10] + "...")
            return True
        
        return False


# Global instances
jwt_manager = JWTManager()
password_manager = PasswordManager()
permission_manager = PermissionManager()
api_key_manager = APIKeyManager()


# FastAPI Dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """Get current authenticated user from JWT token."""
    
    try:
        token = credentials.credentials
        payload = jwt_manager.verify_token(token)
        return payload
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))


async def get_current_active_user(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Get current active user (additional checks can be added here)."""
    
    # In production, check if user is active in database
    return current_user


async def require_role(required_role: UserRole):
    """Dependency to require specific role."""
    
    def role_checker(current_user: Dict = Depends(get_current_user)):
        user_role = UserRole(current_user.get("role", "user"))
        
        # Admin can access everything
        if user_role == UserRole.ADMIN:
            return current_user
        
        if user_role.value < required_role.value:
            raise HTTPException(
                status_code=403, 
                detail="Insufficient permissions"
            )
        
        return current_user
    
    return role_checker


async def verify_api_key_or_jwt(
    api_key: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict:
    """Verify either API key or JWT token."""
    
    settings = get_settings()
    
    # Check API key first (if provided in header)
    if api_key:
        key_info = api_key_manager.verify_api_key(api_key)
        if key_info:
            return {
                "sub": key_info["user_id"],
                "auth_type": "api_key",
                "role": "user"  # Default role for API keys
            }
    
    # Fall back to JWT
    if credentials:
        return await get_current_user(credentials)
    
    raise HTTPException(status_code=401, detail="Authentication required")


# Utility functions
def verify_token(token: str) -> Dict:
    """Verify JWT token (for use outside FastAPI)."""
    return jwt_manager.verify_token(token)


def create_user_token(user: User) -> str:
    """Create token for user object."""
    return jwt_manager.create_access_token(
        user_id=user.user_id,
        email=user.email,
        role=user.role
    )


def hash_password(password: str) -> str:
    """Hash password."""
    return password_manager.hash_password(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify password."""
    return password_manager.verify_password(password, hashed)


# Example usage
if __name__ == "__main__":
    # Test JWT creation and verification
    user = User(
        user_id="user123",
        email="user@example.com",
        role=UserRole.USER
    )
    
    token = create_user_token(user)
    print(f"Created token: {token[:50]}...")
    
    # Verify token
    payload = verify_token(token)
    print(f"Token payload: {payload}")
    
    # Test password hashing
    password = "secure_password"
    hashed = hash_password(password)
    print(f"Hashed password: {hashed}")
    
    # Verify password
    is_valid = verify_password(password, hashed)
    print(f"Password valid: {is_valid}")
    
    # Test permissions
    can_generate = permission_manager.has_permission(UserRole.USER, "generate_image")
    print(f"User can generate images: {can_generate}")
    
    can_admin = permission_manager.has_permission(UserRole.USER, "admin_panel")
    print(f"User can access admin: {can_admin}")