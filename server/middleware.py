"""
JWT authentication middleware for FastAPI.
Extracts and verifies the Supabase JWT from the Authorization header.
"""
import os
from typing import Optional

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase_client import get_supabase_client

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Verify JWT token via Supabase and return user info.
    Injects user data into route handlers via Depends().
    """
    token = credentials.credentials

    try:
        supabase = get_supabase_client()
        # Use Supabase to verify the token and get user
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="无效的认证令牌")

        user = user_response.user
        return {
            "id": str(user.id),
            "email": user.email,
            "username": user.user_metadata.get("username", user.email.split("@")[0]) if user.email else "unknown",
            "token": token,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"认证失败: {str(e)}")
