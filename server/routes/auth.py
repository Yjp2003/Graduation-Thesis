"""
Authentication routes: register, login, logout.
Uses Supabase Auth with username → email mapping (username@yolo-vision.com).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase_client import get_supabase_client, get_supabase_admin

router = APIRouter()

# Virtual email domain for username-based auth
EMAIL_DOMAIN = "yolo-vision.com"


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    token: str | None = None
    username: str | None = None
    user_id: str | None = None


def username_to_email(username: str) -> str:
    """Convert username to virtual email for Supabase Auth."""
    return f"{username}@{EMAIL_DOMAIN}"


@router.post("/register", response_model=AuthResponse)
async def register(req: AuthRequest):
    """Register a new user."""
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="请输入用户名和密码")

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少需要6个字符")

    email = username_to_email(req.username)
    supabase = get_supabase_admin()

    try:
        result = supabase.auth.admin.create_user({
            "email": email,
            "password": req.password,
            "email_confirm": True,
            "user_metadata": {
                "username": req.username,
            }
        })

        if result.user is None:
            raise HTTPException(status_code=400, detail="注册失败")

        return AuthResponse(
            success=True,
            message="注册成功",
            username=req.username,
            user_id=str(result.user.id),
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower() or "already exists" in error_msg.lower():
            raise HTTPException(status_code=400, detail="用户名已存在")
        raise HTTPException(status_code=400, detail=f"注册失败: {error_msg}")


@router.post("/login", response_model=AuthResponse)
async def login(req: AuthRequest):
    """Login with username and password."""
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="请输入用户名和密码")

    email = username_to_email(req.username)
    supabase = get_supabase_client()

    try:
        result = supabase.auth.sign_in_with_password({
            "email": email,
            "password": req.password,
        })

        if result.session is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        return AuthResponse(
            success=True,
            message="登录成功",
            token=result.session.access_token,
            username=req.username,
            user_id=str(result.user.id),
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "invalid" in error_msg.lower() or "credentials" in error_msg.lower() or "email not confirmed" in error_msg.lower():
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        raise HTTPException(status_code=401, detail=f"登录失败: {error_msg}")


@router.post("/logout")
async def logout():
    """
    Logout. Since we use JWT tokens, logout is primarily handled client-side
    by clearing the stored token. This endpoint exists for completeness.
    """
    return {"success": True, "message": "已退出登录"}
