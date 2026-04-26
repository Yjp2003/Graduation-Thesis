"""
User management routes: list users, delete users.
Uses Supabase admin client for cross-user operations.
"""
from fastapi import APIRouter, HTTPException, Depends

from middleware import get_current_user
from supabase_client import get_supabase_admin, get_supabase_client

router = APIRouter()


@router.get("")
async def list_users(user: dict = Depends(get_current_user)):
    """Get list of all registered users (from profiles table)."""
    supabase = get_supabase_client()
    supabase.postgrest.auth(user["token"])

    try:
        result = (
            supabase.table("profiles")
            .select("id, username, role, created_at")
            .order("created_at", desc=False)
            .execute()
        )
        return {"success": True, "users": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")


@router.delete("/{user_id}")
async def delete_user(user_id: str, user: dict = Depends(get_current_user)):
    """Delete a user. Cannot delete self."""
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="不能删除当前登录的用户")

    try:
        # Delete from auth.users (cascades to profiles via FK)
        admin = get_supabase_admin()
        admin.auth.admin.delete_user(user_id)

        return {"success": True, "message": "用户已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除用户失败: {str(e)}")
