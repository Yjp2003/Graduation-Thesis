"""
Detection records routes: CRUD for detection history.
All operations scoped to the authenticated user via RLS.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from middleware import get_current_user
from supabase_client import get_supabase_client

router = APIRouter()


class DetectionItem(BaseModel):
    classId: int
    className: str
    score: float
    box: List[float]  # [x1, y1, x2, y2]


class RecordCreate(BaseModel):
    time: str
    image: Optional[str] = None  # base64 snapshot
    total_detections: int = 0
    fps: float = 0
    avg_confidence: float = 0
    detections: List[DetectionItem] = []
    video_clips: Optional[list] = None  # list of {index, start_sec, end_sec, data}


class RecordResponse(BaseModel):
    id: str
    user_id: str
    time: str
    image: Optional[str] = None
    total_detections: int
    fps: float
    avg_confidence: float
    detections: list
    video_clips: Optional[list] = None
    created_at: str


@router.get("")
async def list_records(
    user: dict = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
):
    """Get detection records for the current user, newest first. Supports pagination."""
    supabase = get_supabase_client()
    # Set auth context for RLS
    supabase.postgrest.auth(user["token"])

    try:
        query = (
            supabase.table("detection_records")
            .select("*", count="exact")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        result = query.execute()
        return {
            "success": True,
            "records": result.data,
            "total": result.count if result.count is not None else len(result.data),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取记录失败: {str(e)}")


@router.post("")
async def create_record(record: RecordCreate, user: dict = Depends(get_current_user)):
    """Save a new detection record."""
    supabase = get_supabase_client()
    supabase.postgrest.auth(user["token"])

    try:
        data = {
            "user_id": user["id"],
            "time": record.time,
            "image": record.image,
            "total_detections": record.total_detections,
            "fps": record.fps,
            "avg_confidence": record.avg_confidence,
            "detections": [d.model_dump() for d in record.detections],
            "video_clips": record.video_clips,
        }

        result = supabase.table("detection_records").insert(data).execute()

        if result.data:
            return {"success": True, "record": result.data[0]}
        raise HTTPException(status_code=500, detail="保存记录失败")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存记录失败: {str(e)}")


@router.delete("/{record_id}")
async def delete_record(record_id: str, user: dict = Depends(get_current_user)):
    """Delete a specific detection record."""
    supabase = get_supabase_client()
    supabase.postgrest.auth(user["token"])

    try:
        result = (
            supabase.table("detection_records")
            .delete()
            .eq("id", record_id)
            .eq("user_id", user["id"])
            .execute()
        )
        return {"success": True, "message": "记录已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除记录失败: {str(e)}")
