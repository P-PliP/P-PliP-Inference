from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from typing import Dict, Any


class PlanRequest(BaseModel):
    user_theme: str = Field(description="사용자가 원하는 여행 테마")
    attraction_id: int = Field(description="중심이 되는 관광지의 ID")


class AttractionResponse(BaseModel):
    no: int
    title: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    addr1: Optional[str] = None
    overview: Optional[str] = None
    first_image1: Optional[str] = None
    first_image2: Optional[str] = None
    content_type: Optional[str] = None


class ToDoItem(BaseModel):
    name: str
    detail_plan_desc: str
    start_at: str  # datetime string
    end_at: str  # datetime string
    attraction: AttractionResponse


class PlanResponse(BaseModel):
    plan_title: str = Field(description="여행 계획의 제목")
    start_date: str = Field(description="여행 시작일 (YYYY-MM-DD)")
    end_date: str = Field(description="여행 종료일 (YYYY-MM-DD)")
    to_dos: List[ToDoItem]
