from pydantic import BaseModel
from typing import Optional


class SuggestRequest(BaseModel):
    query: str
    content_type: Optional[str]
    lat: float
    lng: float
    k: int
    m: int
