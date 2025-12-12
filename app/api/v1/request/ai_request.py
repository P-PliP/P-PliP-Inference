from pydantic import BaseModel
from typing import Optional


class SuggestRequest(BaseModel):
    query: str
    content_type: Optional[Union[str, List[str]]] = None
    lat: float
    lng: float
    k: int
    m: int
