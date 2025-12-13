from pydantic import BaseModel
from typing import List
class SuggestResponse(BaseModel):
    no: int
    title: str
    content_type: str
    address: str
    latitude: float
    longitude: float
    big_image: str
    thumbnail: str
    tags: List[str] 
    homepage: str