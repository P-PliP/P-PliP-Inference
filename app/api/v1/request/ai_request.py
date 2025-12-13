from pydantic import BaseModel
from typing import Optional, Union, List


class SuggestRequest(BaseModel):
    query: str
    content_types: Optional[Union[str, List[str]]] = None
    lat: float
    lng: float
    k: int
    m: int
