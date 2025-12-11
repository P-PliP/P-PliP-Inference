from pydantic import BaseModel, Field
from app.agents.enum import ContentTypes


class ContentTypeExtract(BaseModel):
    content_type: ContentTypes = Field(
        description="사용자의 질문에 가장 적합한 관광지 타입"
    )
