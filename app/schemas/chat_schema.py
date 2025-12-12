from pydantic import BaseModel, Field


class ContentTypeExtract(BaseModel):
    content_type: str  = Field(
        description="사용자의 질문에 가장 적합한 관광지 타입"
    )
