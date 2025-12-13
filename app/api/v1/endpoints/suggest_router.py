from fastapi import APIRouter
from app.schemas.chat_schema import ContentTypeExtract
from app.core.dependencies import get_agent_service
from app.services.agent_service import AgentService
from fastapi import Depends
from app.api.v1.request.ai_request import SuggestRequest
from typing import List
from app.api.v1.response.ai_response import SuggestResponse

router = APIRouter()


@router.post("", response_model=List[SuggestResponse])
async def suggest_attraction(
    request: SuggestRequest, serivce: AgentService = Depends(get_agent_service)
):
    """
    사용자의 요청(query)을 받아 관광지를 추천합니다.
    """
    result = await serivce.suggest_attraction(request)
    return result
