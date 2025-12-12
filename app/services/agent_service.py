from app.api.v1.request.ai_request import SuggestRequest
from app.db.vector_db import get_hybrid_retriever, get_ensemble_retriever
from app.db.filters import build_geo_fileter, build_geo_fileter_with_content_type
from app.api.v1.response.ai_response import SuggestResponse
from typing import List


class AgentService:
    def __init__(self):
        pass

    async def suggest_attraction(
        self, request: SuggestRequest
    ) -> List[SuggestResponse]:
        """
        사용자 쿼리와 k 값을 받아 관광지를 추천합니다.
        """
        filter = build_geo_fileter_with_content_type(
            request.lat, request.lng, request.m, request.content_type
        )
        retriever = get_ensemble_retriever(
            k=request.k, dense_weight=0.3, sparse_weight=0.7, filter=filter
        )
        docs = await retriever.ainvoke(request.query)
        res = []
        for doc in docs:
            raw = doc.metadata.get("tag_names") or ""
            tags = [t.strip() for t in raw.split(",") if t.strip()]

            res.append(
                SuggestResponse(
                    no=doc.metadata["no"],
                    title=doc.metadata["title"],
                    content_type=doc.metadata["content_type"],
                    address=doc.metadata["addr1"],
                    big_image=doc.metadata["first_image1"],
                    thumbnail=doc.metadata["first_image2"],
                    tags=tags,
                    homepage=doc.metadata["homepage"],
                )
            )
        return res
