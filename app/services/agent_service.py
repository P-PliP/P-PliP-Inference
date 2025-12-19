from app.api.v1.request.ai_request import SuggestRequest

from app.db.vector_db import get_hybrid_retriever, get_ensemble_retriever

from app.db.filters import build_geo_fileter, build_geo_fileter_with_content_type

from app.api.v1.response.ai_response import SuggestResponse

from typing import List

from langchain.schema import Document

from langchain.retrievers import MultiQueryRetriever

from app.agents.graph import rerank_chain

from app.core.llm import mini_llm


class AgentService:

    def __init__(self):
        pass

    async def suggest_attraction(
        self, request: SuggestRequest
    ) -> List[SuggestResponse]:
        """

        사용자 쿼리와 k 값을 받아 관광지를 추천합니다.
        """

        if request.content_types:

            request.content_types = (
                request.content_types if len(request.content_types) > 0 else None
            )

        filter = build_geo_fileter_with_content_type(
            request.lat, request.lng, request.m, request.content_types
        )

        retriever = get_ensemble_retriever(
            k=100, dense_weight=0.3, sparse_weight=0.7, filter=filter
        )

        # retriever = MultiQueryRetriever.from_llm(

        #     retriever=get_ensemble_retriever(

        #         k=100, dense_weight=0.7, sparse_weight=0.3, filter=filter

        #     ),

        #     llm=mini_llm,

        # )

        docs = await retriever.ainvoke(request.query)

        # 1. 중복 제거 루틴 추가

        # EnsembleRetriever 특성상, 같은 문서라도 점수가 다르면 중복으로 잡힐 수 있습니다.

        # 따라서 `no`를 기준으로 중복을 제거합니다.

        unique_docs = {}

        for doc in docs:

            doc_id = doc.metadata.get("no")

            if doc_id not in unique_docs:

                unique_docs[doc_id] = doc

        docs = list(unique_docs.values())

        if len(docs) > 100:

            docs = docs[:100]
        print("docs: ", docs)
        docs = await self.rerank_documents(
            query=request.query, retrieved_docs=docs, top_k=request.k
        )

        res = []

        for doc in docs:

            raw = doc.metadata.get("tag_names") or ""

            tags = [t.strip() for t in raw.split(",") if t.strip()]

            res.append(
                SuggestResponse(
                    no=doc.metadata["no"],
                    title=doc.metadata["title"],
                    latitude=doc.metadata["location"]["lat"],
                    longitude=doc.metadata["location"]["lon"],
                    content_type=doc.metadata["content_type"],
                    address=doc.metadata["addr1"],
                    big_image=doc.metadata["first_image1"],
                    thumbnail=doc.metadata["first_image2"],
                    tags=tags,
                    homepage=doc.metadata["homepage"],
                )
            )
        return res

    async def rerank_documents(
        self, query: str, retrieved_docs: List[Document], top_k: int
    ) -> List[Document]:

        # (1) 배치 사이즈 설정 (한 번에 10개씩 평가)

        batch_size = 10

        batches = [
            retrieved_docs[i : i + batch_size]
            for i in range(0, len(retrieved_docs), batch_size)
        ]

        # (2) 배치 입력을 위한 전처리 (문서 내용을 텍스트로 변환 + ID 매핑)

        # LLM에게 보낼 때는 "ID: 내용" 형태로 텍스트를 만들어줍니다.

        batch_inputs = []

        for idx_start, batch in enumerate(batches):

            docs_text = ""

            for i, doc in enumerate(batch):

                # 전체 리스트 기준의 절대 인덱스(global_index)를 ID로 사용

                global_index = (idx_start * batch_size) + i

                content_preview = doc.metadata.get("overview", "")[:500].replace(
                    "\n", " "
                )

                docs_text += f"""
                    =========================

                    Document ID: {global_index}

                    Content: {content_preview}
                    =========================
                    """

            batch_inputs.append({"query": query, "docs_text": docs_text})

        # (3) 비동기 병렬 실행 (chain.abatch 사용) -> 10개의 요청이 동시에 날아감
        print(f"query: {query}")
        print(f"🔄 Reranking {len(retrieved_docs)} docs in {len(batches)} batches...")
        batch_results = await rerank_chain.abatch(batch_inputs)

        print(batch_results)

        # (4) 결과 취합 및 정렬

        unique_scores = {}

        for res in batch_results:

            if res and res.results:

                for item in res.results:

                    unique_scores[item.doc_id] = item.score

        # 점수 기준 내림차순 정렬

        sorted_doc_ids = sorted(
            unique_scores.keys(), key=lambda k: unique_scores[k], reverse=True
        )

        # (5) Top K 추출 및 원본 문서 매핑

        final_docs = []

        print("\n📊 Top Ranked Docs:")
        for item in sorted_doc_ids[:top_k]:

            original_doc = retrieved_docs[item]

            # 메타데이터에 점수 추가 (선택사항)

            original_doc.metadata["relevance_score"] = unique_scores[item]

            if unique_scores[item] > 0:

                final_docs.append(original_doc)

            print(f"- [Score: {unique_scores[item]}] ID: {item}")

        return final_docs
