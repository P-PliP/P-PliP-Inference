# app/db/vector_db.py
from functools import lru_cache
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_upstage import UpstageEmbeddings
from app.core.config import settings
from app.core.sparse_encoder import SparseEncoder
from qdrant_client.http import models
from typing import Optional
from langchain.retrievers import EnsembleRetriever

# app/db/custom_qdrant.py
from typing import Any, Dict, Optional, Iterable, List
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document


class CustomQdrantVectorStore(QdrantVectorStore):
    """
    Qdrant payload:
        {
          "no": ...,
          "title": "...",
          "overview": "...",
          "addr1": "...",
          "location": {...},
          ...
        }

    로 되어 있을 때,
        page_content  = payload["title"]
        metadata      = payload 전체 (title 포함)
    으로 매핑하는 래퍼.
    """

    @classmethod
    def _document_from_point(
        cls,
        scored_point: Any,
        collection_name: str,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        # scored_point.payload 전체를 metadata로 사용
        payload: Dict[str, Any] = scored_point.payload or {}

        # page_content: title 또는 content_payload_key 지정값
        title_key = content_payload_key or "no"
        page_content = str(payload.get(title_key, "no"))

        # metadata: payload 전체 복사 (원하면 title 빼고 싶으면 아래 dict comprehension으로 변경)
        metadata: Dict[str, Any] = dict(payload)

        # QdrantVectorStore 기본 메타 필드 유지
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        metadata["_score"] = scored_point.score
        return Document(
            page_content=page_content,
            metadata=metadata,
        )


# 1. 전역 변수로 클라이언트 관리 (연결 풀 재사용)
_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _qdrant_client


@lru_cache(maxsize=1)
def get_sparse_encoder():
    print("🚀 Loading SPLADE Model... (This happens only once)")
    return SparseEncoder(model_name="yjoonjang/splade-ko-v1")


@lru_cache(maxsize=1)
def get_dense_encoder():
    return UpstageEmbeddings(
        model=settings.DENSE_MODEL_QUERY, upstage_api_key=settings.UPSTAGE_API_KEY
    )


# 2. [핵심 수정] 모드를 인자로 받는 VectorStore 생성 함수
# 이 함수는 무거운 작업 없이 가벼운 껍데기(VectorStore)만 반환하므로 매번 호출해도 괜찮습니다.
def create_vector_store(mode: RetrievalMode) -> QdrantVectorStore:
    client = get_qdrant_client()
    sparse_encoder = get_sparse_encoder()
    dense_encoder = get_dense_encoder()

    return CustomQdrantVectorStore.from_existing_collection(
        # client=client,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=dense_encoder,
        vector_name="overview_dense",
        sparse_embedding=sparse_encoder,
        sparse_vector_name="tags_sparse",
        retrieval_mode=mode,  # [중요] 여기서 모드를 설정합니다!
        content_payload_key="no",
    )


# 3. Retriever 생성 함수들 수정 (search_kwargs에서 retrieval_mode 제거)


def get_sparse_retriever(k: int, filter: Optional[models.Filter] = None):
    # SPARSE 모드로 설정된 VectorStore를 새로 생성
    vector_store = create_vector_store(RetrievalMode.SPARSE)

    search_kwargs = {
        "k": k,
        "with_payload": [
            "no",
            "title",
            "overview",
            "content_type",
            "addr1",
            "addr2",
            "first_image1",
            "first_image2",
            "content_id",
            "homepage",
            "tag_names",
            "location",
        ],
    }
    if filter:
        search_kwargs["filter"] = filter

    return vector_store.as_retriever(search_kwargs=search_kwargs)


def get_dense_retriever(k: int, filter: Optional[models.Filter] = None):
    # DENSE 모드로 설정된 VectorStore를 새로 생성
    vector_store = create_vector_store(RetrievalMode.DENSE)

    search_kwargs = {
        "k": k,
        "with_payload": [
            "no",
            "title",
            "overview",
            "content_type",
            "addr1",
            "addr2",
            "first_image1",
            "first_image2",
            "content_id",
            "homepage",
            "tag_names",
            "location",
        ],
    }
    if filter:
        search_kwargs["filter"] = filter

    return vector_store.as_retriever(search_kwargs=search_kwargs)


def get_hybrid_retriever(k: int, filter: Optional[models.Filter] = None):
    # HYBRID 모드로 설정된 VectorStore를 새로 생성
    vector_store = create_vector_store(RetrievalMode.HYBRID)

    search_kwargs = {
        "k": k,
        "with_payload": [
            "no",
            "title",
            "overview",
            "content_type",
            "addr1",
            "addr2",
            "first_image1",
            "first_image2",
            "content_id",
            "homepage",
            "tag_names",
            "location",
        ],
    }
    if filter:
        search_kwargs["filter"] = filter

    return vector_store.as_retriever(search_kwargs=search_kwargs)


def get_ensemble_retriever(
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    k: int = 10,
    filter: Optional[models.Filter] = None,
):
    dense = get_dense_retriever(k, filter)
    sparse = get_sparse_retriever(k, filter)

    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[dense_weight, sparse_weight],
    )


# 4. [NEW] 직접 검색을 위한 유틸리티 함수
async def search_hybrid(
    query: str, limit: int = 5, filter: Optional[models.Filter] = None
):
    """
    EnsembleRetriever를 사용하여 검색을 수행하고 Document 리스트를 반환합니다.
    비동기 실행을 위해 run_in_executor 등을 사용해야 할 수 있지만,
    여기서는 LangChain Retriever가 비동기 invoke를 지원하므로 ainvoke를 사용합니다.
    """
    retriever = get_ensemble_retriever(k=limit, filter=filter)
    results = await retriever.ainvoke(query)

    # Document 객체에서 payload(metadata) 추출하여 리스트로 반환
    # similar_search Node에서 기대하는 포맷(Attraction TypedDict와 호환)으로 변환하면 좋음
    return [doc.metadata for doc in results]
