# app/db/vector_db.py
from functools import lru_cache
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_upstage import UpstageEmbeddings
from app.core.config import settings
from app.core.sparse_encoder import SparseEncoder

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


# 2. 모델 로딩 (메모리 절약을 위해 lru_cache 사용)
@lru_cache(maxsize=1)
def get_sparse_encoder():
    print("🚀 Loading SPLADE Model... (This happens only once)")
    return SparseEncoder(model_name="yjoonjang/splade-ko-v1")


@lru_cache(maxsize=1)
def get_dense_encoder():
    # 검색 시에는 'embedding-query' 모델을 사용해야 성능이 좋습니다.
    return UpstageEmbeddings(
        model=settings.DENSE_MODEL_QUERY, upstage_api_key=settings.UPSTAGE_API_KEY
    )


# 3. 최종 VectorStore 반환 함수
def get_vector_store() -> QdrantVectorStore:
    """
    LangChain/LangGraph에서 바로 사용할 수 있는 VectorStore 객체를 반환합니다.
    """
    client = get_qdrant_client()
    sparse_encoder = get_sparse_encoder()
    dense_encoder = get_dense_encoder()

    return QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        # Dense 설정
        embedding=dense_encoder,
        vector_name="overview_dense",  # [중요] 컬렉션 만들 때 지정한 이름
        # Sparse 설정
        sparse_embedding=sparse_encoder,
        sparse_vector_name="tags_sparse",  # [중요] 컬렉션 만들 때 지정한 이름
        # Hybrid 모드 활성화
        retrieval_mode=RetrievalMode.HYBRID,
        # 메타데이터 payload 중 page_content로 쓸 필드 지정 (선택사항)
        content_payload_key="title",
    )
