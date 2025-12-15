from langgraph.graph import StateGraph, END
from app.agents.state import PlanState
from app.agents.nodes.attraction_load import load_attraction_node
from app.agents.nodes.query_rewrite import rewrite_query_node
from app.agents.nodes.similar_search import search_similar_attractions_node
from app.agents.nodes.accommodation_search import search_accommodation_node
from app.agents.nodes.plan_generate import generate_plan_node


def create_plan_graph():
    workflow = StateGraph(PlanState)

    # 노드 추가
    workflow.add_node("attraction_load", load_attraction_node)
    workflow.add_node("query_rewrite", rewrite_query_node)
    workflow.add_node("similar_search", search_similar_attractions_node)
    workflow.add_node("accommodation_search", search_accommodation_node)
    workflow.add_node("plan_generate", generate_plan_node)

    # 엣지 연결
    workflow.set_entry_point("attraction_load")
    workflow.add_edge("attraction_load", "query_rewrite")
    workflow.add_edge("query_rewrite", "similar_search")
    workflow.add_edge("similar_search", "accommodation_search")
    workflow.add_edge("accommodation_search", "plan_generate")
    workflow.add_edge("plan_generate", END)

    return workflow.compile()


app_graph = create_plan_graph()

# Rerank Chain 복구 (AgentService에서 사용)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.schemas.chat_schema import RankedDocs
from app.core.llm import mini_llm

RERANK_PROMPT = ChatPromptTemplate.from_template(
    """
당신은 검색 결과 평가자입니다. 사용자의 질문과 문서 내용을 비교하여 관련성 점수를 매겨주세요.

질문: {query}

문서 목록:
{docs_text}

각 문서에 대해 0~100점 사이의 점수를 부여하세요. 높은 점수는 더 관련성이 높음을 의미합니다.
JSON 형식으로 출력하세요.
"""
)

rerank_chain = (
    RERANK_PROMPT | mini_llm | PydanticOutputParser(pydantic_object=RankedDocs)
)
