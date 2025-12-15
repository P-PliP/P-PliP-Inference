from langgraph.graph import StateGraph, END
from app.agents.state import PlanState
from app.agents.nodes.attraction_load import load_attraction_node
from app.agents.nodes.query_rewrite import rewrite_query_node
from app.agents.nodes.similar_search import search_similar_attractions_node
from app.agents.nodes.accommodation_search import search_accommodation_node
from app.agents.nodes.plan_generate import generate_plan_node
from app.core.llm import mini_llm
from app.agents.prompts.templates import reranker_template
from app.schemas.chat_schema import RankedDocs
from langchain_core.output_parsers import PydanticOutputParser


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


rerank_chain = reranker_template | mini_llm.with_structured_output(RankedDocs)
