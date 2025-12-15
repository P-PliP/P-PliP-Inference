from app.agents.state import PlanState
from app.core.llm import mini_llm
from app.agents.prompts.query_rewrite_prompt import QUERY_REWRITE_PROMPT
from langchain_core.output_parsers import StrOutputParser


async def rewrite_query_node(state: PlanState) -> PlanState:
    user_theme = state["user_theme"]
    target = state["target_attraction"]

    chain = QUERY_REWRITE_PROMPT | mini_llm | StrOutputParser()

    # 테마 + 관광지 정보로 쿼리 재작성
    new_query = await chain.ainvoke(
        {
            "user_theme": user_theme,
            "target_attraction_title": target["title"],
            "target_attraction_overview": target["overview"][:300],  # 너무 길면 자름
        }
    )

    # 재작성된 쿼리로 user_theme 업데이트 또는 별도 필드에 저장
    # 여기서는 검색 노드(similar_search)가 user_theme을 사용하므로 이를 업데이트합니다.
    return {"user_theme": new_query}
