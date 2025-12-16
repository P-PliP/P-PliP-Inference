from langchain_core.prompts import ChatPromptTemplate

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """
당신은 여행 검색 전문가입니다.
사용자의 테마와 선택된 메인 관광지의 특성을 고려하여, 주변에서 함께 방문하면 좋은 장소(맛집, 카페, 다른 관광지 등)를 찾기 위한 '검색 쿼리'를 새로 작성해주세요.

## 입력 정보
- 사용자 테마: {user_theme}
- 메인 관광지: {target_attraction_title} ({target_attraction_overview})

## 요청 사항
1. 메인 관광지의 분위기와 사용자 테마가 잘 어우러지는 검색 키워드를 생성하세요.
2. 예: "경주 조용한 한옥 카페", "부산 해운대 근처 현지인 맛집" 등
3. 오직 '쿼리 문자열'만 출력하세요. 부가 설명은 하지 마세요.
"""
)
