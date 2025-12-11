from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

content_type_extract_system_prompt = """
        당신은 사용자의 입력을 분석하여 적절한 관광지 타입을 추출하는 도우미입니다.
        사용자의 입력을 분석하여 적절한 관광지 타입을 추출하세요.
        
        [분류 기준]
        - "카페", "식당", "맛집" 등 -> '음식점'
        - "둘레길", "산책" 등 -> '여행코스'
        - "호텔", "펜션" 등 -> '숙박'
        - "축제", "공연", "행사" 등 -> '축제공연행사'
        - "운동, 레저, 활동" 등 -> '레포츠'
        - "쇼핑" -> '쇼핑'
        - "카페거리", "유명 명소" 포함 그 외 -> '관광지'
        """
