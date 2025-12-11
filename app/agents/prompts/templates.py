from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from app.agents.prompts.system import content_type_extract_system_prompt

content_type_extract_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=content_type_extract_system_prompt),
        HumanMessage(content="Query:{query}"),
    ]
)
