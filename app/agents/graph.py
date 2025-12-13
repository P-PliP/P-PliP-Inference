from app.agents.prompts.templates import reranker_template, parser
from app.core.llm import mini_llm, large_llm
from app.schemas.chat_schema import RankedDocs
from langchain_core.output_parsers import PydanticOutputParser

structed_llm = mini_llm.with_structured_output(schema=RankedDocs)
rerank_chain = reranker_template | structed_llm
