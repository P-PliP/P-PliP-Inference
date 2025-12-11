import json
import os

filepath = r"e:\pythonProject\P-plip-inference\lab\attraction_suggest.ipynb"
with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Helper to split string into list of strings with newlines
def to_source(code):
    # Split by newline and append \n to each line except possibly the last one if we want to be exact,
    # but for ipynb usually every line ends with \n is fine.
    lines = code.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


# New implementations
query_rewrite_code = """from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def query_rewrite(state: AttractionState):
    \"\"\"(옵션) 타입을 못 찾았을 때 쿼리를 수정하거나 기본값 설정\"\"\"
    print(f"⚠️ 타입을 찾을 수 없어 쿼리를 보정합니다. (이전: {state['query']})")
    
    # 쿼리 재작성 프롬프트
    system_prompt = (
        "You are a helpful assistant that refines user queries for a tourism recommendation system. "
        "The previous attempt to extract a content type failed. "
        "Rewrite the query to be more specific about looking for tourist attractions, "
        "or if it's too vague, make it a natural query asking for general tourist spots."
        "Return ONLY the rewritten query."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])
    
    chain = prompt | mini_llm | StrOutputParser()
    
    new_query = chain.invoke({"query": state["query"]})
    
    print(f"✅ 보정된 쿼리: {new_query}")
    
    state['query'] = new_query
    return state"""

content_type_check_code = """def content_type_check(state: AttractionState) -> Literal[retrieve_attraction.__name__, content_type_extraction.__name__]:
    
    if state['content_type'] in ContentTypes:
        return retrieve_attraction.__name__
    else:
        return query_rewrite.__name__"""

graph_code = """from langgraph.graph import StateGraph, START, END
from langchain_teddynote.graphs import visualize_graph
graph = StateGraph(AttractionState)

graph.add_node(content_type_extraction.__name__, content_type_extraction)
graph.add_node(retrieve_attraction.__name__, retrieve_attraction)
graph.add_node(query_rewrite.__name__, query_rewrite)
graph.add_edge(START, content_type_extraction.__name__)
graph.add_conditional_edges(
    content_type_extraction.__name__,
    content_type_check,
    {
        retrieve_attraction.__name__ : retrieve_attraction.__name__,\n        query_rewrite.__name__ : query_rewrite.__name__
    }
)
graph.add_edge(query_rewrite.__name__, content_type_extraction.__name__)
graph.add_edge(retrieve_attraction.__name__, END)

suggest_graph = graph.compile()

visualize_graph(suggest_graph)"""

# Update cells
for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if "def query_rewrite(state: AttractionState):" in source:
        cell["source"] = to_source(query_rewrite_code)
    elif "def content_type_check(state: AttractionState)" in source:
        cell["source"] = to_source(content_type_check_code)
    elif "graph = StateGraph(AttractionState)" in source:
        cell["source"] = to_source(graph_code)

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
