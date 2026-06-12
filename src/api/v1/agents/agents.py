import os
import json
from typing import Literal, List, Dict, Any
import cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

from src.api.v1.schemas.query_schema import AIResponse
# RAGState imported directly from tools where we configured the Annotated messages reducer
from src.api.v1.tools.tools import RAGState, vector_search, keyword_search, hybrid_search
from src.core.db import get_sql_database

load_dotenv(override=True)

# ── Global Thread Configuration Memory Engine ────────────────────────────────
memory = MemorySaver()
config = {"configurable": {"thread_id": "session-1"}}


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

# Document retrieval tool directory dictionary map
DOCUMENT_TOOLS = {
    "vector_search": vector_search,
    "keyword_search": keyword_search,
    "hybrid_search": hybrid_search
}


# ── Node 0: Query Router (Memory Aware) ──────────────────────────────────────
class _RouteDecision(BaseModel):
    route: Literal["product", "vector_search", "keyword_search", "hybrid_search", "general", "hybrid_demand"]
    reason: str

def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an enterprise query router. Classify incoming customer prompts into EXACTLY one target path option based on the ongoing conversation history:

            1. "general" — Select this for ANY query that falls OUTSIDE bank operations, or simple greetings.
                            This includes:
                            - Casual greetings or small talk (e.g., "hi", "hello", "how are you today?").
                            - Out-of-scope general knowledge questions entirely unrelated to either "Product" or "Document" mentioned below.
                            - Any request unrelated to NorthStar Bank services, credit cards, or account data.

            2. "product" — Select this ONLY if the query strictly requires live relational database data (e.g., balance, due dates, calculations) AND does NOT require looking up bank policy rules.

            3. "hybrid_demand" — Select this CRITICAL option if the query requires BOTH checking live database transaction entries AND comparing them against knowledge base policies/guidelines (e.g., "Look at my transactions and tell me if they qualify for the Gold tier reward rule points").

            4. Knowledge Base Searches — Select one of these if the query is strictly about rules, tiers, or fees without needing your live profile numbers:
               - "vector_search"  — Long, scenario-based or descriptive questions.
               - "keyword_search" — Specific alphanumeric codes, acronyms, or metrics.
               - "hybrid_search"  — Complex policy queries mixing technical terms and situational text.
            """
        ),
        ("placeholder", "{messages}")  # ◄── Injects user query & conversational chat history dynamically
    ])

    decision = (prompt | structured_llm).invoke({"messages": state["messages"]})
    print(f"[router_node] Integrated Routing Decision ──► Route: '{decision.route}' | Reason: {decision.reason}")
    return {**state, "route": decision.route}


# ── Node NL2SQL: Credit Card Database Query Translation ───────────────────────
def nl2sql_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    db = get_sql_database()
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a PostgreSQL expert. Given the database schema and the ongoing conversation history,
            write a single valid SELECT query that answers the user's latest question. Use the chat history
            to resolve any implicit pronouns, filters, or entity names (like identifying which card or user is being discussed).

            Rules:
            - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
            - Use only the tables and columns present in the schema.
            - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
            
            CRITICAL QUERY STYLE RULES:
            1. Whenever you compute a spend aggregation or total dollar amount (e.g., USING SUM(amount)) 
               grouped by metrics like merchant_name, card_id, or transaction categories, you MUST 
               ALWAYS include the transaction count as well using exactly `COUNT(*) AS txns` inside the 
               SELECT statement parameters.
            
            Schema info:
            {schema}"""
        ),
        ("placeholder", "{messages}")  # ◄── Injects complete chat history so the LLM knows what to query next
    ])

    # Pass the message history block array along with the schema metadata
    raw_sql = (sql_prompt | llm).invoke({
        "schema": schema_info, 
        "messages": state["messages"]  # ◄── Pulls the history directly from the LangGraph state
    })
    
    generated_sql = raw_sql.content.strip().strip("```").strip().replace("sql\n", "")

    try:
        print("[nl2sql_node] Executing relational database operations...")
        sql_result = db.run(generated_sql)
    except Exception as exc:
        sql_result = f"Postgres query runtime error: {exc}"
        print(f"[nl2sql_node] Error encountered: {sql_result}")

    # Build response context
    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an analytical credit support expert. Answer the user prompt explicitly using the database records below."),
        ("placeholder", "{messages}"), # ◄── Keeps the answer contextual to previous statements
        ("human", "SQL Used:\n{sql}\nQuery Results:\n{result}")
    ])

    answer = (answer_prompt | structured_llm).invoke({
        "messages": state["messages"], 
        "sql": generated_sql, 
        "result": sql_result
    })
    
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql

    # Keep memory in sync for follow-up questions
    assistant_msg = AIMessage(content=response.get("answer", "Processed relational database metrics."))

    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response,
        "messages": [assistant_msg]
    }

# ── Node General Chat: Greetings and Refusing Out-Of-Scope Content ───────────
def general_chat_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful, professional credit support assistant for NorthStar Bank.\n\n"
            "1. If it is a casual greeting, respond warmly and ask how you can assist with credit analytics.\n"
            "2. If it is an out-of-scope question unrelated to banking, credit cards, or policy guidelines, politely decline to answer."
        ),
        ("placeholder", "{messages}") # ◄── Memory awareness hook
    ])
    
    answer = (chat_prompt | structured_llm).invoke({"messages": state["messages"]})
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = None
    
    return {
        **state, 
        "response": response,
        "messages": [AIMessage(content=response.get("answer", ""))] # ◄── Commits assistant turn to history
    }


# ── Node 1C: Direct Tool Execution Engine ─────────────────────────────────────
def execute_tools_node(state: RAGState) -> Dict[str, Any]:
    route_choice = state.get("route")
    retrieved_raw_chunks = list(state.get("retrieved_docs", []))
    
    target_tool = route_choice if route_choice in DOCUMENT_TOOLS else "hybrid_search"
    
    print(f"[execute_tools_node] Running retrieval lookup via tool: '{target_tool}'")
    results = DOCUMENT_TOOLS[target_tool].invoke({"query": state["query"]})
    
    for chunk in results:
        retrieved_raw_chunks.append(
            Document(page_content=chunk["content"], metadata=chunk["metadata"])
        )
            
    return {"retrieved_docs": retrieved_raw_chunks}


# ── Node 2: Cohere Cross-Encoder Reranking ────────────────────────────────────
def rerank_node(state: RAGState) -> RAGState:
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {**state, "reranked_docs": []}

    seen = set()
    unique_docs = []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)

    rerank_response = co.rerank(
        model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        query=state["query"],
        documents=[doc.page_content for doc in unique_docs],
        top_n=10
    )

    reranked_docs = [unique_docs[r.index] for r in rerank_response.results]
    print(f"[rerank_node] Filtered down to top {len(reranked_docs)} high-confidence components.")
    return {**state, "reranked_docs": reranked_docs}


# ── Node 3: OpenAI Multimodal Vision Generation ─────────────────────────────
def generate_answer_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    system_instruction = (
        "You are an expert credit policy analyst and database operations coordinator.\n"
        "Your task is to synthesize a single clean response using the available information collections and ongoing chat logging history:\n"
        "1. Prioritize combining the relational database metrics and transaction findings logs provided.\n"
        "2. Contextualize those findings perfectly against the guidelines pulled from our policy documents.\n\n"
        "Citation structural layout rules:\n"
        "  - document_name: exact filenames used.\n"
        "  - page_no: associated page metrics.\n"
        "  - policy_citations: formatted human readable citation references (e.g. 'Handbook_2026.pdf, Page 12')."
    )

    multimodal_payload_blocks = []
    
    # Inject database context logs into execution chain engine layout block if available
    if state.get("sql_result"):
        db_context = (
            f"\n\n--- [LIVE DATABASE CONTEXT RECORDS FOR COMBINATION] ---\n"
            f"SQL Executed: {state.get('generated_sql')}\n"
            f"Returned Records: {state.get('sql_result')}\n"
        )
        multimodal_payload_blocks.append({"type": "text", "text": db_context})

    # Inject policy text/image context elements
    for idx, doc in enumerate(state.get("reranked_docs", [])):
        meta = doc.metadata
        modality = str(meta.get("content_type", "text")).upper()
        doc_name = meta.get("document_name")
        page_num = meta.get("page")
        
        block_header = f"\n\n--- [Context Block {idx+1} | Source: {doc_name} | Page: {page_num} | Modality: {modality}] ---\n"
        block_content = block_header + doc.page_content
        multimodal_payload_blocks.append({"type": "text", "text": block_content})
        
        if meta.get("content_type") == "image" and meta.get("image_base64"):
            multimodal_payload_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{meta['image_base64']}",
                    "detail": "low"
                }
            })

    # Pull current raw instruction alongside memory references
    multimodal_payload_blocks.append({"type": "text", "text": f"\n\nResolve the following explicit question using historical context logs where necessary: {state['query']}"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("placeholder", "{messages}"), # ◄── Dynamic conversation context injection point
        ("human", multimodal_payload_blocks)
    ])

    result = (prompt | structured_llm).invoke({"messages": state["messages"]})
    response_dict = result.model_dump()
    
    if state.get("generated_sql"):
        response_dict["sql_query_executed"] = state.get("generated_sql")
        
    return {
        **state, 
        "response": response_dict,
        "messages": [AIMessage(content=response_dict.get("answer", ""))] # ◄── Commits unified RAG turn to history
    }


# ── Assemble Graph State Machine ──────────────────────────────────────────────
def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("router", router_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("general_chat", general_chat_node) 
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    # 1. Router Entry Gate
    def _route_resolver(state: RAGState) -> str:
        if state["route"] == "general":
            return "general_chat"
        elif state["route"] == "product" or state["route"] == "hybrid_demand":
            return "nl2sql"
        return "execute_tools"

    graph.add_conditional_edges(
        "router",
        _route_resolver,
        {
            "general_chat": "general_chat", 
            "nl2sql": "nl2sql",
            "execute_tools": "execute_tools"
        }
    )

    # 2. Sequential Hybrid Gate
    def _post_sql_resolver(state: RAGState) -> str:
        if state["route"] == "hybrid_demand":
            return "execute_tools"
        return "end_directly"

    graph.add_conditional_edges(
        "nl2sql",
        _post_sql_resolver,
        {
            "execute_tools": "execute_tools",
            "end_directly": END  
        }
    )

    graph.add_edge("execute_tools", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("general_chat", END) 
    graph.add_edge("generate_answer", END)

    try:
        compiled_agent = graph.compile(checkpointer=memory) # ◄── State machine compile checkpoint attached
        # Generate layout representation
        graph_image = compiled_agent.get_graph().draw_mermaid_png()
        os.makedirs("src/data", exist_ok=True)
        with open("src/data/sequential_hybrid_flow.png", "wb") as f:
            f.write(graph_image)
    except Exception:
        pass

    return compiled_agent

rag_graph = build_rag_graph()


# ── Thread-Aware Multi-turn Execution Loop Engine ──────────────────────────────
def run_search_agent(query: str) -> dict:
    # 1. Construct standard HumanMessage format interface
    new_message = HumanMessage(content=query)
    
    # 2. Setup standard default initialization context structures
    initial_state = {
        "query": query,
        "messages": [new_message],  # ◄── This gets appended automatically by the add_messages reducer
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "route": "",
        "generated_sql": "",
        "sql_result": "",
        "agent_history": [],
        "tool_calls": []
    }
    
    # 3. Invoke using thread configuration context memory references
    final_state = rag_graph.invoke(initial_state, config=config)
    return final_state["response"]