# import os
# from typing import Literal
# import cohere
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, END
# from pydantic import BaseModel

# from src.api.v1.schemas.query_schema import AIResponse
# from src.api.v1.tools.tools import RAGState, vector_search_node
# from src.core.db import get_sql_database

# load_dotenv(override=True)

# def _get_llm() -> ChatOpenAI:
#     return ChatOpenAI(
#         model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
#         api_key=os.getenv("OPENAI_API_KEY"),
#         temperature=0
#     )

# # ── Node 0: Query Router ──────────────────────────────────────────────────────
# class _RouteDecision(BaseModel):
#     route: Literal["product", "document"]
#     reason: str

# def router_node(state: RAGState) -> RAGState:
#     llm = _get_llm()
#     structured_llm = llm.with_structured_output(_RouteDecision)

#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             """You are an enterprise query router. Classify incoming customer prompts into EXACTLY one of two pathways:
            
#             "product"  — Queries regarding live transactions, accounts, billing histories, metrics, or tables inside the 
#                          relational credit card database (customers, credit_cards, card_transactions, reward_transactions, billing_statements).
            
#             "document" — Queries regarding structural guideline documents, insurance policy rules, handbooks, or image charts."""
#         ),
#         ("human", "Query: {query}")
#     ])

#     decision = (prompt | structured_llm).invoke({"query": state["query"]})
#     print(f"[router_node] Dynamic Route Target Set ──► '{decision.route}' | Reason: {decision.reason}")
#     return {**state, "route": decision.route}


# # ── Node NL2SQL: Credit Card Database Query Translation ───────────────────────
# def nl2sql_node(state: RAGState) -> RAGState:
#     llm = _get_llm()
#     db = get_sql_database()
#     schema_info = db.get_table_info()

#     sql_prompt = ChatPromptTemplate.from_messages([
#         (
#             """You are a PostgreSQL expert. Given the database schema below,
#            write a single valid SELECT query that answers the user's question.


#            Rules:
#            - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
#            - Use only the tables and columns present in the schema.
#            - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
#            - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
#            - For product or text searches: NEVER search for the full multi-word phrase as one
#            ILIKE pattern. Instead, split the search into individual meaningful keywords
#            and OR them together across both name and description columns.
#            Example — user asks "wireless headset":
#                WHERE (name ILIKE '%wireless%' OR description ILIKE '%wireless%')
#                OR (name ILIKE '%headset%'  OR description ILIKE '%headset%')
#                OR (name ILIKE '%headphones%' OR description ILIKE '%headphones%')
#            Use your knowledge of synonyms (headset/headphones, laptop/notebook, etc.)
#            to cast a wider net when the exact term may not match.
            
#             Schema info:
#             {schema}"""
#         ),
#         ("human", "Question: {question}")
#     ])

#     raw_sql = (sql_prompt | llm).invoke({"schema": schema_info, "question": state["query"]})
#     generated_sql = raw_sql.content.strip().strip("```").strip().replace("sql\n", "")

#     try:
#         print("Running the query..")
#         sql_result = db.run(generated_sql)
#         print(" query. executed.")
#     except Exception as exc:

#         sql_result = f"Postgres query runtime error: {exc}"
#         print(sql_result)

#     structured_llm = llm.with_structured_output(AIResponse)
#     answer_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are an analytical credit support expert. Answer the user prompt explicitly using the database records below."),
#         ("human", "Question: {query}\nSQL Used:\n{sql}\nQuery Results:\n{result}")
#     ])

#     answer = (answer_prompt | structured_llm).invoke({
#         "query": state["query"], "sql": generated_sql, "result": sql_result
#     })
    
#     response = answer.model_dump()
#     response["policy_citations"] = "N/A"
#     response["sql_query_executed"] = generated_sql
#     #print(schema_info)
#     return {
#         **state,
#         "generated_sql": generated_sql,
#         "sql_result": str(sql_result),
#         "response": response
#     }


# # ── Node 2: Cohere Cross-Encoder Reranking ────────────────────────────────────
# def rerank_node(state: RAGState) -> RAGState:
#     co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
#     docs = state["retrieved_docs"]

#     if not docs:
#         return {**state, "reranked_docs": []}

#     rerank_response = co.rerank(
#         model="rerank-english-v3.0",
#         query=state["query"],
#         documents=[doc.page_content for doc in docs],
#         top_n=10
#     )

#     reranked_docs = [docs[r.index] for r in rerank_response.results]
#     print(f"[rerank_node] Filtered down to top {len(reranked_docs)} high-confidence components.")
#     return {**state, "reranked_docs": reranked_docs}


# # ── Node 3: OpenAI Multimodal Vision Generation ─────────────────────────────
# def generate_answer_node(state: RAGState) -> RAGState:
#     llm = _get_llm()
#     structured_llm = llm.with_structured_output(AIResponse)

#     system_instruction = (
#         "You are an expert credit policy analyst. Answer the question using ONLY the provided text, table values, or diagrams.\n\n"
#         "CRITICAL: The context may contain chunks from MULTIPLE versions of the same document "
#         "(e.g., a 2025 edition and a 2026 edition). When answers differ across versions, do NOT pick only one:\n"
#         "  - Lead with the most recent / current version's answer (highest year).\n"
#         "  - Explicitly note how earlier versions differed (e.g., 'As of the 2026 policy...; previously, under 2025...').\n"
#         "  - If all versions agree, simply provide the single consolidated answer.\n\n"
#         "Citation structural layout rules:\n"
#         "  - document_name: exact filenames used.\n"
#         "  - page_no: associated page metrics.\n"
#         "  - policy_citations: formatted human readable citation references (e.g. 'Handbook_2026.pdf, Page 12')."
#     )

#     # Construct the multimodal token block array for vision parsing
#     multimodal_payload_blocks = []
    
#     for idx, doc in enumerate(state["reranked_docs"]):
#         meta = doc.metadata
#         modality = str(meta.get("content_type", "text")).upper()
#         doc_name = meta.get("document_name")
#         page_num = meta.get("page")
        
#         block_header = f"\n\n--- [Context Block {idx+1} | Source: {doc_name} | Page: {page_num} | Modality: {modality}] ---\n"
#         block_content = block_header + doc.page_content
        
#         # 1. Attach text payload components
#         multimodal_payload_blocks.append({"type": "text", "text": block_content})
        
#         # 2. Attach base64 data components if the document contains a parsed image asset
#         if meta.get("content_type") == "image" and meta.get("image_base64"):
#             multimodal_payload_blocks.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/png;base64,{meta['image_base64']}",
#                     "detail": "low"
#                 }
#             })

#     # Append question context to the end of the prompt payload array
#     multimodal_payload_blocks.append({"type": "text", "text": f"\n\nUser Question to resolve: {state['query']}"})

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_instruction),
#         ("human", multimodal_payload_blocks)
#     ])

#     result = (prompt | structured_llm).invoke({})
#     return {**state, "response": result.model_dump()}


# # ── Assemble Graph State Machine ──────────────────────────────────────────────
# def build_rag_graph():
#     graph = StateGraph(RAGState)

#     graph.add_node("router", router_node)
#     graph.add_node("nl2sql", nl2sql_node)
#     graph.add_node("vector_search", vector_search_node)
#     graph.add_node("rerank", rerank_node)
#     graph.add_node("generate_answer", generate_answer_node)

#     graph.set_entry_point("router")

#     graph.add_conditional_edges(
#         "router",
#         lambda state: state["route"],
#         {"product": "nl2sql", "document": "vector_search"}
#     )

#     graph.add_edge("nl2sql", END)
#     graph.add_edge("vector_search", "rerank")
#     graph.add_edge("rerank", "generate_answer")
#     graph.add_edge("generate_answer", END)

#     # Generates documentation diagram architecture locally automatically upon compilation setup
#     try:
#         compiled_agent = graph.compile()
#         graph_image = compiled_agent.get_graph().draw_mermaid_png()
#         os.makedirs("src/data", exist_ok=True)
#         with open("src/data/reranking_workflow.png", "wb") as f:
#             f.write(graph_image)
#     except Exception:
#         pass # Handle cases where local dependencies do not support rendering

#     return compiled_agent

# rag_graph = build_rag_graph()

# def run_search_agent(query: str) -> dict:
#     initial_state: RAGState = {
#         "query": query,
#         "retrieved_docs": [],
#         "reranked_docs": [],
#         "response": {},
#         "route": "",
#         "generated_sql": "",
#         "sql_result": "",
#     }
#     final_state = rag_graph.invoke(initial_state)
#     return final_state["response"]

# import os
# import json
# from typing import Literal, List, Dict, Any
# import cohere
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
# from langchain_core.documents import Document
# from langgraph.graph import StateGraph, END
# from pydantic import BaseModel

# from src.api.v1.schemas.query_schema import AIResponse
# from src.api.v1.tools.tools import RAGState, vector_search, keyword_search, hybrid_search
# from src.core.db import get_sql_database

# load_dotenv(override=True)

# def _get_llm() -> ChatOpenAI:
    
#     return ChatOpenAI(
#         model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4"),
#         api_key=os.getenv("OPENAI_API_KEY"),
#         temperature=0
#     )

# # Document retrieval tool directory dictionary map
# DOCUMENT_TOOLS = {
#     "vector_search": vector_search,
#     "keyword_search": keyword_search,
#     "hybrid_search": hybrid_search
# }

# # ── Node 0: Query Router ──────────────────────────────────────────────────────
# class _RouteDecision(BaseModel):
#     route: Literal["product", "document"]
#     reason: str

# def router_node(state: RAGState) -> RAGState:
#     llm = _get_llm()
#     structured_llm = llm.with_structured_output(_RouteDecision)

#     prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             """You are an enterprise query router. Classify incoming customer prompts into EXACTLY one of two pathways:
            
            
#                 "product"  — Queries requiring retrieval or calculation of live customer data. 
#                             This includes:
#                             - Customer profiles (demographics, KYC status).
#                             - Credit card account status (limits, outstanding balance, due dates).
#                             - Transaction history (purchases, EMI details, refunds, category spending).
#                             - Reward point balances and redemption history.
#                             - Billing statement summaries (monthly balances, payments, fees).
#                             (Database scope: customers, credit_cards, card_transactions, reward_transactions, billing_statements)

#                 "document" — Queries regarding knowledge base policies, product guidelines, or instructional content.
#                             This includes:
#                             - Bank product tier features (Classic, Gold, Platinum, Signature).
#                             - Fee schedules and penalty structures.
#                             - Reward program earning rules and expiry policies.
#                             - Spend categorization logic and Merchant Category Code (MCC) definitions.
#                             - Billing cycle rules and payment processing guidelines.
#                             (Knowledge base scope: NorthStar Bank Credit Card Product Guide v2.1)"""
#         ),
#         ("human", "Query: {query}")
#     ])

#     decision = (prompt | structured_llm).invoke({"query": state["query"]})
#     print(f"[router_node] Dynamic Route Target Set ──► '{decision.route}' | Reason: {decision.reason}")
#     return {**state, "route": decision.route}


# # ── Node NL2SQL: Credit Card Database Query Translation ───────────────────────
# def nl2sql_node(state: RAGState) -> RAGState:
#     llm = _get_llm()
#     db = get_sql_database()
#     schema_info = db.get_table_info()

#     sql_prompt = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             """You are a PostgreSQL expert. Given the database schema below,
#             write a single valid SELECT query that answers the user's question.

#             Rules:
#             - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
#             - Use only the tables and columns present in the schema.
#             - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
#             - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
            
#             CRITICAL QUERY STYLE RULES:
#             1. Whenever you compute a spend aggregation or total dollar amount (e.g., USING SUM(amount)) 
#                grouped by metrics like merchant_name, card_id, or transaction categories, you MUST 
#                ALWAYS include the transaction count as well using exactly `COUNT(*) AS txns` inside the 
#                SELECT statement parameters.
            
#             Schema info:
#             {schema}"""
#         ),
#         ("human", "Question: {question}")
#     ])

#     raw_sql = (sql_prompt | llm).invoke({"schema": schema_info, "question": state["query"]})
#     generated_sql = raw_sql.content.strip().strip("```").strip().replace("sql\n", "")

#     try:
#         print("Running the query..")
#         sql_result = db.run(generated_sql)
#         print(" query. executed.")
#     except Exception as exc:
#         sql_result = f"Postgres query runtime error: {exc}"
#         print(sql_result)

#     structured_llm = llm.with_structured_output(AIResponse)
#     answer_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are an analytical credit support expert. Answer the user prompt explicitly using the database records below."),
#         ("human", "Question: {query}\nSQL Used:\n{sql}\nQuery Results:\n{result}")
#     ])

#     answer = (answer_prompt | structured_llm).invoke({
#         "query": state["query"], "sql": generated_sql, "result": sql_result
#     })
    
#     response = answer.model_dump()
#     response["policy_citations"] = "N/A"
#     response["sql_query_executed"] = generated_sql
    
#     return {
#         **state,
#         "generated_sql": generated_sql,
#         "sql_result": str(sql_result),
#         "response": response
#     }


# # ── Node 1B: Document Agent Call Definition (Tool Routing Node) ──────────────
# def document_agent_node(state: RAGState) -> Dict[str, Any]:
#     """Evaluates the state context query and selects the optimal document lookup tool function."""
#     print("[document_agent_node] Deciding optimal retrieval search tools...")
#     llm = _get_llm()
    
#     # Bind our distinct set of tools
#     tools_list = list(DOCUMENT_TOOLS.values())
#     llm_with_tools = llm.bind_tools(tools_list)
    
#     system_directive = (
#         "You are an information extraction coordinator. Select the single best search tool to retrieve context for the user query:\n"
#         "- Call `keyword_search` for strict IDs, codes, numeric terms, short acronyms, or specific document names.\n"
#         "- Call `vector_search` for descriptive, natural-language, or scenario-based queries.\n"
#         "- Call `hybrid_search` for ambiguous queries or queries containing a mix of technical jargon and description.\n\n"
#         "Execute your selected tool immediately by creating an explicit tool call sequence."
#     )
    
#     messages = [
#         SystemMessage(content=system_directive),
#         HumanMessage(content=state["query"])
#     ]
    
#     # Keep historical thread tracking to preserve loop state context
#     if "agent_history" in state and state["agent_history"]:
#         messages.extend(state["agent_history"])
        
#     ai_msg = llm_with_tools.invoke(messages)
    
#     history = state.get("agent_history", []) if state.get("agent_history") is not None else []
#     history.append(ai_msg)
    
#     return {"agent_history": history, "tool_calls": ai_msg.tool_calls}


# # ── Node 1C: Dynamic Tool Sub-Execution Node Engine ───────────────────────────
# def execute_tools_node(state: RAGState) -> Dict[str, Any]:
#     """Iterates through and runs the functional tool parameters requested by the LLM."""
#     tool_calls = state.get("tool_calls", [])
#     history = state.get("agent_history", []) if state.get("agent_history") is not None else []
#     retrieved_raw_chunks = state.get("retrieved_docs", []) if state.get("retrieved_docs") is not None else []
    
#     for tool_call in tool_calls:
#         name = tool_call["name"]
#         args = tool_call["args"]
#         call_id = tool_call["id"]
        
#         if name in DOCUMENT_TOOLS:
#             results = DOCUMENT_TOOLS[name].invoke(args)
            
#             # Map structural dictionary lists directly back to standard LangChain Document instances
#             for chunk in results:
#                 retrieved_raw_chunks.append(
#                     Document(page_content=chunk["content"], metadata=chunk["metadata"])
#                 )
            
#             history.append(ToolMessage(content=json.dumps(results), tool_call_id=call_id, name=name))
            
#     print(f"[execute_tools_node] Successfully pulled {len(retrieved_raw_chunks)} document blocks.")
#     return {"retrieved_docs": retrieved_raw_chunks, "agent_history": history, "tool_calls": []}


# # ── Node 2: Cohere Cross-Encoder Reranking ────────────────────────────────────
# def rerank_node(state: RAGState) -> RAGState:
#     co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
#     docs = state["retrieved_docs"]

#     if not docs:
#         return {**state, "reranked_docs": []}

#     # Deduplicate matching string contexts to protect cross-encoder context boundaries
#     seen = set()
#     unique_docs = []
#     for d in docs:
#         if d.page_content not in seen:
#             seen.add(d.page_content)
#             unique_docs.append(d)

#     # Note: Reads from standard COHERE_API_KEY environment binding
#     rerank_response = co.rerank(
#         model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
#         query=state["query"],
#         documents=[doc.page_content for doc in unique_docs],
#         top_n=10
#     )

#     reranked_docs = [unique_docs[r.index] for r in rerank_response.results]
#     print(f"[rerank_node] Filtered down to top {len(reranked_docs)} high-confidence components.")
#     return {**state, "reranked_docs": reranked_docs}


# # ── Node 3: OpenAI Multimodal Vision Generation ─────────────────────────────
# def generate_answer_node(state: RAGState) -> RAGState:
#     llm = _get_llm()
#     structured_llm = llm.with_structured_output(AIResponse)

#     system_instruction = (
#         "You are an expert credit policy analyst. Answer the question using ONLY the provided text, table values, or diagrams.\n\n"
#         "CRITICAL: The context may contain chunks from MULTIPLE versions of the same document "
#         "(e.g., a 2025 edition and a 2026 edition). When answers differ across versions, do NOT pick only one:\n"
#         "  - Lead with the most recent / current version's answer (highest year).\n"
#         "  - Explicitly note how earlier versions differed (e.g., 'As of the 2026 policy...; previously, under 2025...').\n"
#         "  - If all versions agree, simply provide the single consolidated answer.\n\n"
#         "Citation structural layout rules:\n"
#         "  - document_name: exact filenames used.\n"
#         "  - page_no: associated page metrics.\n"
#         "  - policy_citations: formatted human readable citation references (e.g. 'Handbook_2026.pdf, Page 12')."
#     )

#     multimodal_payload_blocks = []
    
#     for idx, doc in enumerate(state["reranked_docs"]):
#         meta = doc.metadata
#         modality = str(meta.get("content_type", "text")).upper()
#         doc_name = meta.get("document_name")
#         page_num = meta.get("page")
        
#         block_header = f"\n\n--- [Context Block {idx+1} | Source: {doc_name} | Page: {page_num} | Modality: {modality}] ---\n"
#         block_content = block_header + doc.page_content
#         multimodal_payload_blocks.append({"type": "text", "text": block_content})
        
#         if meta.get("content_type") == "image" and meta.get("image_base64"):
#             multimodal_payload_blocks.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/png;base64,{meta['image_base64']}",
#                     "detail": "low"
#                 }
#             })

#     multimodal_payload_blocks.append({"type": "text", "text": f"\n\nUser Question to resolve: {state['query']}"})

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_instruction),
#         ("human", multimodal_payload_blocks)
#     ])

#     result = (prompt | structured_llm).invoke({})
#     return {**state, "response": result.model_dump()}


# # ── Assemble Graph State Machine ──────────────────────────────────────────────
# def build_rag_graph():
#     graph = StateGraph(RAGState)

#     # Register nodes
#     graph.add_node("router", router_node)
#     graph.add_node("nl2sql", nl2sql_node)
#     graph.add_node("document_agent", document_agent_node)
#     graph.add_node("execute_tools", execute_tools_node)
#     graph.add_node("rerank", rerank_node)
#     graph.add_node("generate_answer", generate_answer_node)

#     graph.set_entry_point("router")

#     # Entry point conditional routing edge split
#     graph.add_conditional_edges(
#         "router",
#         lambda state: state["route"],
#         {"product": "nl2sql", "document": "document_agent"}
#     )

#     # Document routing sub-loop logic based on active tool presence
#     graph.add_conditional_edges(
#         "document_agent",
#         lambda state: "execute" if state.get("tool_calls") else "rerank",
#         {
#             "execute": "execute_tools",
#             "rerank": "rerank"
#         }
#     )

#     # Re-evaluate route loop closure pattern 
#     graph.add_edge("execute_tools", "document_agent")
    
#     # Terminal edges
#     graph.add_edge("nl2sql", END)
#     graph.add_edge("rerank", "generate_answer")
#     graph.add_edge("generate_answer", END)

#     try:
#         compiled_agent = graph.compile()
#         graph_image = compiled_agent.get_graph().draw_mermaid_png()
#         os.makedirs("src/data", exist_ok=True)
#         with open("src/data/hybrid_workflow.png", "wb") as f:
#             f.write(graph_image)
#     except Exception:
#         pass

#     return compiled_agent

# rag_graph = build_rag_graph()

# def run_search_agent(query: str) -> dict:
#     initial_state = {
#         "query": query,
#         "retrieved_docs": [],
#         "reranked_docs": [],
#         "response": {},
#         "route": "",
#         "generated_sql": "",
#         "sql_result": "",
#         "agent_history": [],
#         "tool_calls": []
#     }
#     final_state = rag_graph.invoke(initial_state)
#     return final_state["response"]

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
from pydantic import BaseModel

from src.api.v1.schemas.query_schema import AIResponse
from src.api.v1.tools.tools import RAGState, vector_search, keyword_search, hybrid_search
from src.core.db import get_sql_database

load_dotenv(override=True)

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"), # Updated to stable deployment target variant
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

# Document retrieval tool directory dictionary map
DOCUMENT_TOOLS = {
    "vector_search": vector_search,
    "keyword_search": keyword_search,
    "hybrid_search": hybrid_search
}

# ── Node 0: Multi-Task Query Router & Tool Selection Node ───────────────────
class _RouteDecision(BaseModel):
    route: Literal["product", "vector_search", "keyword_search", "hybrid_search"]
    reason: str

def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an enterprise query router and technical retrieval coordinator. 
            Classify incoming customer prompts into EXACTLY one target path option:

            1. "product"  — Queries requiring retrieval or calculation of live customer data. 
                            This includes:
                            - Customer profiles (demographics, KYC status).
                            - Credit card account status (limits, outstanding balance, due dates).
                            - Transaction history (purchases, EMI details, refunds, category spending).
                            - Reward point balances and redemption history.
                            - Billing statement summaries (monthly balances, payments, fees).
                            (Database scope: customers, credit_cards, card_transactions, reward_transactions, billing_statements)

            Queries regarding knowledge base policies, product guidelines, or instructional content.
                    This includes:
                    - Bank product tier features (Classic, Gold, Platinum, Signature).
                    - Fee schedules and penalty structures.
                    - Reward program earning rules and expiry policies.
                    - Spend categorization logic and Merchant Category Code (MCC) definitions.
                    - Billing cycle rules and payment processing guidelines.
                    (Knowledge base scope: NorthStar Bank Credit Card Product Guide v2.1)
            Then, select the best search tool:
            2. "vector_search"  — For long, scenario-based descriptive questions or natural-language concepts.
            3. "keyword_search" — For queries targeting precise codes, strict acronyms, policy IDs, metrics, or specific numbers.
            4. "hybrid_search"  — For complex, ambiguous queries containing mixed technical keywords and situational text.

            Ensure absolute accuracy when mapping target routes."""
        ),
        ("human", "Query: {query}")
    ])

    decision = (prompt | structured_llm).invoke({"query": state["query"]})
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
            """You are a PostgreSQL expert. Given the database schema below,
            write a single valid SELECT query that answers the user's question.

            Rules:
            - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
            - Use only the tables and columns present in the schema.
            - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
            - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
            
            CRITICAL QUERY STYLE RULES:
            1. Whenever you compute a spend aggregation or total dollar amount (e.g., USING SUM(amount)) 
               grouped by metrics like merchant_name, card_id, or transaction categories, you MUST 
               ALWAYS include the transaction count as well using exactly `COUNT(*) AS txns` inside the 
               SELECT statement parameters.
            
            Schema info:
            {schema}"""
        ),
        ("human", "Question: {question}")
    ])

    raw_sql = (sql_prompt | llm).invoke({"schema": schema_info, "question": state["query"]})
    generated_sql = raw_sql.content.strip().strip("```").strip().replace("sql\n", "")

    try:
        print("Running the query..")
        sql_result = db.run(generated_sql)
        print(" query. executed.")
    except Exception as exc:
        sql_result = f"Postgres query runtime error: {exc}"
        print(sql_result)

    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an analytical credit support expert. Answer the user prompt explicitly using the database records below."),
        ("human", "Question: {query}\nSQL Used:\n{sql}\nQuery Results:\n{result}")
    ])

    answer = (answer_prompt | structured_llm).invoke({
        "query": state["query"], "sql": generated_sql, "result": sql_result
    })
    
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql
    
    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response
    }


# ── Node 1C: Direct Tool Execution Engine ─────────────────────────────────────
def execute_tools_node(state: RAGState) -> Dict[str, Any]:
    """Dynamically routes execution based on the router choice and forces an exit directly toward the rerank phase."""
    target_tool = state.get("route")
    retrieved_raw_chunks = []
    
    if target_tool in DOCUMENT_TOOLS:
        print(f"[execute_tools_node] Executing non-looping direct call to: {target_tool}")
        # Invoke the targeted tool directly using the original user query parameters
        results = DOCUMENT_TOOLS[target_tool].invoke({"query": state["query"]})
        
        for chunk in results:
            retrieved_raw_chunks.append(
                Document(page_content=chunk["content"], metadata=chunk["metadata"])
            )
            
    print(f"[execute_tools_node] Extracted {len(retrieved_raw_chunks)} chunks. Sending directly to rerank.")
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
        "You are an expert credit policy analyst. Answer the question using ONLY the provided text, table values, or diagrams.\n\n"
        "CRITICAL: The context may contain chunks from MULTIPLE versions of the same document:\n"
        "  - Lead with the most recent / current version's answer (highest year).\n"
        "  - Explicitly note how earlier versions differed.\n"
        "  - If all versions agree, simply provide the single consolidated answer.\n\n"
        "Citation structural layout rules:\n"
        "  - document_name: exact filenames used.\n"
        "  - page_no: associated page metrics.\n"
        "  - policy_citations: formatted human readable citation references (e.g. 'Handbook_2026.pdf, Page 12')."
    )

    multimodal_payload_blocks = []
    
    for idx, doc in enumerate(state["reranked_docs"]):
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

    multimodal_payload_blocks.append({"type": "text", "text": f"\n\nUser Question to resolve: {state['query']}"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", multimodal_payload_blocks)
    ])

    result = (prompt | structured_llm).invoke({})
    return {**state, "response": result.model_dump()}


# ── Assemble Graph State Machine ──────────────────────────────────────────────
def build_rag_graph():
    graph = StateGraph(RAGState)

    # Register flattened node topology map
    graph.add_node("router", router_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    # 1. Router evaluate condition edge split setup
    def _route_resolver(state: RAGState) -> str:
        if state["route"] == "product":
            return "nl2sql"
        return "execute_tools"

    graph.add_conditional_edges(
        "router",
        _route_resolver,
        {
            "nl2sql": "nl2sql",
            "execute_tools": "execute_tools"
        }
    )

    # 2. Sequential pipeline links (No more back-tracking agent loops)
    graph.add_edge("execute_tools", "rerank")
    graph.add_edge("rerank", "generate_answer")
    
    # 3. Graph terminal paths
    graph.add_edge("nl2sql", END)
    graph.add_edge("generate_answer", END)

    try:
        compiled_agent = graph.compile()
        graph_image = compiled_agent.get_graph().draw_mermaid_png()
        os.makedirs("src/data", exist_ok=True)
        with open("src/data/final_flow.png", "wb") as f:
            f.write(graph_image)
    except Exception:
        pass

    return compiled_agent

rag_graph = build_rag_graph()

def run_search_agent(query: str) -> dict:
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "route": "",
        "generated_sql": "",
        "sql_result": "",
        "agent_history": [],
        "tool_calls": []
    }
    final_state = rag_graph.invoke(initial_state)
    return final_state["response"]