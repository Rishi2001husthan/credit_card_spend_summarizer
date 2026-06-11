import os
import base64
import time
import psycopg
from psycopg.rows import dict_row
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool

# Native connections using your custom db module singletons
from src.core.db import _embed_texts, get_db_conn

load_dotenv(override=True)

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]   # Candidate pool
    reranked_docs: List[Document]    # Final filtered elements
    response: dict                   # Structured output schema 
    route: str                       # Routing choice: "product" or "document"
    generated_sql: str               # Executed text-to-sql query
    sql_result: str                  # DB output string
    agent_history: List[Any]         # Message track history for agent loop
    tool_calls: List[Dict[str, Any]] # Active tool execution tasks

# Fallback parameter if COLLECTION_NAME is missing in .env
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "credit_policy_documents")


# ── Internal Helper: Database Row to Multimodal Document Mapper ───────────────
def _map_row_to_document(row: dict) -> Dict[str, Any]:
    """
    Normalizes SQL dictionary data rows into standardized dictionary outputs,
    safely re-encoding local filesystem images back to Base64 strings.
    """
    img_b64 = None
    local_path = row.get("image_path")
    
    if local_path and os.path.exists(local_path):
        try:
            with open(local_path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            print(f"[tools] Warning: Failed to parse visual asset at {local_path}: {e}")

    return {
        "content": row.get("content") or "",
        "metadata": {
            "content_type": row.get("chunk_type") or "text",
            "page": row.get("page_number") if row.get("page_number") is not None else 1,
            "section": row.get("section") or "General",
            "source_file": row.get("source_file") or "Unknown",
            "document_name": os.path.basename(row.get("source_file")) if row.get("source_file") else "Unknown",
            "image_base64": img_b64
        }
    }


# ── Tool 1: Semantic Vector Search ────────────────────────────────────────────
@tool
def vector_search(query: str) -> list:
    """Use this for long natural language queries where semantic meaning, context, and intent matter."""
    print("🤖 [Agent Action] Triggering RATE-RESILIENT VECTOR semantic search...")
    max_retries = 3
    initial_delay = 2.0
    
    # 1. Transform query using the embedding configuration mapping
    try:
        query_vector = _embed_texts([query])[0]
        vector_string = "[" + ",".join(str(v) for v in query_vector) + "]"
    except Exception as e:
        print(f"❌ Embedding Generation Error: {e}")
        return []

    # 2. Query multimodal chunk vectors using pgvector cosine distance operators
    sql = """
        SELECT chunk_type, element_type, content, image_path, page_number, section, source_file
        FROM multimodal_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 5;
    """
    
    for attempt in range(max_retries):
        try:
            with get_db_conn() as conn:
                # Force row data mapping out as key-value dictionaries
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, (vector_string,))
                    rows = cur.fetchall()
            return [_map_row_to_document(row) for row in rows]
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RATE_LIMIT_EXCEEDED" in error_msg:
                if attempt == max_retries - 1:
                    return []
                time.sleep(initial_delay * (2 ** attempt))
            else:
                print(f"❌ pgvector Query Runtime Error: {e}")
                return []
    return []


# ── Tool 2: Full-Text Keyword Search ──────────────────────────────────────────
@tool
def keyword_search(query: str) -> list:
    """Use this for exact keyword queries like product codes, terms, abbreviations, or structural section IDs."""
    print("🔍 [Agent Action] Triggering KEYWORD full-text search layout...")
    
    sql = """
        SELECT chunk_type, element_type, content, image_path, page_number, section, source_file,
               ts_rank(to_tsvector('english', content), plainto_tsquery('english', %(query)s)) AS fts_rank
        FROM multimodal_chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC 
        LIMIT 5;
    """
    try:
        with get_db_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, {"query": query})
                rows = cur.fetchall()
        return [_map_row_to_document(row) for row in rows]
    except Exception as e:
        print(f"❌ Full-Text Search Error: {e}")
        return []


# ── Tool 3: Hybrid Reciprocal Rank Fusion (RRF) Search ────────────────────────
@tool
def hybrid_search(query: str) -> list:
    """Use this for short, tricky, or ambiguous queries that require a blend of keyword matching and semantic context."""
    print("🧬 [Agent Action] Triggering HYBRID RRF search layout...")
    
    # 1. Fetch exact matching pool variants from both tools
    vector_results = vector_search.invoke({"query": query})
    keyword_results = keyword_search.invoke({"query": query})
    
    rrf_scores = {}
    chunk_map = {}

    # 2. Score Vector Results
    for rank, doc in enumerate(vector_results):
        key = doc["content"][:120]  # Deduplication token anchor key
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1.0)
        chunk_map[key] = doc

    # 3. Score Full-Text Keyword Results and merge
    for rank, doc in enumerate(keyword_results):
        key = doc["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1.0)
        chunk_map[key] = doc
    
    # 4. Sort and return the highest scored unified objects
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:5]]