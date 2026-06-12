import os
import base64
import time
import psycopg
from psycopg.rows import dict_row
from typing import TypedDict, List, Dict, Any, Annotated
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
# Import the native LangGraph message reducer helper
from langgraph.graph.message import add_messages

# Native connections using your validated db module singletons
from src.core.db import _embed_texts, get_db_conn

load_dotenv(override=True)

# ── Updated State Definition With Channel Reducers ────────────────────────────
class RAGState(TypedDict):
    query: str
    # Using add_messages ensures multi-turn user/assistant responses append automatically
    messages: Annotated[list, add_messages] 
    retrieved_docs: List[Document]   # Candidate pool
    reranked_docs: List[Document]    # Final filtered elements
    response: dict                   # Structured output schema 
    route: str                       # Routing choice: "product", "general", etc.
    generated_sql: str               # Executed text-to-sql query
    sql_result: str                  # DB output string
    agent_history: List[Any]         # Secondary trace logs if manually needed
    tool_calls: List[Dict[str, Any]] # Active tool execution tasks

# Read and sanitize the raw connection string for standard psycopg drivers
_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_raw_conn = _PG_CONNECTION.replace("postgresql+psycopg2://", "postgresql://")
COLLECTION_NAME = "KB_Credit_Card_Spend_Summarizer.pdf"


# ── Internal Helper: Database Row to Multimodal Document Mapper ───────────────
def _map_row_to_document(row: dict) -> Dict[str, Any]:
    """
    Normalizes SQL dictionary data rows into standardized dictionary outputs,
    safely re-encoding local filesystem images back to Base64 strings.
    """
    img_b64 = None
    metadata = row.get("metadata") or row.get("cmetadata") or {}
    local_path = metadata.get("image_path") if isinstance(metadata, dict) else None
    
    if local_path and os.path.exists(local_path):
        try:
            with open(local_path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            print(f"[tools] Warning: Failed to parse visual asset at {local_path}: {e}")

    return {
        "content": row.get("content") or row.get("document") or "",
        "metadata": {
            "content_type": metadata.get("chunk_type") or row.get("chunk_type") or "text",
            "page": metadata.get("page_number") or row.get("page_number") or 1,
            "section": metadata.get("section") or row.get("section") or "General",
            "source_file": metadata.get("source_file") or row.get("source_file") or COLLECTION_NAME,
            "document_name": os.path.basename(metadata.get("source_file") or COLLECTION_NAME),
            "image_base64": img_b64
        }
    }


# ── Tool 1: Vector Search ──────────────────────────────────────────────────────
@tool
def vector_search(query: str) -> list:
    """Use this for long natural language queries where semantic meaning, context, and intent matter."""
    print("[Agent Action] Triggering BI-ENCODER vector similarity search...")
    
    try:
        query_vector = _embed_texts([query])[0]
        vector_string = "[" + ",".join(str(v) for v in query_vector) + "]"
        
        sql = """
            SELECT chunk_type, element_type, content, image_path, page_number, section, source_file
            FROM multimodal_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT 20;
        """
        
        with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_string,))
                rows = cur.fetchall()
                
        print(f"[vector_search] Retrieved {len(rows)} chunks from PGVector native lookup")
        return [_map_row_to_document(row) for row in rows]
        
    except Exception as e:
        print(f"❌ Vector Search Operator Failure: {e}")
        return []


# ── Tool 2: Keyword Search (Full-Text Search Layout) ──────────────────────────
@tool
def keyword_search(query: str) -> list:
    """
    Performs a Full-Text Search (FTS) on the regulatory compliance documents using PostgreSQL.
    Useful for finding exact keyword matches or specific terminology.
    """
    print("🔍 [Agent Action] keyword-searching (FTS).....")
    
    sql = """
        SELECT chunk_type, element_type, content, image_path, page_number, section, source_file,
        ts_rank(to_tsvector('english', content), plainto_tsquery('english', %(query)s)) AS fts_rank
        FROM multimodal_chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC 
        LIMIT 5;
    """
    
    try:
        with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"query": query})
                rows = cur.fetchall()

        print(f"[keyword_search] Retrieved {len(rows)} chunks from PGVector")
        return [_map_row_to_document(row) for row in rows]
        
    except Exception as e:
        print(f"❌ Full-Text Keyword Search Error: {e}")
        return []


# ── Tool 3: Hybrid Search (RRF Interface) ─────────────────────────────────────
@tool
def hybrid_search(query: str) -> list:
    """Use this for short, tricky, or ambiguous queries that require a blend of keyword matching and semantic context."""
    print("[Agent Action] Triggering HYBRID RRF search layout...")
    
    vector_results = vector_search.invoke({"query": query})
    keyword_results = keyword_search.invoke({"query": query})
    
    rrf_scores = {}
    chunk_map = {}

    for rank, doc in enumerate(vector_results):
        key = doc["content"][:120]  
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1.0)
        chunk_map[key] = doc

    for rank, doc in enumerate(keyword_results):
        key = doc["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60.0 + rank + 1.0)
        chunk_map[key] = doc
    
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:10]]