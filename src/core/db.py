import base64
import hashlib
import json
import os
import pathlib


import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase


load_dotenv()


_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg2://", "postgresql://")


_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


_embeddings = OpenAIEmbeddings(
   model=_EMBED_MODEL,
   api_key=os.getenv("OPENAI_API_KEY"),
   # default is 1536, when you not set this
)


def _embed_texts(texts: list[str]) -> list[list[float]]:
   """Embed a batch of text strings with OpenAI text-embedding-3-small.


   OpenAIEmbeddings handles request batching internally, so we pass the whole
   list and get back one 1536-dimensional vector per input string.
   """
   return _embeddings.embed_documents(texts)


_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
   """Return the module-level connection pool, creating it on first call."""
   global _pool
   if _pool is None:
       _pool = ConnectionPool(
           _PG_DSN,
           min_size=2,
           max_size=10,
           kwargs={"row_factory": dict_row},
       )
   return _pool


def get_db_conn():
   """Return a pooled connection context manager.


   Usage:
       with get_db_conn() as conn:
           with conn.cursor() as cur: ...
   """
   return _get_pool().connection()


def upsert_document(filename: str, source_path: str) -> str:
   """Insert a document record and return its UUID.


   Uses ON CONFLICT so re-ingesting the same filename updates the path
   and returns the *existing* doc_id rather than creating a duplicate.
   This makes ingestion idempotent at the document level.
   """
   with get_db_conn() as conn:
       with conn.cursor() as cur:
           cur.execute(
               """
               INSERT INTO documents (filename, source_path)
               VALUES (%s, %s)
               ON CONFLICT (filename) DO UPDATE
                   SET source_path = EXCLUDED.source_path,
                       ingested_at  = now()
               RETURNING id
               """,
               (filename, source_path),
           )
           row = cur.fetchone()
       conn.commit()
   return str(row["id"])


def store_chunks(chunks: list[dict], doc_id: str) -> int:
   """Embed each chunk and insert it into the multimodal_chunks table.


   Args:
       chunks:  List of dicts produced by parse_document() / ingestion.py.
                Each dict must have: content (str), content_type (str),
                metadata (dict with page_number, section, source_file,
                element_type, position, image_base64).
       doc_id:  UUID string of the parent document (from upsert_document).


   Returns:
       Number of rows inserted.


   Embedding strategy:
       Every chunk — text, table, and image — is embedded from its `content`
       text via _embed_texts() (OpenAI text-embedding-3-small). Image chunks
       carry a vision-generated description as their content, so they remain
       retrievable by natural-language queries even though OpenAI embeddings
       cannot read pixels directly.


   Vector storage:
       pgvector accepts the '[f1,f2,…]' string literal when cast with
       ::vector. We build that string directly to avoid needing the
       separate pgvector Python package.


   Image storage:
       image_base64 from metadata is decoded to raw bytes and stored in
       the BYTEA column. The JSONB metadata column does NOT duplicate it,
       keeping metadata lean.
   """
   if not chunks:
       return 0


   all_embeddings = _embed_texts([chunk["content"] for chunk in chunks])


   _DEDICATED_COLUMNS = {
       "content_type", "element_type", "section",
       "page_number", "source_file", "position", "image_base64",
   }


   rows_inserted = 0
   with get_db_conn() as conn:
       with conn.cursor() as cur:
           cur.execute(
               "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
               (doc_id,),
           )


           for chunk, embedding in zip(chunks, all_embeddings):
               meta = chunk["metadata"]

               img_b64 = meta.get("image_base64")
               image_path: str | None = None
               mime_type = "image/png" if img_b64 else None
               if img_b64:
                   image_bytes = base64.b64decode(img_b64)
                   img_dir = pathlib.Path("data/images")
                   img_dir.mkdir(parents=True, exist_ok=True)
                   img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                   img_file = img_dir / f"{doc_id}_{img_hash}.png"
                   img_file.write_bytes(image_bytes)
                   image_path = str(img_file)


               embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"


               clean_meta = {k: v for k, v in meta.items() if k not in _DEDICATED_COLUMNS}


               cur.execute(
                   """
                   INSERT INTO multimodal_chunks (
                       doc_id, chunk_type, element_type, content,
                       image_path, mime_type,
                       page_number, section, source_file,
                       position, embedding, metadata
                   ) VALUES (
                       %s::uuid, %s, %s, %s,
                       %s, %s,
                       %s, %s, %s,
                       %s::jsonb, %s::vector, %s::jsonb
                   )
                   """,
                   (
                       doc_id,
                       chunk["content_type"],       # chunk_type column
                       meta.get("element_type"),    # raw Docling label
                       chunk["content"],            # text / markdown / caption
                       image_path,                  # filesystem path (None for text/table)
                       mime_type,
                       meta.get("page_number"),
                       meta.get("section"),
                       meta.get("source_file"),
                       json.dumps(meta.get("position")) if meta.get("position") else None,
                       embedding_str,               # ::vector cast
                       json.dumps(clean_meta),      # JSONB catch-all
                   ),
               )
               rows_inserted += 1
       conn.commit()


   return rows_inserted

def get_sql_database() -> SQLDatabase:
   """Return a LangChain SQLDatabase connected to the credit_card_db (read-only).
    Uses the credit_readonly role from sql/seed.sql - SELECT privileges only.
   Connection string is read from RDBMS_URL in the environment.
   """
   db_url = os.getenv("RDBMS_URL")
   if not db_url:
       raise ValueError("RDBMS_URL is not set. Check your .env file.")
   return SQLDatabase.from_uri(
       db_url,
       include_tables=["customers", "credit_cards", "card_transactions", "reward_transactions", "billing_statements"],
       sample_rows_in_table_info=2,
   )