import os
import pathlib


from dotenv import load_dotenv


from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document


load_dotenv()


# ---------------------------------------------------------------------------
# Chunking configuration
#
# Long text elements (paragraphs that span half a page or more) are split into
# overlapping windows so that a single dense paragraph doesn't dominate a
# retrieval result and context from surrounding sentences is preserved.
#
# _TEXT_CHUNK_SIZE    — maximum characters per chunk
# _TEXT_CHUNK_OVERLAP — characters shared between adjacent chunks so that
#                       sentences cut at a boundary still appear in both chunks
# Tables and images are never split — they must be stored as atomic units.
# ---------------------------------------------------------------------------
_TEXT_CHUNK_SIZE = 1000
_TEXT_CHUNK_OVERLAP = 200




def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
   """Split a long string into overlapping character windows.


   Splitting strategy:
     - Walks through the text in steps of (chunk_size - overlap)
     - Each window is exactly chunk_size characters (or shorter at the end)
     - The overlap ensures sentences cut at a boundary appear in both the
       preceding and following chunk, preserving retrieval context


   This is a lightweight alternative to langchain_text_splitters which is
   not installed in this environment.
   """
   chunks: list[str] = []
   start = 0
   step = chunk_size - overlap
   while start < len(text):
       chunks.append(text[start : start + chunk_size])
       start += step
   return chunks




def run_ingestion(file_path: str) -> dict:
   """Run the full ingestion pipeline for a single PDF file.


   Steps:
     1. Register the document in the `documents` table → get a stable doc_id
     2. Parse PDF with Docling → typed elements (text / table / image)
     3. Split long text elements into overlapping chunks
     4. Embed all chunks and store in `multimodal_chunks` via db.store_chunks()


   Args:
       file_path: Absolute or relative path to the source PDF.


   Returns:
       Dict with "status", "doc_id", and "chunks_ingested" count.
   """
   resolved = pathlib.Path(file_path).resolve()


   # ── Step 1: Register (or update) the document record ─────────────────────
   doc_id = upsert_document(resolved.name, str(resolved))
   print(f"[ingestion] doc_id={doc_id}  file={file_path}")


   # ── Step 2: Parse the PDF ─────────────────────────────────────────────────
   print(f"[ingestion] Parsing: {file_path}")
   parsed_elements = parse_document(file_path)
   print(f"[ingestion] Docling produced {len(parsed_elements)} elements")


   # ── Step 3: Split long text elements into overlapping chunks ──────────────
   chunks: list[dict] = []
   for elem in parsed_elements:
       if elem["content_type"] == "text" and len(elem["content"]) > _TEXT_CHUNK_SIZE:
           for sub in _split_text(elem["content"], _TEXT_CHUNK_SIZE, _TEXT_CHUNK_OVERLAP):
               # Each sub-chunk inherits the parent element's full metadata
               chunks.append({
                   "content": sub,
                   "content_type": elem["content_type"],
                   "metadata": elem["metadata"],
               })
       else:
           chunks.append(elem)


   print(f"[ingestion] {len(chunks)} chunks ready for embedding")


   # ── Step 4: Embed chunks and store in multimodal_chunks ───────────────────
   count = store_chunks(chunks, doc_id)
   print(f"[ingestion] Stored {count} chunks → multimodal_chunks")


   return {"status": "success", "doc_id": doc_id, "chunks_ingested": count}


if __name__ == "__main__":
   import sys


   if len(sys.argv) >= 2:
       pdf_path = pathlib.Path(sys.argv[1])
   else:
       pdf_path = pathlib.Path("data/KB_Credit_Card_Spend_Summarizer.pdf")


   if not pdf_path.exists():
       raise FileNotFoundError(f"PDF not found at: {pdf_path.resolve()}")


   result = run_ingestion(str(pdf_path))
   print(f"\nIngestion complete: {result}")


