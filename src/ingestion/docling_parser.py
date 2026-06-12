import base64
import io
import os


from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from docling.datamodel.pipeline_options import (
   AcceleratorDevice,
   AcceleratorOptions,
   PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from src.ingestion.image import get_image_description


load_dotenv()


def parse_document(file_path: str) -> list[dict]:
   """Parse a PDF into a flat list of typed content chunks using Docling.


   Each chunk is a dict with three keys:
     content      — text or markdown representation of the element
     content_type — one of: "text", "table", "image"
     metadata     — dict with: content_type, element_type, section,
                    page_number, source_file, image_base64


   The metadata is passed to PGVector, so every
   retrieved chunk tells the query layer what kind of content it is
   and where in the document it came from.
   """


   pipeline_options = PdfPipelineOptions(
       do_ocr=True,
       do_table_structure=True,
       generate_picture_images=True,
       accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
   )


   converter = DocumentConverter(
       allowed_formats=[InputFormat.PDF],
       format_options={
           InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
       },
   )


   # ── Step 2: Convert the PDF ───────────────────────────────────────────────
   result = converter.convert(file_path)
   doc = result.document


   parsed_chunks: list[dict] = []
   current_section: str | None = None
   source_file = os.path.basename(file_path)


   # ── Step 3: Walk the document element tree ────────────────────────────────
   for item in doc.iterate_items():
       if isinstance(item, tuple):
           node, _ = item  # iterate_items() yields (node, level); discard level
       else:
           node = item     # older Docling versions yield bare nodes


       label = str(getattr(node, "label", "")).lower()


       # ── Skip page headers/footers ─────────────────────────────────────────
       if label in ("page_header", "page_footer"):
           continue


       # ── Extract page number and bounding box from provenance ──────────────
       prov = getattr(node, "prov", None)
       page_no = prov[0].page_no if prov else None
       position: dict | None = None
       if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
           b = prov[0].bbox
           position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}


       def _make_metadata(content_type: str, element_type: str, img_b64=None):
           """Build a metadata dict that is stored alongside every chunk.


           content_type  — "text" | "table" | "image"  (used by the query
                           layer to decide how to render retrieved content)
           element_type  — raw Docling label ("section_header", "table", …)
           img_b64       — base64-encoded PNG string for image elements;
                           None for text and table elements
           """
           return {
               "content_type": content_type,
               "element_type": element_type,
               "section": current_section,
               "page_number": page_no,
               "source_file": source_file,
               "position": position,       # bounding box stored in JSONB position column
               "image_base64": img_b64,    # decoded to BYTEA by db.store_chunks()
           }


       # ── Section headings & document title ─────────────────────────────────
       if "section_header" in label or label == "title":
           text = getattr(node, "text", "").strip()
           if text:
               current_section = text
               parsed_chunks.append(
                   {
                       "content": text,
                       "content_type": "text",
                       "metadata": _make_metadata("text", label),
                   }
               )


       # ── Tables ────────────────────────────────────────────────────────────
       elif "table" in label:
           table_text = ""
           if hasattr(node, "export_to_dataframe"):
               try:
                   df = node.export_to_dataframe()
                   if df is not None and not df.empty:
                       rows_text: list[str] = []
                       headers = [str(c).strip() for c in df.columns]
                       for _, row in df.iterrows():
                           pairs = [
                               f"{h}: {str(v).strip()}"
                               for h, v in zip(headers, row)
                               if str(v).strip() not in ("", "nan", "None")
                           ]
                           if pairs:
                               rows_text.append("  |  ".join(pairs))
                       table_text = "\n".join(rows_text)
               except Exception:
                   pass


           # Fallback: strip HTML tags from export_to_html()
           if not table_text and hasattr(node, "export_to_html"):
               try:
                   import re as _re
                   raw_html = node.export_to_html(doc)
                   table_text = _re.sub(r"<[^>]+>", " ", raw_html or "")
                   table_text = _re.sub(r"\s+", " ", table_text).strip()
               except Exception:
                   pass


           # Last resort: raw text attribute
           if not table_text:
               table_text = getattr(node, "text", "")


           if table_text and table_text.strip():
               parsed_chunks.append(
                   {
                       "content": table_text.strip(),
                       "content_type": "table",
                       "metadata": _make_metadata("table", "table"),
                   }
               )


       # ── Pictures, figures, and charts ─────────────────────────────────────
       elif "picture" in label or "figure" in label or label == "chart":
           img_b64 = None
           # .text on a PictureItem is the inline caption, if any
           caption = getattr(node, "text", "") or ""


           try:
               if hasattr(node, "get_image"):
                   pil_img = node.get_image(doc)
                   if pil_img:
                       buf = io.BytesIO()
                       pil_img.save(buf, format="PNG")
                       img_b64 = base64.b64encode(buf.getvalue()).decode()


               # Fallback path for older Docling versions
               if img_b64 is None and hasattr(node, "image") and node.image:
                   pil_img = getattr(node.image, "pil_image", None)
                   if pil_img:
                       buf = io.BytesIO()
                       pil_img.save(buf, format="PNG")
                       img_b64 = base64.b64encode(buf.getvalue()).decode()
           except Exception:
               # Image extraction is best-effort; a missing image is not
               # fatal — the caption / placeholder text is still indexed.
               pass


           # Use an OpenAI vision model to generate a rich description for this
           # image. This becomes the chunk's searchable text content — far more
           # useful than a sparse caption like "Figure 3" for embedding and
           # retrieval. Falls back to Docling caption → placeholder on failure.
           if img_b64:
               description = get_image_description(img_b64, caption)
               content = description
           else:
               content = caption.strip() or f"[Image on page {page_no}]"
           parsed_chunks.append(
               {
                   "content": content,
                   "content_type": "image",
                   "metadata": _make_metadata("image", "picture", img_b64),
               }
           )


       # ── Plain text: paragraphs, list items, captions, footnotes, etc. ─────
       else:
           text = getattr(node, "text", "")
           if text and text.strip():
               parsed_chunks.append(
                   {
                       "content": text.strip(),
                       "content_type": "text",
                       "metadata": _make_metadata("text", label),
                   }
               )


   return parsed_chunks
