import os
import shutil
from fastapi import APIRouter
from src.api.v1.services.query_service   import query_documents
from src.api.v1.schemas.query_schema import QueryRequest
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from src.ingestion.ingestion import run_ingestion

router = APIRouter()

@router.post("/query")
def query_endpoint(request: QueryRequest):
    docs = query_documents(request.query)
    return docs


@router.post("/api/v1/ingest")
async def upload_and_ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default_collection")
):
    """
    Receives standard multi-part form file data streams directly over HTTP networks, 
    stages them locally inside a temp directory, and passes them to the Docling pipeline.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only standard document PDF formats are supported.")
        
    try:
        # Create staging buffer directories securely
        temp_dir = os.path.join("src", "data", "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        staged_file_path = os.path.join(temp_dir, file.filename)
        
        # Save structural stream segments down onto disk blocks
        with open(staged_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Execute your core ingestion workflow script logic
        ingestion_result = run_ingestion(staged_file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "collection": collection_name,
            "details": ingestion_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing execution error: {str(e)}")
