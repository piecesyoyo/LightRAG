from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks, Depends
from typing import List
import logging
import traceback
import asyncio
from ..models import InsertResponse, InsertTextRequest, DocsStatusesResponse
from ..document_manager import DocumentManager, DocumentProcessor
from lightrag.base import DocStatus
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

router = APIRouter(tags=["documents"])

# Global variables for dependency injection
doc_manager: DocumentManager = None
doc_processor: DocumentProcessor = None
api_key_header = None

def init_routes(manager: DocumentManager, processor: DocumentProcessor, api_key: str = None):
    """
    Initialize routes with required dependencies
    
    Args:
        manager: DocumentManager instance
        processor: DocumentProcessor instance
        api_key: Optional API key for authentication
    """
    global doc_manager, doc_processor, api_key_header
    doc_manager = manager
    doc_processor = processor
    if api_key:
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key_header_value: str = Depends(api_key_header)):
    """
    Verify API key if authentication is enabled
    
    Args:
        api_key_header_value: API key from request header
        
    Raises:
        HTTPException: If API key is invalid or missing when required
    """
    if api_key_header and not api_key_header_value:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="API Key required")
    if api_key_header and api_key_header_value != api_key_header:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key")

@router.post("/documents/scan")
async def scan_for_new_documents(
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key)
):
    """
    Trigger document scanning process
    
    Args:
        background_tasks: FastAPI BackgroundTasks for async processing
        
    Returns:
        Dict containing scan status
    """
    with doc_processor.progress_lock:
        if doc_processor.scan_progress["is_scanning"]:
            return {"status": "already_scanning"}

        doc_processor.scan_progress.update({
            "is_scanning": True,
            "indexed_count": 0,
            "progress": 0
        })

    background_tasks.add_task(run_scanning_process)
    return {"status": "scanning_started"}

@router.get("/documents/scan-progress")
async def get_scan_progress():
    """
    Get current scanning progress
    
    Returns:
        Dict containing current scan progress
    """
    with doc_processor.progress_lock:
        return doc_processor.scan_progress

@router.post("/documents/upload")
async def upload_to_input_dir(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key)
):
    """
    Upload and process a single document
    
    Args:
        background_tasks: FastAPI BackgroundTasks for async processing
        file: File to be uploaded
        
    Returns:
        InsertResponse containing operation status
        
    Raises:
        HTTPException: If file type is not supported or processing fails
    """
    try:
        if not doc_manager.is_supported_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}"
            )

        temp_path = await doc_manager.save_temp_file(file)
        background_tasks.add_task(doc_processor.pipeline_index_files, [temp_path])

        return InsertResponse(
            status="success",
            message=f"File '{file.filename}' uploaded successfully. Processing will continue in background."
        )
    except Exception as e:
        logging.error(f"Error uploading file {file.filename}: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/text")
async def insert_text(
    request: InsertTextRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key)
):
    """
    Insert text content for processing
    
    Args:
        request: InsertTextRequest containing text to process
        background_tasks: FastAPI BackgroundTasks for async processing
        
    Returns:
        InsertResponse containing operation status
        
    Raises:
        HTTPException: If text processing fails
    """
    try:
        background_tasks.add_task(doc_processor.pipeline_index_texts, [request.text])
        return InsertResponse(
            status="success",
            message="Text successfully received. Processing will continue in background."
        )
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/batch")
async def insert_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    _: None = Depends(verify_api_key)
):
    """
    Process multiple files in batch mode
    
    Args:
        background_tasks: FastAPI BackgroundTasks for async processing
        files: List of files to process
        
    Returns:
        InsertResponse containing operation status
        
    Raises:
        HTTPException: If batch processing fails
    """
    try:
        inserted_count = 0
        failed_files = []
        temp_files = []

        for file in files:
            if doc_manager.is_supported_file(file.filename):
                temp_files.append(await doc_manager.save_temp_file(file))
                inserted_count += 1
            else:
                failed_files.append(f"{file.filename} (unsupported type)")

        if temp_files:
            background_tasks.add_task(doc_processor.pipeline_index_files, temp_files)

        # Prepare status message
        if inserted_count == len(files):
            status = "success"
            status_message = f"Successfully inserted all {inserted_count} documents"
        elif inserted_count > 0:
            status = "partial_success"
            status_message = f"Successfully inserted {inserted_count} out of {len(files)} documents"
            if failed_files:
                status_message += f". Failed files: {', '.join(failed_files)}"
        else:
            status = "failure"
            status_message = "No documents were successfully inserted"
            if failed_files:
                status_message += f". Failed files: {', '.join(failed_files)}"

        return InsertResponse(status=status, message=status_message)

    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def get_documents(_: None = Depends(verify_api_key)) -> DocsStatusesResponse:
    """
    Get status of all documents
    
    Returns:
        DocsStatusesResponse containing document statuses
        
    Raises:
        HTTPException: If status retrieval fails
    """
    try:
        statuses = (
            DocStatus.PENDING,
            DocStatus.PROCESSING,
            DocStatus.PROCESSED,
            DocStatus.FAILED,
        )

        tasks = [doc_processor.rag.get_docs_by_status(status) for status in statuses]
        results = await asyncio.gather(*tasks)

        response = DocsStatusesResponse()
        for idx, result in enumerate(results):
            status = statuses[idx]
            response.statuses[status] = [
                doc_status for doc_status in result.values()
            ]

        return response
    except Exception as e:
        logging.error(f"Error retrieving document statuses: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents")
async def clear_documents(_: None = Depends(verify_api_key)):
    """
    Clear all documents from the system
    
    Returns:
        InsertResponse containing operation status
        
    Raises:
        HTTPException: If document clearing fails
    """
    try:
        doc_processor.rag.text_chunks = []
        doc_processor.rag.entities_vdb = None
        doc_processor.rag.relationships_vdb = None
        return InsertResponse(
            status="success",
            message="All documents cleared successfully"
        )
    except Exception as e:
        logging.error(f"Error clearing documents: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 