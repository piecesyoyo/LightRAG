from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any
import logging
import traceback

router = APIRouter(tags=["graph"])

# Global variables for dependency injection
rag = None
api_key_header = None

def init_routes(rag_instance, api_key: str = None):
    """
    Initialize graph routes with required dependencies
    
    Args:
        rag_instance: LightRAG instance for graph operations
        api_key: Optional API key for authentication
    """
    global rag, api_key_header
    rag = rag_instance
    if api_key:
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

@router.get("/graph/label/list")
async def get_graph_labels(_: None = Depends(verify_api_key)):
    """
    Get all graph labels
    
    Returns:
        List of available graph labels
        
    Raises:
        HTTPException: If label retrieval fails
    """
    try:
        return await rag.get_graph_labels()
    except Exception as e:
        logging.error(f"Error retrieving graph labels: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graphs")
async def get_knowledge_graph(
    label: str,
    _: None = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get knowledge graph for specified label
    
    Args:
        label: Graph label to retrieve
        
    Returns:
        Dictionary containing graph data
        
    Raises:
        HTTPException: If graph retrieval fails
    """
    try:
        return await rag.get_knowledge_graph(nodel_label=label, max_depth=100)
    except Exception as e:
        logging.error(f"Error retrieving knowledge graph: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 