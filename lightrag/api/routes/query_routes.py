from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import logging
import traceback
import json
from typing import Optional
from ..models import QueryRequest, QueryResponse
from lightrag import LightRAG, QueryParam
from ..utils import trace_exception

router = APIRouter(tags=["query"])

# Global variables for dependency injection
rag: LightRAG = None
api_key_header = None

def init_routes(rag_instance: LightRAG, api_key: str = None):
    """
    Initialize query routes with required dependencies
    
    Args:
        rag_instance: LightRAG instance for query processing
        api_key: Optional API key for authentication
    """
    global rag, api_key_header
    rag = rag_instance
    if api_key:
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def query_request_to_params(request: QueryRequest) -> QueryParam:
    """
    Convert QueryRequest to QueryParam
    
    Args:
        request: QueryRequest object from API
        
    Returns:
        QueryParam object for RAG processing
    """
    param = QueryParam(mode=request.mode, stream=request.stream)
    
    # Copy optional parameters if they exist
    optional_params = [
        'only_need_context', 'only_need_prompt', 'response_type',
        'top_k', 'max_token_for_text_unit', 'max_token_for_global_context',
        'max_token_for_local_context', 'hl_keywords', 'll_keywords',
        'conversation_history', 'history_turns'
    ]
    
    for param_name in optional_params:
        value = getattr(request, param_name)
        if value is not None:
            setattr(param, param_name, value)
            
    return param

@router.post("/query")
async def query_text(
    request: QueryRequest,
    _: None = Depends(verify_api_key)
) -> QueryResponse:
    """
    Process a query request
    
    Args:
        request: QueryRequest containing query parameters
        
    Returns:
        QueryResponse containing query results
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        response = await rag.aquery(
            request.query,
            param=query_request_to_params(request)
        )

        # Handle different response types
        if isinstance(response, str):
            return QueryResponse(response=response)
        elif request.stream or hasattr(response, "__aiter__"):
            result = ""
            async for chunk in response:
                result += chunk
            return QueryResponse(response=result)
        elif isinstance(response, dict):
            return QueryResponse(response=json.dumps(response, indent=2))
        else:
            return QueryResponse(response=str(response))
            
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def query_text_stream(
    request: QueryRequest,
    _: None = Depends(verify_api_key)
):
    """
    Process a query request with streaming response
    
    Args:
        request: QueryRequest containing query parameters
        
    Returns:
        StreamingResponse for real-time results
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        params = query_request_to_params(request)
        params.stream = True
        response = await rag.aquery(request.query, param=params)

        async def stream_generator():
            """Generate streaming response chunks"""
            if isinstance(response, str):
                yield f"{json.dumps({'response': response})}\n"
            else:
                try:
                    async for chunk in response:
                        if chunk:
                            yield f"{json.dumps({'response': chunk})}\n"
                except Exception as e:
                    logging.error(f"Streaming error: {str(e)}")
                    yield f"{json.dumps({'error': str(e)})}\n"

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/x-ndjson",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=str(e)) 