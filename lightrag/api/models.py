from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from .config import SearchMode
from lightrag.base import DocStatus

class QueryRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.hybrid
    stream: Optional[bool] = None
    only_need_context: Optional[bool] = None
    only_need_prompt: Optional[bool] = None
    response_type: Optional[str] = None
    top_k: Optional[int] = None
    max_token_for_text_unit: Optional[int] = None
    max_token_for_global_context: Optional[int] = None
    max_token_for_local_context: Optional[int] = None
    hl_keywords: Optional[List[str]] = None
    ll_keywords: Optional[List[str]] = None
    conversation_history: Optional[List[dict[str, Any]]] = None
    history_turns: Optional[int] = None

class QueryResponse(BaseModel):
    response: str

class InsertTextRequest(BaseModel):
    text: str

class InsertResponse(BaseModel):
    status: str
    message: str

class DocStatusResponse(BaseModel):
    id: str
    content_summary: str
    content_length: int
    status: DocStatus
    created_at: str
    updated_at: str
    chunks_count: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

class DocsStatusesResponse(BaseModel):
    statuses: Dict[DocStatus, List[DocStatusResponse]] = {} 