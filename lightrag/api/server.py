from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
from typing import Set
import os

from .config import parse_args, get_cors_origins
from .document_manager import DocumentManager, DocumentProcessor
from .routes import document_router, query_router, graph_router
from .utils import display_splash_screen, set_verbose_debug
from .ollama_api import OllamaAPI
from lightrag import LightRAG
from lightrag.storage import (
    PostgreSQLDB, OracleDB, TiDB,
    PGKVStorage, PGVectorStorage, PGGraphStorage, PGDocStatusStorage,
    OracleKVStorage, OracleVectorDBStorage, OracleGraphStorage,
    TiDBKVStorage, TiDBVectorDBStorage, TiDBGraphStorage
)

class LightRAGServer:
    """LightRAG Server implementation"""

    def __init__(self, args):
        """
        Initialize LightRAG Server
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.app = None
        self.rag = None
        self.doc_manager = None
        self.doc_processor = None
        self.background_tasks: Set = set()

    async def setup_database_connections(self):
        """Setup database connections based on storage configuration"""
        storage_instances = []
        
        # Initialize databases if needed
        if self._needs_postgres():
            self.postgres_db = await self._init_postgres()
            
        if self._needs_oracle():
            self.oracle_db = await self._init_oracle()
            
        if self._needs_tidb():
            self.tidb_db = await self._init_tidb()

    async def cleanup_database_connections(self):
        """Cleanup database connections"""
        for db in [self.postgres_db, self.oracle_db, self.tidb_db]:
            if db and hasattr(db, "pool"):
                await db.pool.close()
                logging.info(f"Closed {db.__class__.__name__} connection pool")

    def create_app(self) -> FastAPI:
        """
        Create and configure FastAPI application
        
        Returns:
            Configured FastAPI application
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Setup
            await self.setup_database_connections()
            
            # Start background scanning if enabled
            if self.args.auto_scan_at_startup:
                self._start_background_scan()
                
            yield
            
            # Cleanup
            await self.cleanup_database_connections()

        # Initialize FastAPI
        self.app = FastAPI(
            title="LightRAG API",
            description="API for querying text using LightRAG",
            version="1.0.0",
            lifespan=lifespan
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=get_cors_origins(),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize components
        self._init_rag()
        self._init_document_handlers()
        self._init_routes()
        self._mount_static_files()

        return self.app

    def _init_rag(self):
        """Initialize LightRAG instance"""
        # RAG initialization code here
        pass

    def _init_document_handlers(self):
        """Initialize document management components"""
        self.doc_manager = DocumentManager(self.args.input_dir)
        self.doc_processor = DocumentProcessor(self.rag)

    def _init_routes(self):
        """Initialize API routes"""
        # Initialize route modules with dependencies
        document_router.init_routes(self.doc_manager, self.doc_processor, self.args.key)
        query_router.init_routes(self.rag, self.args.key)
        graph_router.init_routes(self.rag, self.args.key)

        # Include routers
        self.app.include_router(document_router)
        self.app.include_router(query_router)
        self.app.include_router(graph_router)

        # Add Ollama API routes
        ollama_api = OllamaAPI(self.rag, top_k=self.args.top_k)
        self.app.include_router(ollama_api.router, prefix="/api")

    def _mount_static_files(self):
        """Mount static files for web UI"""
        static_dir = Path(__file__).parent / "webui"
        static_dir.mkdir(exist_ok=True)
        self.app.mount("/webui", StaticFiles(directory=static_dir, html=True), name="webui")

def create_app(args):
    """
    Create LightRAG server application
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured FastAPI application
    """
    server = LightRAGServer(args)
    return server.create_app()

def main():
    """Main entry point for LightRAG server"""
    args = parse_args()
    set_verbose_debug(args.verbose)
    
    app = create_app(args)
    display_splash_screen(args)
    
    import uvicorn
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
    }
    
    if args.ssl:
        uvicorn_config.update({
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
        })
        
    uvicorn.run(**uvicorn_config)

if __name__ == "__main__":
    main() 