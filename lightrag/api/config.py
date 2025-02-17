import os
import argparse
from typing import Any
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv
import configparser
from ..utils import logger

# Load environment variables
try:
    load_dotenv(override=True)
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")

class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "NetworkXStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"

class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"
    mix = "mix"

def get_default_host(binding_type: str) -> str:
    """
    Get default host for different binding types
    
    Args:
        binding_type: Type of binding (ollama, lollms, azure_openai, openai)
        
    Returns:
        Default host URL for the specified binding type
    """
    default_hosts = {
        "ollama": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
        "lollms": os.getenv("LLM_BINDING_HOST", "http://localhost:9600"),
        "azure_openai": os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        "openai": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
    }
    return default_hosts.get(binding_type, os.getenv("LLM_BINDING_HOST", "http://localhost:11434"))

def get_env_value(env_key: str, default: Any, value_type: type = str) -> Any:
    """
    Get value from environment variables with type conversion
    
    Args:
        env_key: Environment variable key
        default: Default value if env var is not set
        value_type: Type to convert the value to
        
    Returns:
        Converted value from environment variable or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")
    try:
        return value_type(value)
    except ValueError:
        return default

def timeout_type(value):
    """
    Custom type for timeout parameter
    
    Args:
        value: Input value to convert
        
    Returns:
        None if value is "None", otherwise converted float
    """
    if value is None or value.lower() == "none":
        return None
    return float(value)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with separate working and input directories"
    )
    
    # Storage configuration
    parser.add_argument(
        "--kv-storage",
        default=get_env_value("LIGHTRAG_KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE),
        help=f"KV storage implementation (default: {DefaultRAGStorageConfig.KV_STORAGE})"
    )
    parser.add_argument(
        "--vector-storage",
        default=get_env_value("LIGHTRAG_VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE),
        help=f"Vector storage implementation (default: {DefaultRAGStorageConfig.VECTOR_STORAGE})"
    )
    parser.add_argument(
        "--graph-storage",
        default=get_env_value("LIGHTRAG_GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE),
        help=f"Graph storage implementation (default: {DefaultRAGStorageConfig.GRAPH_STORAGE})"
    )
    parser.add_argument(
        "--doc-status-storage",
        default=get_env_value("LIGHTRAG_DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE),
        help=f"Document status storage implementation (default: {DefaultRAGStorageConfig.DOC_STATUS_STORAGE})"
    )

    # Directory configuration
    parser.add_argument(
        "--working-dir",
        default=get_env_value("WORKING_DIR", "./workdir"),
        help="Working directory for storage (default: from env or ./workdir)"
    )
    parser.add_argument(
        "--input-dir",
        default=get_env_value("INPUT_DIR", "./input"),
        help="Input directory for documents (default: from env or ./input)"
    )

    # Server configuration
    parser.add_argument(
        "--host",
        default=get_env_value("HOST", "127.0.0.1"),
        help="Server host (default: from env or 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_value("PORT", 8000, int),
        help="Server port (default: from env or 8000)"
    )
    parser.add_argument(
        "--timeout",
        default=get_env_value("TIMEOUT", None, timeout_type),
        type=timeout_type,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout"
    )

    # LLM configuration
    parser.add_argument(
        "--llm-binding",
        default=get_env_value("LLM_BINDING", "ollama"),
        help="LLM binding type (default: from env or ollama)"
    )
    parser.add_argument(
        "--llm-binding-host",
        default=None,
        help="LLM binding host (default: auto-detected based on binding type)"
    )
    parser.add_argument(
        "--llm-model",
        default=get_env_value("LLM_MODEL", "llama2"),
        help="LLM model name (default: from env or llama2)"
    )
    parser.add_argument(
        "--llm-binding-api-key",
        default=get_env_value("LLM_BINDING_API_KEY", None),
        help="API key for LLM binding"
    )

    # Embedding configuration
    parser.add_argument(
        "--embedding-binding",
        default=get_env_value("EMBEDDING_BINDING", "ollama"),
        help="Embedding binding type (default: from env or ollama)"
    )
    parser.add_argument(
        "--embedding-binding-host",
        default=None,
        help="Embedding binding host (default: auto-detected based on binding type)"
    )
    parser.add_argument(
        "--embedding-model",
        default=get_env_value("EMBEDDING_MODEL", "llama2"),
        help="Embedding model name (default: from env or llama2)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=get_env_value("EMBEDDING_DIM", 1024, int),
        help="Embedding dimensions (default: from env or 1024)"
    )

    # RAG configuration
    parser.add_argument(
        "--max-async",
        type=int,
        default=get_env_value("MAX_ASYNC", 4, int),
        help="Maximum async operations (default: from env or 4)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_env_value("MAX_TOKENS", 32768, int),
        help="Maximum token size (default: from env or 32768)"
    )
    parser.add_argument(
        "--max-embed-tokens",
        type=int,
        default=get_env_value("MAX_EMBED_TOKENS", 8192, int),
        help="Maximum embedding token size (default: from env or 8192)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=get_env_value("CHUNK_SIZE", 512, int),
        help="Chunk size for text splitting (default: from env or 512)"
    )
    parser.add_argument(
        "--chunk-overlap-size",
        type=int,
        default=get_env_value("CHUNK_OVERLAP_SIZE", 50, int),
        help="Chunk overlap size (default: from env or 50)"
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=get_env_value("HISTORY_TURNS", 3, int),
        help="Number of conversation history turns (default: from env or 3)"
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=get_env_value("COSINE_THRESHOLD", 0.0, float),
        help="Cosine similarity threshold (default: from env or 0.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=get_env_value("TOP_K", 3, int),
        help="Top-K results to return (default: from env or 3)"
    )

    # Security configuration
    parser.add_argument(
        "--key",
        type=str,
        default=get_env_value("LIGHTRAG_API_KEY", None),
        help="API key for authentication"
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=get_env_value("SSL", False, bool),
        help="Enable HTTPS (default: from env or False)"
    )
    parser.add_argument(
        "--ssl-certfile",
        default=get_env_value("SSL_CERTFILE", None),
        help="Path to SSL certificate file (required if --ssl is enabled)"
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=get_env_value("SSL_KEYFILE", None),
        help="Path to SSL private key file (required if --ssl is enabled)"
    )

    # System configuration
    parser.add_argument(
        "--log-level",
        default=get_env_value("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from env or INFO)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=get_env_value("VERBOSE", False, bool),
        help="Enable verbose debug output"
    )
    parser.add_argument(
        "--auto-scan-at-startup",
        action="store_true",
        default=get_env_value("AUTO_SCAN_AT_STARTUP", False, bool),
        help="Enable automatic scanning when the program starts"
    )
    parser.add_argument(
        "--namespace-prefix",
        default=get_env_value("NAMESPACE_PREFIX", "default"),
        help="Namespace prefix for storage (default: from env or 'default')"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    args.working_dir = os.path.abspath(args.working_dir)
    args.input_dir = os.path.abspath(args.input_dir)
    
    return args

def get_cors_origins():
    """
    Get allowed CORS origins
    Returns ["*"] if not set
    
    Returns:
        List of allowed CORS origins
    """
    origins_str = os.getenv("CORS_ORIGINS", "*")
    if origins_str == "*":
        return ["*"]
    return [origin.strip() for origin in origins_str.split(",")] 