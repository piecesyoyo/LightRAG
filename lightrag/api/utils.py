import sys
import logging
import traceback
from typing import Optional
from dataclasses import dataclass
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()

class ASCIIColors:
    """Utility class for colored console output"""
    
    @staticmethod
    def green(text: str):
        """Print text in green"""
        print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")

    @staticmethod
    def yellow(text: str):
        """Print text in yellow"""
        print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

    @staticmethod
    def red(text: str):
        """Print text in red"""
        print(f"{Fore.RED}{text}{Style.RESET_ALL}")

    @staticmethod
    def white(text: str):
        """Print text in white"""
        print(f"{Fore.WHITE}{text}{Style.RESET_ALL}")

    @staticmethod
    def info(text: str):
        """Print info message in blue"""
        print(f"{Fore.BLUE}[INFO] {text}{Style.RESET_ALL}")

def trace_exception(e: Exception):
    """
    Log exception with full traceback
    
    Args:
        e: Exception to log
    """
    logging.error(f"Error: {str(e)}")
    logging.error(traceback.format_exc())

def display_splash_screen(args):
    """
    Display server startup information
    
    Args:
        args: Parsed command line arguments
    """
    ASCIIColors.green("\nLightRAG Server Configuration:")
    ASCIIColors.white(f"""
    Working Directory: {args.working_dir}
    Input Directory:  {args.input_dir}
    Host:            {args.host}
    Port:            {args.port}
    
    Storage Configuration:
    - KV Storage:          {args.kv_storage}
    - Vector Storage:      {args.vector_storage}
    - Graph Storage:       {args.graph_storage}
    - Doc Status Storage:  {args.doc_status_storage}
    
    Model Configuration:
    - LLM Binding:         {args.llm_binding}
    - LLM Model:           {args.llm_model}
    - Embedding Binding:   {args.embedding_binding}
    - Embedding Model:     {args.embedding_model}
    
    Performance Settings:
    - Max Async:           {args.max_async}
    - Max Tokens:          {args.max_tokens}
    - Embedding Dim:       {args.embedding_dim}
    - Max Embed Tokens:    {args.max_embed_tokens}
    """)

    if args.ssl:
        ASCIIColors.yellow("\n🔒 SSL/HTTPS is enabled")
        ASCIIColors.white(f"""    Certificate: {args.ssl_certfile}
    Key:         {args.ssl_keyfile}
    """)

    if args.key:
        ASCIIColors.yellow("\n⚠️  Security Notice:")
        ASCIIColors.white("""    API Key authentication is enabled.
    Make sure to include the X-API-Key header in all your requests.
    """)

    ASCIIColors.green("Server is ready to accept connections! 🚀\n")
    sys.stdout.flush()

@dataclass
class DatabaseConfig:
    """Database configuration container"""
    host: str
    port: int
    user: str
    password: str
    database: str
    workspace: str = "default"
    config_dir: Optional[str] = None
    wallet_location: Optional[str] = None
    wallet_password: Optional[str] = None

def set_verbose_debug(enabled: bool):
    """
    Set verbose debug logging
    
    Args:
        enabled: Whether to enable verbose debug logging
    """
    if enabled:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO) 