import os
from typing import Optional

from mem0 import Memory


memory_client = None


def get_memory_client(custom_instructions: Optional[str] = None):
    """
    Get or initialize the Mem0 client.

    Args:
        custom_instructions: Optional instructions for the memory project.

    Returns:
        Initialized Mem0 client instance.

    Raises:
        Exception: If required API keys are not set.
    """
    global memory_client

    if memory_client is not None:
        return memory_client

    # Initialize config outside try block to ensure it's always defined
    config = {}
    
    try:
        # Get LLM provider from environment variables
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # Base configuration for vector store
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "openmemory",
                    "host": "mem0_store",
                    "port": 6333,
                }
            }
        }
        
        # Add LLM configuration based on provider
        if llm_provider == "ollama":
            # Check if ollama package is available
            try:
                import importlib.util
                ollama_spec = importlib.util.find_spec('ollama')
                ollama_available = ollama_spec is not None
            except ImportError:
                ollama_available = False
            
            if not ollama_available:
                print("Warning: 'ollama' package is not installed. Falling back to OpenAI provider.")
                # Fall back to OpenAI
                if os.getenv("OPENAI_API_KEY"):
                    config["llm"] = {
                        "provider": "openai",
                        "config": {}
                    }
                    config["embedder"] = {
                        "provider": "openai",
                        "config": {}
                    }
                else:
                    # If no OpenAI API key, use a mock embedder
                    print("Warning: No OPENAI_API_KEY found. Using mock embedder.")
                    config["llm"] = {
                        "provider": "openai",
                        "config": {
                            "api_key": "sk-dummy-key-for-mock-embedder"
                        }
                    }
                    config["embedder"] = {
                        "provider": "mock",
                        "config": {}
                    }
            else:
                # Ollama is available, proceed with Ollama configuration
                ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://host.docker.internal:11434")
                config["llm"] = {
                    "provider": "ollama",
                    "config": {
                        "model": os.getenv("OLLAMA_MODEL", "llama2"),
                        "ollama_base_url": ollama_api_base
                    }
                }
                # Add embedding model configuration for Ollama
                config["embedder"] = {
                    "provider": "ollama",
                    "config": {
                        "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
                        "embedding_dims": int(os.getenv("OLLAMA_EMBEDDING_DIMENSION", "768")),
                        "ollama_base_url": ollama_api_base
                    }
                }
        elif llm_provider == "openai":
            # Using OpenAI client
            config["llm"] = {
                "provider": "openai",
                "config": {}
            }
            
            config["embedder"] = {
                "provider": "openai",
                "config": {}
            }
            
            # Add API base URL if specified
            if os.getenv("OPENAI_API_BASE"):
                config["llm"]["config"]["api_base"] = os.getenv("OPENAI_API_BASE")
                config["embedder"]["config"]["openai_base_url"] = os.getenv("OPENAI_API_BASE")
            
            # Add API key if specified
            if os.getenv("OPENAI_API_KEY"):
                config["llm"]["config"]["api_key"] = os.getenv("OPENAI_API_KEY")
                config["embedder"]["config"]["api_key"] = os.getenv("OPENAI_API_KEY")

        memory_client = Memory.from_config(config_dict=config)
    except Exception as e:
        print(f"Error initializing memory client: {str(e)}")
        print(f"Config used: {config}")
        raise Exception(f"Exception occurred while initializing memory client: {str(e)}")

    # Custom instructions are not currently supported in this version
    # Keeping the parameter for backward compatibility

    return memory_client


def get_default_user_id():
    return "default_user"
