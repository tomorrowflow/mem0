import logging
import os
from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, Field, model_validator


def get_default_ollama_embedding_dims() -> int:
    """Get default embedding dimensions from environment variable or fallback to 1536."""
    dim_env_var = os.getenv("OLLAMA_EMBEDDING_DIMENSION")
    logging.warning(f"get_default_ollama_embedding_dims: OLLAMA_EMBEDDING_DIMENSION env var value: {dim_env_var}")
    
    if dim_env_var is not None:
        try:
            logging.error(f"Using OLLAMA_EMBEDDING_DIMENSION ('{dim_env_var}') to create vector stores.")
            final_dim = int(dim_env_var)
            logging.warning(f"get_default_ollama_embedding_dims: Returning dimension value: {final_dim}")
            return final_dim
        except ValueError:
            # If the environment variable is set but not a valid integer, fallback to default
            logging.error(f"OLLAMA_EMBEDDING_DIMENSION ('{dim_env_var}') is set but not a valid integer. Falling back to default dimension 1536.")
            pass
    
    final_dim = 1536
    logging.warning(f"get_default_ollama_embedding_dims: Returning default dimension value: {final_dim}")
    return final_dim


class QdrantConfig(BaseModel):
    from qdrant_client import QdrantClient

    QdrantClient: ClassVar[type] = QdrantClient

    collection_name: str = Field("mem0", description="Name of the collection")
    embedding_model_dims: Optional[int] = Field(default_factory=get_default_ollama_embedding_dims, description="Dimensions of the embedding model. Defaults to OLLAMA_EMBEDDING_DIMENSION env var or 1536.")
    client: Optional[QdrantClient] = Field(None, description="Existing Qdrant client instance")
    host: Optional[str] = Field(None, description="Host address for Qdrant server")
    port: Optional[int] = Field(None, description="Port for Qdrant server")
    path: Optional[str] = Field("/tmp/qdrant", description="Path for local Qdrant database")
    url: Optional[str] = Field(None, description="Full URL for Qdrant server")
    api_key: Optional[str] = Field(None, description="API key for Qdrant server")
    on_disk: Optional[bool] = Field(False, description="Enables persistent storage")

    @model_validator(mode="before")
    @classmethod
    def check_host_port_or_path(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        host, port, path, url, api_key = (
            values.get("host"),
            values.get("port"),
            values.get("path"),
            values.get("url"),
            values.get("api_key"),
        )
        if not path and not (host and port) and not (url and api_key):
            raise ValueError("Either 'host' and 'port' or 'url' and 'api_key' or 'path' must be provided.")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = {
        "arbitrary_types_allowed": True,
    }
