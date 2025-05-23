import json
import logging
import os
import uuid
from mem0.configs.vector_stores.qdrant import get_default_ollama_embedding_dims

# Set up the directory path
VECTOR_ID = str(uuid.uuid4())
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")
os.makedirs(mem0_dir, exist_ok=True)


def setup_config():
    config_path = os.path.join(mem0_dir, "config.json")
    if not os.path.exists(config_path):
        user_id = str(uuid.uuid4())
        config = {"user_id": user_id}
        with open(config_path, "w") as config_file:
            json.dump(config, config_file, indent=4)


def get_user_id():
    config_path = os.path.join(mem0_dir, "config.json")
    if not os.path.exists(config_path):
        return "anonymous_user"

    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            user_id = config.get("user_id")
            return user_id
    except Exception:
        return "anonymous_user"


def get_or_create_user_id(vector_store):
    """Store user_id in vector store and return it."""
    logging.error("get_or_create_user_id called")
    user_id = get_user_id()

    logging.error(f"get_or_create_user_id: Checking for user_id {user_id}")
    # Try to get existing user_id from vector store
    try:
        user_id_exists = vector_store.get_by_id(user_id)
        if user_id_exists:
            logging.error(f"get_or_create_user_id: user_id {user_id} found, returning.")
            return user_id
    except Exception:
        pass

    # If we get here, we need to insert the user_id
    logging.error("get_or_create_user_id: user_id not found, attempting to insert.")
    try:
        dims = getattr(vector_store, "embedding_model_dims", get_default_ollama_embedding_dims())
        logging.error(f"get_or_create_user_id: Calculated dims for insertion: {dims}")
        logging.error(f"get_or_create_user_id: Attempting vector_store.insert for user_id {user_id} with dims {dims}")
        vector_store.insert(
            vectors=[[0.1] * dims], payloads=[{"user_id": user_id, "type": "user_identity"}], ids=[user_id]
        )
    except Exception as e:
        logging.error(f"get_or_create_user_id: Exception during insert: {e}")

    return user_id
