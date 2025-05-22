import json
import logging
import os
import importlib.util
from typing import List, Optional

from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT

load_dotenv()

# Get LLM provider from environment variables
llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

# Initialize client based on LLM provider
openai_client = None
ollama_available = False

if llm_provider == "openai":
    try:
        from openai import OpenAI
        # Get API key and base URL from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        
        if not api_key:
            logging.warning("OPENAI_API_KEY not set. Categorization may not work properly.")
            api_key = "dummy-key"  # Fallback to dummy key
            
        # Initialize OpenAI client with explicit API key and base URL
        openai_client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
    except ImportError:
        logging.error("OpenAI package not installed. Categorization will be disabled.")
elif llm_provider == "ollama":
    # Check if ollama package is available
    ollama_spec = importlib.util.find_spec('ollama')
    if ollama_spec is not None:
        try:
            import ollama
            # Get the Ollama API base URL from environment
            ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://host.docker.internal:11434")
            
            # Set the OLLAMA_HOST environment variable for the client to use
            os.environ["OLLAMA_HOST"] = ollama_api_base
            logging.error(f"Set OLLAMA_HOST environment variable to: {ollama_api_base}")
            
            # Test connection to Ollama
            try:
                # Simple ping to check if Ollama is accessible
                client = ollama.Client(host=ollama_api_base)
                client.list()
                logging.info("Successfully connected to Ollama API")
                ollama_available = True
            except Exception as conn_err:
                logging.error(f"Failed to connect to Ollama at {ollama_api_base}: {conn_err}")
                ollama_available = False
        except ImportError:
            logging.error("Ollama package not properly installed. Categorization will be disabled.")
    else:
        logging.warning("Ollama package not found. Falling back to simple categorization.")
else:
    logging.warning(f"Unsupported LLM provider: {llm_provider}. Categorization will be disabled.")


class MemoryCategories(BaseModel):
    categories: List[str]


def get_simple_categories(memory: str) -> List[str]:
    """Fallback function to get basic categories without using LLMs."""
    # Simple keyword-based categorization
    categories = []
    
    # Define some basic category keywords
    category_keywords = {
        "personal": ["i", "me", "my", "mine", "family", "friend", "home"],
        "work": ["work", "job", "project", "task", "meeting", "deadline", "colleague"],
        "technical": ["code", "programming", "software", "hardware", "tech", "computer", "algorithm"],
        "health": ["health", "exercise", "workout", "diet", "doctor", "medical"],
        "finance": ["money", "finance", "bank", "payment", "cost", "price", "budget"],
    }
    
    # Convert memory to lowercase for case-insensitive matching
    memory_lower = memory.lower()
    
    # Check for keywords in memory
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in memory_lower:
                categories.append(category)
                break
    
    # If no categories found, add "uncategorized"
    if not categories:
        categories.append("uncategorized")
    
    return categories

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """Get categories for a memory."""
    # Check if any LLM client is available
    if openai_client is None and not ollama_available:
        logging.warning("No LLM client available. Using simple categorization.")
        return get_simple_categories(memory)
    
    try:
        if llm_provider == "openai" and openai_client is not None:
            try:
                response = openai_client.responses.parse(
                    model="gpt-4o-mini",
                    instructions=MEMORY_CATEGORIZATION_PROMPT,
                    input=memory,
                    temperature=0,
                    text_format=MemoryCategories,
                )
                response_json = json.loads(response.output[0].content[0].text)
                categories = response_json['categories']
                categories = [cat.strip().lower() for cat in categories]
                return categories
            except Exception as e:
                logging.error(f"Error using OpenAI for categorization: {e}")
                return get_simple_categories(memory)
        elif llm_provider == "ollama" and ollama_available:
            try:
                import ollama
                # Use Ollama for categorization
                ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
                
                # Format the prompt for Ollama
                prompt = f"{MEMORY_CATEGORIZATION_PROMPT}\n\nInput: {memory}\n\nOutput:"
                
                # Call Ollama API with explicit client instance
                client = ollama.Client(host=ollama_api_base)
                response = client.generate(
                    model=ollama_model,
                    prompt=prompt,
                    format="json"
                )
                
                # Parse the response
                try:
                    response_text = response['response']
                    response_json = json.loads(response_text)
                    categories = response_json.get('categories', [])
                    categories = [cat.strip().lower() for cat in categories]
                    return categories
                except (json.JSONDecodeError, KeyError) as e:
                    logging.error(f"Error parsing Ollama response: {e}")
                    return get_simple_categories(memory)
            except Exception as e:
                logging.error(f"Error calling Ollama API: {e}")
                return get_simple_categories(memory)
        else:
            # Fallback to simple categorization
            return get_simple_categories(memory)
    except Exception as e:
        logging.error(f"Error in categorization: {e}")
        # Fallback to simple categorization on error
        return get_simple_categories(memory)
