

import os
import numpy as np
from typing import List,Optional
from langchain.embeddings.base import Embeddings
from google import genai
from google.genai import types
from config import config
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddings(Embeddings):
    """Custom Gemini embeddings with task-specific optimization"""
    
    def __init__(self, model_name: Optional[str] = None, dimensions: Optional[int] = None):
        self.model_name = model_name or config.embedding_model
        self.dimensions = dimensions or config.embedding_dimensions
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector for accurate similarity comparison"""
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            return embedding
        return (embedding_np / norm).tolist()
    
    def _get_embedding(self, text: str, task_type: str) -> List[float]:
        """Get embedding for a single text with specific task type"""
        try:
            config_obj = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensions
            )
            
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=config_obj
            )
            
            # Extract embedding values from the response
            if result and result.embeddings:
                [embedding_obj] = result.embeddings
                embedding_values = embedding_obj.values
                
                # Ensure we have a list of floats
                if embedding_values is not None:
                    if isinstance(embedding_values, list):
                        embedding = embedding_values
                    elif hasattr(embedding_values, 'tolist'):
                        embedding = embedding_values.tolist()
                    else:
                        embedding = list(embedding_values)
                    
                    # Normalize if not using full 3072 dimensions
                    if self.dimensions != 3072:
                        embedding = self._normalize_embedding(embedding)
                        
                    return embedding
                else:
                    raise ValueError("Embedding values are None")
            else:
                raise ValueError("No embeddings returned from API")
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise ValueError(f"Error generating embedding: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using RETRIEVAL_DOCUMENT task type"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text, "RETRIEVAL_DOCUMENT")
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using RETRIEVAL_QUERY task type"""
        return self._get_embedding(text, "RETRIEVAL_QUERY")
