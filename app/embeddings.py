import os
import numpy as np
import threading
import json
import hashlib
import logging
from typing import List, Optional, Dict, Any
from langchain.embeddings.base import Embeddings
from google import genai
from google.genai import types
from config import CONFIG, config  # Import both for compatibility
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

QUERY_CACHE_FILE = "./data/query_cache.json"
EMBEDDING_METADATA_FILE = "./data/embedding_metadata.json"
os.makedirs(os.path.dirname(QUERY_CACHE_FILE), exist_ok=True)

def load_query_cache() -> dict:
    try:
        with open(QUERY_CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_query_cache(cache: dict):
    with open(QUERY_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def load_embedding_metadata() -> dict:
    try:
        with open(EMBEDDING_METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"stats": {}, "quality_metrics": {}}

def save_embedding_metadata(metadata: dict):
    with open(EMBEDDING_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_query_hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()

class EnhancedGeminiEmbeddings(Embeddings):
    """Enhanced Gemini embeddings with quality improvements"""
    
    def __init__(self, model_name: Optional[str] = None, dimensions: Optional[int] = None):
        self.model_name = model_name or getattr(config, 'embedding_model', CONFIG.get('embedding_model', 'text-embedding-004'))
        self.dimensions = dimensions or getattr(config, 'embedding_dimensions', CONFIG.get('embedding_dimensions', 768))
        
        self.api_keys = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 44)]
        self.api_keys = [key for key in self.api_keys if key and not key.startswith("YOUR_API_KEY")]
        
        if not self.api_keys:
            fallback_key = os.getenv("GEMINI_API_KEY")
            if fallback_key:
                self.api_keys = [fallback_key]
            else:
                raise ValueError("No valid API keys found")
        
        self.clients = [genai.Client(api_key=key) for key in self.api_keys]
        self.current_client_index = 0
        self.client_lock = threading.Lock()
        
        self.query_cache = load_query_cache()
        self.cache_lock = threading.Lock()
        self.embedding_metadata = load_embedding_metadata()
        
        # Quality tracking
        self.embedding_stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "average_embedding_time": 0.0
        }
    
    def _preprocess_text_for_embedding(self, text: str, task_type: str) -> str:
        """Preprocess text for better embedding quality"""
        
        # For queries, enhance with context
        if task_type == "RETRIEVAL_QUERY":
            # Add query enhancement
            enhanced_text = self._enhance_query_text(text)
        else:
            # For documents, clean and structure
            enhanced_text = self._enhance_document_text(text)
        
        return enhanced_text
    
    def _enhance_query_text(self, query: str) -> str:
        """Enhance query text for better retrieval"""
        
        # Add insurance domain context
        insurance_keywords = [
            "insurance", "policy", "coverage", "claim", "benefit", 
            "premium", "deductible", "exclusion", "limitation"
        ]
        
        query_lower = query.lower()
        domain_context = []
        
        # Check for specific insurance contexts
        if any(word in query_lower for word in ["health", "medical", "hospital"]):
            domain_context.append("health insurance")
        if any(word in query_lower for word in ["life", "death", "beneficiary"]):
            domain_context.append("life insurance")
        if any(word in query_lower for word in ["vehicle", "car", "accident"]):
            domain_context.append("vehicle insurance")
        
        # Add semantic markers
        if any(word in query_lower for word in ["cover", "include", "benefit"]):
            domain_context.append("coverage inquiry")
        elif any(word in query_lower for word in ["exclude", "not cover", "limitation"]):
            domain_context.append("exclusion inquiry")
        elif any(word in query_lower for word in ["claim", "process", "procedure"]):
            domain_context.append("claims inquiry")
        
        if domain_context:
            enhanced_query = f"Insurance {' '.join(domain_context)}: {query}"
        else:
            enhanced_query = f"Insurance policy question: {query}"
        
        return enhanced_query
    
    def _enhance_document_text(self, text: str) -> str:
        """Enhance document text for better embedding"""
        
        # Don't over-enhance - keep original semantic meaning
        # Just add subtle context markers
        if len(text) > 500:
            # For longer texts, add section markers
            text_lower = text.lower()
            if any(word in text_lower for word in ["cover", "benefit", "include"]):
                return f"[COVERAGE] {text}"
            elif any(word in text_lower for word in ["exclude", "not cover", "limitation"]):
                return f"[EXCLUSION] {text}"
            elif any(word in text_lower for word in ["claim", "procedure", "process"]):
                return f"[CLAIMS] {text}"
            elif any(word in text_lower for word in ["condition", "term", "definition"]):
                return f"[CONDITIONS] {text}"
        
        return text
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding with quality checks"""
        embedding_np = np.array(embedding)
        
        # Check for zero embeddings
        if np.allclose(embedding_np, 0):
            logger.warning("Zero embedding detected - may indicate poor text quality")
            return embedding
        
        # L2 normalization
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            return embedding
        
        normalized = (embedding_np / norm).tolist()
        
        # Quality check: ensure reasonable distribution
        std_dev = np.std(normalized)
        if std_dev < 0.01:  # Very low standard deviation might indicate poor embedding
            logger.warning(f"Low embedding variance detected: {std_dev}")
        
        return normalized
    
    def _get_next_client(self):
        with self.client_lock:
            client = self.clients[self.current_client_index]
            key_num = self.current_client_index + 1
            self.current_client_index = (self.current_client_index + 1) % len(self.clients)
            return client, key_num
    
    def _get_embedding(self, text: str, task_type: str) -> List[float]:
        import time
        start_time = time.time()
        
        # Preprocess text
        enhanced_text = self._preprocess_text_for_embedding(text, task_type)
        
        client, key_num = self._get_next_client()
        if task_type == "RETRIEVAL_DOCUMENT":
            logger.info(f"🔑 [DOCUMENT EMBEDDING] Using API key #{key_num}")
        else:
            logger.info(f"🔑 [QUERY EMBEDDING] Using API key #{key_num}")
        
        try:
            config_obj = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimensions
            )
            
            result = client.models.embed_content(
                model=self.model_name,
                contents=enhanced_text,
                config=config_obj
            )
            
            if result and result.embeddings:
                [embedding_obj] = result.embeddings
                embedding_values = embedding_obj.values
                
                if embedding_values is not None:
                    if isinstance(embedding_values, list):
                        embedding = embedding_values
                    elif hasattr(embedding_values, 'tolist'):
                        embedding = embedding_values.tolist()
                    else:
                        embedding = list(embedding_values)
                    
                    # Always normalize for consistency
                    embedding = self._normalize_embedding(embedding)
                    
                    # Update stats
                    embedding_time = time.time() - start_time
                    self.embedding_stats["api_calls"] += 1
                    self.embedding_stats["total_embeddings"] += 1
                    self.embedding_stats["average_embedding_time"] = (
                        (self.embedding_stats["average_embedding_time"] * (self.embedding_stats["api_calls"] - 1) + embedding_time) 
                        / self.embedding_stats["api_calls"]
                    )
                    
                    return embedding
                else:
                    raise ValueError("Embedding values are None")
            else:
                raise ValueError("No embeddings returned from API")
            
        except Exception as e:
            raise ValueError(f"Error generating embedding: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if len(texts) <= 5:
            return [self._get_embedding(text, "RETRIEVAL_DOCUMENT") for text in texts]
        
        return self._embed_documents_batched(texts)
    
    def _embed_documents_batched(self, texts: List[str]) -> List[List[float]]:
        import concurrent.futures
        
        batch_size = max(1, len(texts) // len(self.clients))
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        def process_batch_with_client(batch_and_client):
            batch, client_index = batch_and_client
            client = self.clients[client_index]
            key_num = client_index + 1
            logger.info(f"🔑 [BATCH EMBEDDING] Processing batch {client_index + 1}/{len(self.clients)} using API key #{key_num}")
            batch_embeddings = []
            
            for text in batch:
                try:
                    config_obj = types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dimensions
                    )
                    
                    result = client.models.embed_content(
                        model=self.model_name,
                        contents=text,
                        config=config_obj
                    )
                    
                    if result and result.embeddings:
                        [embedding_obj] = result.embeddings
                        embedding_values = embedding_obj.values
                        
                        if embedding_values is not None:
                            if isinstance(embedding_values, list):
                                embedding = embedding_values
                            elif hasattr(embedding_values, 'tolist'):
                                embedding = embedding_values.tolist()
                            else:
                                embedding = list(embedding_values)
                            
                            if self.dimensions != 3072:
                                embedding = self._normalize_embedding(embedding)
                                
                            batch_embeddings.append(embedding)
                        else:
                            raise ValueError("Embedding values are None")
                    else:
                        raise ValueError("No embeddings returned from API")
                        
                except Exception as e:
                    raise ValueError(f"Error generating embedding: {e}")
            
            return batch_embeddings
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            batch_client_pairs = [(batch, i % len(self.clients)) for i, batch in enumerate(batches)]
            futures = [executor.submit(process_batch_with_client, pair) for pair in batch_client_pairs]
            
            all_embeddings = []
            for future in futures:
                batch_results = future.result()
                all_embeddings.extend(batch_results)
            
            return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Enhanced query embedding with caching and quality checks"""
        query_hash = get_query_hash(text)
        
        with self.cache_lock:
            if query_hash in self.query_cache:
                self.embedding_stats["cache_hits"] += 1
                self.embedding_stats["total_embeddings"] += 1
                logger.info(f"💾 Cache hit for query: {text[:50]}...")
                return self.query_cache[query_hash]
        
        embedding = self._get_embedding(text, "RETRIEVAL_QUERY")
        
        with self.cache_lock:
            self.query_cache[query_hash] = embedding
            if len(self.query_cache) % 10 == 0:  # Save cache every 10 new queries
                save_query_cache(self.query_cache)
        
        return embedding
    
    def get_embedding_quality_stats(self) -> Dict[str, Any]:
        """Get embedding quality statistics"""
        cache_hit_rate = (
            self.embedding_stats["cache_hits"] / max(1, self.embedding_stats["total_embeddings"])
        ) * 100
        
        return {
            "total_embeddings": self.embedding_stats["total_embeddings"],
            "api_calls": self.embedding_stats["api_calls"],
            "cache_hits": self.embedding_stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "average_embedding_time": f"{self.embedding_stats['average_embedding_time']:.3f}s",
            "available_api_keys": len(self.api_keys)
        }
    
    def save_cache(self):
        """Save cache and metadata"""
        with self.cache_lock:
            save_query_cache(self.query_cache)
            
        # Save quality stats
        self.embedding_metadata["stats"] = self.get_embedding_quality_stats()
        save_embedding_metadata(self.embedding_metadata)


# Backward compatibility
GeminiEmbeddings = EnhancedGeminiEmbeddings
