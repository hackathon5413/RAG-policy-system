
from typing import Optional
from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    """Unified application configuration"""
    
    # API Configuration
    app_name: str = "LLM-Powered Intelligent Query-Retrieval System"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Authentication
    bearer_token: str = "43e704a77310d35ab207cbb456481b2657cbf41a97bd1d2a3800e648acacb5c1"
    

    # Optimized RAG Configuration for Better Accuracy
    chunk_size: int = 800  # Reduced from 1800 for better semantic focus
    chunk_overlap: int = 200  # Increased from 400 for better context preservation
    embedding_model: str = "text-embedding-004"  # Latest model for better accuracy
    embedding_dimensions: int = 768  # Optimized dimensions for balance
    vector_db_path: str = "./data/chroma_db"
    gemini_model: str = "gemini-2.5-flash"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    top_k: int = 10  # Reduced from 15 for more focused results
    
    # Enhanced Retrieval Settings
    use_hybrid_retrieval: bool = True
    use_semantic_chunking: bool = True
    use_advanced_preprocessing: bool = True
    retrieval_strategy: str = "hybrid"  # "hybrid" or "contextual"
    
    # Quality Settings
    min_chunk_length: int = 150  # Minimum chunk size for quality
    max_chunk_length: int = 1200  # Maximum chunk size
    similarity_threshold: float = 0.3  # Minimum similarity for results
    diversity_threshold: float = 0.8  # Similarity threshold for diversity  
    
    # API Keys
    gemini_api_key: Optional[str] = None
    
    # File Processing
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    allowed_file_types: list = ["pdf", "docx", "doc"]
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"
        
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "vector_db_path": self.vector_db_path,
            "gemini_model": self.gemini_model,
            "gemini_url": self.gemini_url,
            "top_k": self.top_k,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "use_semantic_chunking": self.use_semantic_chunking,
            "use_advanced_preprocessing": self.use_advanced_preprocessing,
            "retrieval_strategy": self.retrieval_strategy,
            "min_chunk_length": self.min_chunk_length,
            "max_chunk_length": self.max_chunk_length,
            "similarity_threshold": self.similarity_threshold,
            "diversity_threshold": self.diversity_threshold
        }

# Global settings instance
config = AppConfig()

# Backward compatibility - for existing rag_system.py
CONFIG = config.to_dict()
