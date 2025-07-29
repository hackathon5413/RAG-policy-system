
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
    

    chunk_size: int = 1200
    chunk_overlap: int = 300
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    vector_db_path: str = "./data/chroma_db"
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    top_k: int = 12  
    
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
            "top_k": self.top_k
        }

# Global settings instance
config = AppConfig()

# Backward compatibility - for existing rag_system.py
CONFIG = config.to_dict()
