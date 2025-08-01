from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .embeddings import EnhancedGeminiEmbeddings
from config import CONFIG, config  # Import both for compatibility
import os
import logging

logger = logging.getLogger(__name__)

# Fallback to basic chunking if advanced components aren't available
try:
    from .advanced_preprocessing import SemanticChunkSplitter, AdvancedTextPreprocessor
    from .advanced_retrieval import create_enhanced_retriever
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Advanced features not available, using basic chunking")
    ADVANCED_FEATURES_AVAILABLE = False

# Choose chunking strategy based on configuration
use_semantic_chunking = getattr(config, 'use_semantic_chunking', False) or CONFIG.get('use_semantic_chunking', False)
chunk_size = getattr(config, 'chunk_size', CONFIG.get('chunk_size', 800))
chunk_overlap = getattr(config, 'chunk_overlap', CONFIG.get('chunk_overlap', 200))

if use_semantic_chunking and ADVANCED_FEATURES_AVAILABLE:
    text_splitter = SemanticChunkSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    logger.info("Using semantic chunk splitter for better document structure")
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    logger.info("Using standard recursive chunk splitter")

# Get vector database path
vector_db_path = getattr(config, 'vector_db_path', CONFIG.get('vector_db_path', './data/chroma_db'))
os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)

def get_embeddings():
    """Get enhanced embeddings instance"""
    return EnhancedGeminiEmbeddings()

def get_vectorstore():
    """Get vectorstore with enhanced embeddings"""
    return Chroma(
        persist_directory=vector_db_path,
        embedding_function=get_embeddings()
    )

vectorstore = None
retriever = None

def init_vectorstore():
    """Initialize vectorstore singleton"""
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
        logger.info(f"Initialized vectorstore at {vector_db_path}")
    return vectorstore

def get_enhanced_retriever():
    """Get enhanced retriever with advanced strategies"""
    global retriever
    if retriever is None:
        vectorstore_instance = init_vectorstore()
        use_hybrid_retrieval = getattr(config, 'use_hybrid_retrieval', CONFIG.get('use_hybrid_retrieval', False))
        retrieval_strategy = getattr(config, 'retrieval_strategy', CONFIG.get('retrieval_strategy', 'hybrid'))
        top_k = getattr(config, 'top_k', CONFIG.get('top_k', 10))
        
        if use_hybrid_retrieval and ADVANCED_FEATURES_AVAILABLE:
            retriever = create_enhanced_retriever(
                vectorstore_instance, 
                retrieval_strategy=retrieval_strategy,
                top_k=top_k
            )
            logger.info(f"Initialized {retrieval_strategy} retriever")
        else:
            # Fallback to basic retrieval
            class BasicRetriever:
                def __init__(self, vs, top_k):
                    self.vectorstore = vs
                    self.top_k = top_k
                
                def retrieve(self, query):
                    return self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            
            retriever = BasicRetriever(vectorstore_instance, top_k)
            logger.info("Initialized basic retriever")
    
    return retriever

# Enhanced search function for document processor
def enhanced_similarity_search_with_score(query: str, k: int = None):
    """Enhanced search function using advanced retrieval"""
    top_k = k or getattr(config, 'top_k', CONFIG.get('top_k', 10))
    use_hybrid_retrieval = getattr(config, 'use_hybrid_retrieval', CONFIG.get('use_hybrid_retrieval', False))
    
    if use_hybrid_retrieval and ADVANCED_FEATURES_AVAILABLE:
        enhanced_retriever = get_enhanced_retriever()
        return enhanced_retriever.retrieve(query)
    else:
        vectorstore_instance = init_vectorstore()
        return vectorstore_instance.similarity_search_with_score(query, k=top_k)
