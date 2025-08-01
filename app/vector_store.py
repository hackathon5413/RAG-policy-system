from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .embeddings import EnhancedGeminiEmbeddings
from .advanced_preprocessing import SemanticChunkSplitter, AdvancedTextPreprocessor
from .advanced_retrieval import create_enhanced_retriever
from config import config
import os
import logging

logger = logging.getLogger(__name__)

# Choose chunking strategy based on configuration
if config.use_semantic_chunking:
    text_splitter = SemanticChunkSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    logger.info("Using semantic chunk splitter for better document structure")
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    logger.info("Using standard recursive chunk splitter")

os.makedirs(os.path.dirname(config.vector_db_path), exist_ok=True)

def get_embeddings():
    """Get enhanced embeddings instance"""
    return EnhancedGeminiEmbeddings()

def get_vectorstore():
    """Get vectorstore with enhanced embeddings"""
    return Chroma(
        persist_directory=config.vector_db_path,
        embedding_function=get_embeddings()
    )

vectorstore = None
retriever = None

def init_vectorstore():
    """Initialize vectorstore singleton"""
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
        logger.info(f"Initialized vectorstore at {config.vector_db_path}")
    return vectorstore

def get_enhanced_retriever():
    """Get enhanced retriever with advanced strategies"""
    global retriever
    if retriever is None:
        vectorstore_instance = init_vectorstore()
        if config.use_hybrid_retrieval:
            retriever = create_enhanced_retriever(
                vectorstore_instance, 
                retrieval_strategy=config.retrieval_strategy,
                top_k=config.top_k
            )
            logger.info(f"Initialized {config.retrieval_strategy} retriever")
        else:
            # Fallback to basic retrieval
            class BasicRetriever:
                def __init__(self, vs, top_k):
                    self.vectorstore = vs
                    self.top_k = top_k
                
                def retrieve(self, query):
                    return self.vectorstore.similarity_search_with_score(query, k=self.top_k)
            
            retriever = BasicRetriever(vectorstore_instance, config.top_k)
            logger.info("Initialized basic retriever")
    
    return retriever

# Enhanced search function for document processor
def enhanced_similarity_search_with_score(query: str, k: int = None):
    """Enhanced search function using advanced retrieval"""
    k = k or config.top_k
    
    if config.use_hybrid_retrieval:
        enhanced_retriever = get_enhanced_retriever()
        return enhanced_retriever.retrieve(query)
    else:
        vectorstore_instance = init_vectorstore()
        return vectorstore_instance.similarity_search_with_score(query, k=k)
