import os

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import config

from .embeddings import GeminiEmbeddings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    separators=["\n\n", "\n", ". ", " "],
)

os.makedirs(os.path.dirname(config.vector_db_path), exist_ok=True)


def get_embeddings():
    return GeminiEmbeddings()


def get_vectorstore():
    return Chroma(
        persist_directory=config.vector_db_path, embedding_function=get_embeddings()
    )


vectorstore = None


def init_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore


def semantic_similarity_search(
    query: str, k: int = 20, task_type: str = "RETRIEVAL_QUERY"
) -> list[tuple[Document, float]]:
    """Perform semantic similarity search using embeddings."""
    vs = init_vectorstore()
    emb = get_embeddings()
    query_emb = emb.embed_query(query, task_type=task_type)
    return vs.similarity_search_by_vector_with_relevance_scores(query_emb, k=k)
