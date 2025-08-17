import os

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

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
cross_encoder = None


def get_cross_encoder():
    """Initialize cross-encoder model for re-ranking"""
    global cross_encoder
    if cross_encoder is None:
        cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    return cross_encoder


def rerank_chunks(
    query: str, chunks: list[tuple[Document, float]], top_k: int | None = None
) -> list[tuple[Document, float]]:
    """Re-rank chunks using cross-encoder for better relevance"""
    if not chunks or len(chunks) <= 3:
        return chunks[:top_k] if top_k else chunks

    try:
        reranker = get_cross_encoder()

        # Prepare query-document pairs
        pairs = [
            (query, doc.page_content[:512]) for doc, _ in chunks
        ]  # Limit content length

        # Get cross-encoder scores
        cross_scores = reranker.predict(pairs)

        # Combine with original chunks
        reranked_chunks = [
            (doc, float(cross_score))
            for (doc, _), cross_score in zip(chunks, cross_scores, strict=True)
        ]

        # Sort by cross-encoder scores (higher = more relevant)
        reranked_chunks.sort(key=lambda x: x[1], reverse=True)

        return reranked_chunks[:top_k] if top_k else reranked_chunks

    except Exception as e:
        # Fallback to original chunks if re-ranking fails
        print(f"Re-ranking failed: {e}, using original order")
        return chunks[:top_k] if top_k else chunks


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
