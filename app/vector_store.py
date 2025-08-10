import os

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import config

from .embeddings import GeminiEmbeddings

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

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


def refresh_bm25_index():
    global bm25_index
    bm25_index = None
    if config.hybrid_search_enabled:
        _build_bm25_index()


vectorstore = None
bm25_index = None
bm25_documents = []


def init_vectorstore():
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore


def _build_bm25_index():
    global bm25_index, bm25_documents
    if not config.hybrid_search_enabled or BM25Okapi is None:
        return

    vs = init_vectorstore()
    docs = vs.get()
    if docs and docs.get("documents"):
        bm25_documents = docs["documents"]
        tokenized_docs = [doc.lower().split() for doc in bm25_documents]
        bm25_index = BM25Okapi(tokenized_docs)


def hybrid_similarity_search(
    query: str, k: int = 20, task_type: str = "RETRIEVAL_QUERY"
) -> list[tuple[Document, float]]:
    if not config.hybrid_search_enabled or BM25Okapi is None:
        vs = init_vectorstore()
        emb = get_embeddings()
        query_emb = emb.embed_query(query, task_type=task_type)
        return vs.similarity_search_by_vector_with_relevance_scores(query_emb, k=k)

    global bm25_index, bm25_documents
    if bm25_index is None:
        _build_bm25_index()

    vs = init_vectorstore()
    emb = get_embeddings()
    query_emb = emb.embed_query(query, task_type=task_type)

    semantic_results = vs.similarity_search_by_vector_with_relevance_scores(
        query_emb, k=config.bm25_top_k
    )

    if not bm25_index or not bm25_documents:
        return semantic_results[:k]

    query_tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(query_tokens)

    doc_scores = {}
    for doc, sem_score in semantic_results:
        doc_id = doc.metadata.get("chunk_id", id(doc))
        doc_scores[doc_id] = {"doc": doc, "semantic": sem_score, "bm25": 0.0}

    all_docs = vs.get()
    if all_docs and all_docs.get("metadatas"):
        for i, (bm25_score, metadata) in enumerate(
            zip(bm25_scores, all_docs["metadatas"], strict=False)
        ):
            if i < len(all_docs["documents"]):
                chunk_id = metadata.get("chunk_id") if metadata else f"doc_{i}"
                if chunk_id in doc_scores:
                    doc_scores[chunk_id]["bm25"] = float(bm25_score)

    combined_results = []
    for data in doc_scores.values():
        combined_score = (
            config.hybrid_weight_semantic * data["semantic"]
            + config.hybrid_weight_keyword * data["bm25"]
        )
        combined_results.append((data["doc"], combined_score))

    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results[:k]
