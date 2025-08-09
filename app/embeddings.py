import hashlib
import json
import logging
import os
import threading

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.embeddings.base import Embeddings

from config import config

load_dotenv()

logger = logging.getLogger(__name__)

QUERY_CACHE_FILE = "./data/query_cache.json"
os.makedirs(os.path.dirname(QUERY_CACHE_FILE), exist_ok=True)


def load_query_cache() -> dict:
    try:
        with open(QUERY_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_query_cache(cache: dict):
    with open(QUERY_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def get_query_hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str | None = None, dimensions: int | None = None):
        self.model_name = model_name or config.embedding_model
        self.dimensions = dimensions or config.embedding_dimensions

        self.api_keys = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 44)]
        self.api_keys = [
            key for key in self.api_keys if key and not key.startswith("YOUR_API_KEY")
        ]

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

    def _normalize_embedding(self, embedding: list[float]) -> list[float]:
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            return embedding
        return (embedding_np / norm).tolist()

    def _get_next_client(self):
        with self.client_lock:
            client = self.clients[self.current_client_index]
            key_num = self.current_client_index + 1
            self.current_client_index = (self.current_client_index + 1) % len(
                self.clients
            )
            return client, key_num

    def _get_embedding(self, text: str, task_type: str) -> list[float]:
        client, key_num = self._get_next_client()
        if task_type == "RETRIEVAL_DOCUMENT":
            logger.info(f"🔑 [DOCUMENT EMBEDDING] Using API key #{key_num}")
        else:
            logger.info(f"🔑 [QUERY EMBEDDING - {task_type}] Using API key #{key_num}")
        try:
            config_obj = types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self.dimensions
            )

            result = client.models.embed_content(
                model=self.model_name, contents=text, config=config_obj
            )

            if result and result.embeddings:
                [embedding_obj] = result.embeddings
                embedding_values = embedding_obj.values

                if embedding_values is not None:
                    if isinstance(embedding_values, list):
                        embedding = embedding_values
                    elif hasattr(embedding_values, "tolist"):
                        embedding = embedding_values.tolist()
                    else:
                        embedding = list(embedding_values)

                    if self.dimensions != 3072:
                        embedding = self._normalize_embedding(embedding)

                    return embedding
                else:
                    raise ValueError("Embedding values are None")
            else:
                raise ValueError("No embeddings returned from API")

        except Exception as e:
            raise ValueError(f"Error generating embedding: {e}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= 5:
            return [self._get_embedding(text, "RETRIEVAL_DOCUMENT") for text in texts]

        return self._embed_documents_batched(texts)

    def _embed_documents_batched(self, texts: list[str]) -> list[list[float]]:
        import concurrent.futures

        batch_size = max(1, len(texts) // len(self.clients))
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        def process_batch_with_client(batch_and_client):
            batch, client_index = batch_and_client
            client = self.clients[client_index]
            key_num = client_index + 1
            logger.info(
                f"🔑 [BATCH EMBEDDING] Processing batch {client_index + 1}/{len(self.clients)} using API key #{key_num}"
            )
            batch_embeddings = []

            for text in batch:
                try:
                    config_obj = types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dimensions,
                    )

                    result = client.models.embed_content(
                        model=self.model_name, contents=text, config=config_obj
                    )

                    if result and result.embeddings:
                        [embedding_obj] = result.embeddings
                        embedding_values = embedding_obj.values

                        if embedding_values is not None:
                            if isinstance(embedding_values, list):
                                embedding = embedding_values
                            elif hasattr(embedding_values, "tolist"):
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

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.clients)
        ) as executor:
            batch_client_pairs = [
                (batch, i % len(self.clients)) for i, batch in enumerate(batches)
            ]
            futures = [
                executor.submit(process_batch_with_client, pair)
                for pair in batch_client_pairs
            ]

            all_embeddings = []
            for future in futures:
                batch_results = future.result()
                all_embeddings.extend(batch_results)

            return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        from .task_classifier import get_task_and_queries

        query_hash = get_query_hash(text)

        with self.cache_lock:
            if query_hash in self.query_cache:
                return self.query_cache[query_hash]

        task_result = get_task_and_queries(text)
        optimal_task_type = task_result["task_type"]
        logger.info(f"🎯 Using {optimal_task_type} for: {text[:50]}...")

        embedding = self._get_embedding(text, optimal_task_type)

        with self.cache_lock:
            self.query_cache[query_hash] = embedding
            if len(self.query_cache) % 5 == 0:
                save_query_cache(self.query_cache)

        return embedding

    def save_cache(self):
        with self.cache_lock:
            save_query_cache(self.query_cache)
