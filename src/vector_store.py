import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

from config import VECTOR_DB_PATH, METADATA_DB_PATH, EMBEDDING_MODEL, BATCH_SIZE

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="insurance_policies",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        self.conn = sqlite3.connect(METADATA_DB_PATH, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                filename TEXT,
                page INTEGER,
                section_index INTEGER,
                section_type TEXT,
                word_count INTEGER,
                content TEXT,
                created_at TEXT
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                results TEXT,
                timestamp TEXT
            )
        ''')
        self.conn.commit()
    
    def batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
            embeddings.extend(batch_embeddings.tolist())
        return embeddings
    
    def store_chunks(self, chunks: List) -> None:
        if not chunks:
            return
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.batch_embed_texts(texts)
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        for chunk in chunks:
            self.conn.execute('''
                INSERT OR REPLACE INTO chunks 
                (chunk_id, filename, page, section_index, section_type, word_count, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk.chunk_id,
                chunk.metadata['filename'],
                chunk.metadata['page'],
                chunk.metadata['section_index'],
                chunk.metadata['section_type'],
                chunk.metadata['word_count'],
                chunk.content,
                datetime.now().isoformat()
            ))
        
        self.conn.commit()
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> Dict:
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        where_clause = filters if filters else {}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                'content': doc,
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i],
                'distance': results['distances'][0][i]
            })
        
        self.conn.execute('''
            INSERT INTO queries (query, results, timestamp)
            VALUES (?, ?, ?)
        ''', (query, json.dumps(formatted_results), datetime.now().isoformat()))
        self.conn.commit()
        
        return {
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results)
        }
    
    def get_stats(self) -> Dict:
        count = self.collection.count()
        
        cursor = self.conn.execute('''
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT filename) as total_files,
                section_type,
                COUNT(*) as section_count
            FROM chunks 
            GROUP BY section_type
        ''')
        
        section_stats = {}
        total_chunks = 0
        total_files = 0
        
        for row in cursor.fetchall():
            if row[2]:  # section_type
                section_stats[row[2]] = row[3]
            total_chunks = row[0]
            total_files = row[1]
        
        return {
            'vector_count': count,
            'total_chunks': total_chunks,
            'total_files': total_files,
            'section_distribution': section_stats
        }
    
    def close(self):
        self.conn.close()
