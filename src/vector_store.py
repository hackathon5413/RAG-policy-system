import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sqlite3
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np


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
    
    def batch_embed_texts(self, texts: List[str]) -> np.ndarray:
        """ EMBEDDING GENERATION - Generate embeddings for multiple texts in batches."""
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            # THIS IS WHERE TEXT BECOMES VECTORS
            batch_embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    
    def store_chunks(self, chunks: List[Any]) -> None:
        """Store document chunks in both vector and metadata databases."""
        if not chunks:
            return
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.batch_embed_texts(texts)
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Convert numpy array to list for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        # Store metadata in SQLite
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
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        try:
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            
            # ChromaDB 1.0+ uses 'where' parameter for filtering
            # Convert our simple filters to ChromaDB format
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = {"$eq": value}
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results and results.get('documents') and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'] and len(results['metadatas'][0]) > i else {}
                    distance = results['distances'][0][i] if results.get('distances') and results['distances'] and len(results['distances'][0]) > i else 1.0
                    
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity': 1 - distance,
                        'distance': distance
                    })
            
            # Log query
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
            
        except Exception as e:
            print(f"Search error: {e}")
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored data."""
        try:
            count = self.collection.count()
            
            cursor = self.conn.execute('''
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT filename) as total_files
                FROM chunks
            ''')
            
            result = cursor.fetchone()
            total_chunks = result[0] if result else 0
            total_files = result[1] if result else 0
            
            # Get section distribution
            cursor = self.conn.execute('''
                SELECT section_type, COUNT(*) as count
                FROM chunks 
                GROUP BY section_type
            ''')
            
            section_stats = {}
            for row in cursor.fetchall():
                section_stats[row[0]] = row[1]
            
            return {
                'vector_count': count,
                'total_chunks': total_chunks,
                'total_files': total_files,
                'section_distribution': section_stats
            }
            
        except Exception as e:
            print(f"Stats error: {e}")
            return {
                'vector_count': 0,
                'total_chunks': 0,
                'total_files': 0,
                'section_distribution': {},
                'error': str(e)
            }
    
    def close(self):
        """Close database connections."""
        self.conn.close()
