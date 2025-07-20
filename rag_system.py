#!/usr/bin/env python3

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import requests
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


@dataclass
class Config:
    chunk_size: int = 600
    chunk_overlap: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "./data/chroma_db"
    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434/api/generate"
    top_k: int = 5


class InsuranceRAG:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        self._init_vectorstore()
    
    def _init_vectorstore(self):
        os.makedirs(os.path.dirname(self.config.vector_db_path), exist_ok=True)
        # Always use the standard Chroma constructor for both new and existing databases
        self.vectorstore = Chroma(
            persist_directory=self.config.vector_db_path,
            embedding_function=self.embeddings
        )
    
    def _classify_section(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ["exclusion", "not cover", "will not", "excluded"]):
            return "exclusion"
        elif any(word in text_lower for word in ["cover", "benefit", "we will pay", "covered"]):
            return "coverage"
        elif any(word in text_lower for word in ["claim", "procedure", "documentation"]):
            return "claims"
        return "general"
    
    def _clean_text(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            if any(skip in line.lower() for skip in ["bajaj allianz", "page ", "www.", "uin-"]):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            filename = Path(pdf_path).stem
            processed_docs = []
            
            for i, doc in enumerate(documents):
                cleaned_content = self._clean_text(doc.page_content)
                if len(cleaned_content) < 100:
                    continue
                
                doc.page_content = cleaned_content
                doc.metadata.update({
                    "filename": filename,
                    "page": i + 1,
                    "section_type": self._classify_section(cleaned_content)
                })
                processed_docs.append(doc)
            
            chunks = self.text_splitter.split_documents(processed_docs)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{filename}_chunk_{i}"
            
            if self.vectorstore is None:
                self._init_vectorstore()
            
            if self.vectorstore is not None:
                self.vectorstore.add_documents(chunks)
                self.vectorstore.persist()
            else:
                return {"success": False, "error": "Failed to initialize vector store"}
            
            return {
                "success": True,
                "filename": filename,
                "chunks_created": len(chunks),
                "pages_processed": len(processed_docs)
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        directory = Path(directory_path)
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            return {"success": False, "error": "No PDF files found"}
        
        results = {"processed_files": [], "total_chunks": 0, "errors": []}
        
        for pdf_file in pdf_files:
            result = self.ingest_pdf(str(pdf_file))
            if result["success"]:
                results["processed_files"].append(result)
                results["total_chunks"] += result["chunks_created"]
                print(f"✅ {result['filename']}: {result['chunks_created']} chunks")
            else:
                results["errors"].append(f"{pdf_file.name}: {result['error']}")
                print(f"❌ {pdf_file.name}: {result['error']}")
        
        return results
    
    def search(self, query: str, section_type: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            if self.vectorstore is None:
                return []
                
            filter_dict = {"section_type": section_type} if section_type else None
            results = self.vectorstore.similarity_search_with_score(
                query, k=self.config.top_k, filter=filter_dict
            )
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": 1 - score,
                    "source": f"{doc.metadata.get('filename', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})"
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _call_ollama(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9}
            }
            
            response = requests.post(self.config.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return response.json().get("response", "No response generated")
        
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, question: str) -> Dict[str, Any]:
        search_results = self.search(question)
        
        if not search_results:
            return {
                "question": question,
                "answer": "No relevant information found in the policies.",
                "sources": []
            }
        
        context = "\n\n".join([
            f"Source: {result['source']}\nContent: {result['content']}"
            for result in search_results[:3]
        ])
        
        prompt = f"""Based on the following insurance policy information, answer the user's question accurately and concisely.

Context from insurance policies:
{context}

Question: {question}

Instructions:
- Answer based only on the provided policy information
- If coverage is mentioned, also check for any exclusions
- Be specific about conditions, deductibles, or limitations
- If information is unclear or missing, state that clearly
- Provide a direct yes/no answer when possible

Answer:"""
        
        answer = self._call_ollama(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [result["source"] for result in search_results[:3]],
            "relevant_sections": search_results[:3]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            if self.vectorstore is None:
                return {
                    "total_chunks": 0,
                    "embedding_model": self.config.embedding_model,
                    "chunk_settings": {
                        "size": self.config.chunk_size,
                        "overlap": self.config.chunk_overlap
                    },
                    "status": "Vector store not initialized"
                }
                
            collection = self.vectorstore._collection
            total_docs = collection.count()
            
            return {
                "total_chunks": total_docs,
                "embedding_model": self.config.embedding_model,
                "chunk_settings": {
                    "size": self.config.chunk_size,
                    "overlap": self.config.chunk_overlap
                }
            }
        except Exception as e:
            return {"error": str(e)}
