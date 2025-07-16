from typing import List, Dict, Optional
from pathlib import Path
import time
import json

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from config import ASSETS_DIR

class EmbeddingEngine:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
    
    def process_all_documents(self, assets_path: str = None) -> Dict:
        assets_path = Path(assets_path) if assets_path else ASSETS_DIR
        pdf_files = list(assets_path.glob("*.pdf"))
        
        if not pdf_files:
            return {"error": "No PDF files found", "processed": 0}
        
        results = {
            "processed_files": [],
            "total_chunks": 0,
            "processing_time": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file.name}...")
                
                chunks = self.processor.process_document(str(pdf_file))
                self.vector_store.store_chunks(chunks)
                
                results["processed_files"].append({
                    "filename": pdf_file.name,
                    "chunks_created": len(chunks)
                })
                results["total_chunks"] += len(chunks)
                
                print(f"✓ {pdf_file.name}: {len(chunks)} chunks created")
                
            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"✗ {error_msg}")
        
        results["processing_time"] = time.time() - start_time
        return results
    
    def search_documents(self, query: str, top_k: int = 10, 
                        section_type: Optional[str] = None) -> Dict:
        filters = {"section_type": section_type} if section_type else None
        return self.vector_store.search(query, top_k, filters)
    
    def get_system_stats(self) -> Dict:
        return self.vector_store.get_stats()
    
    def close(self):
        self.vector_store.close()

class PolicyQueryEngine:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
    
    def process_insurance_query(self, query: str) -> Dict:
        search_results = self.embedding_engine.search_documents(query, top_k=5)
        
        relevant_chunks = []
        for result in search_results['results']:
            if result['similarity'] > 0.3:  # Threshold for relevance
                relevant_chunks.append({
                    'content': result['content'],
                    'source': f"{result['metadata']['filename']} (Page {result['metadata']['page']})",
                    'section_type': result['metadata']['section_type'],
                    'similarity': result['similarity']
                })
        
        decision_logic = self._analyze_coverage(query, relevant_chunks)
        
        return {
            'query': query,
            'decision': decision_logic['decision'],
            'confidence': decision_logic['confidence'],
            'amount': decision_logic.get('amount'),
            'justification': {
                'relevant_sections': relevant_chunks,
                'reasoning': decision_logic['reasoning'],
                'conditions_met': decision_logic['conditions_met']
            }
        }
    
    def _analyze_coverage(self, query: str, chunks: List[Dict]) -> Dict:
        query_lower = query.lower()
        
        coverage_chunks = [c for c in chunks if c['section_type'] == 'coverage']
        exclusion_chunks = [c for c in chunks if c['section_type'] == 'exclusion']
        
        decision = "needs_review"
        confidence = 0.5
        reasoning = "Analysis based on document similarity"
        conditions_met = []
        
        if coverage_chunks and not exclusion_chunks:
            decision = "likely_covered"
            confidence = 0.8
            reasoning = "Found relevant coverage sections, no exclusions identified"
            conditions_met = ["Coverage section found"]
        elif exclusion_chunks:
            decision = "likely_excluded"
            confidence = 0.7
            reasoning = "Found exclusion clauses that may apply"
            conditions_met = ["Exclusion conditions identified"]
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning,
            'conditions_met': conditions_met,
            'amount': None  # Would need more complex logic for amount calculation
        }
    
    def close(self):
        self.embedding_engine.close()
