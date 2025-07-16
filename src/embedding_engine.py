from typing import List, Dict, Optional, Any
from pathlib import Path
import time
import json

from src.vector_store import VectorStore
from src.document_processor import process_multiple_pdfs, DocumentChunk
from config import ASSETS_DIR

# Global vector store instance (initialized once)
_vector_store = None

def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def close_vector_store():
    """Close the global vector store instance."""
    global _vector_store
    if _vector_store is not None:
        _vector_store.close()
        _vector_store = None

def process_all_documents(assets_path: Optional[str] = None) -> Dict[str, Any]:
    """Process all PDF documents in the assets directory."""
    if assets_path is None:
        assets_path = str(ASSETS_DIR)
    
    assets_path_obj = Path(assets_path)
    pdf_files = list(assets_path_obj.glob("*.pdf"))
    
    if not pdf_files:
        return {"error": "No PDF files found", "processed": 0}
    
    results = {
        "processed_files": [],
        "total_chunks": 0,
        "processing_time": 0,
        "errors": []
    }
    
    start_time = time.time()
    vector_store = get_vector_store()
    
    try:
        # Convert Path objects to strings for the function call
        pdf_paths = [str(pdf_file) for pdf_file in pdf_files]
        all_chunks = process_multiple_pdfs(pdf_paths)
        
        # Store all chunks at once
        vector_store.store_chunks(all_chunks)
        
        # Group chunks by filename for reporting
        chunks_by_file = {}
        for chunk in all_chunks:
            filename = chunk.metadata['filename']
            chunks_by_file[filename] = chunks_by_file.get(filename, 0) + 1
        
        for filename, count in chunks_by_file.items():
            results["processed_files"].append({
                "filename": filename,
                "chunks_created": count
            })
            print(f"✅ {filename}: {count} chunks created")
        
        results["total_chunks"] = len(all_chunks)
        
    except Exception as e:
        results["errors"].append(f"Error processing documents: {str(e)}")
        print(f"❌ Error: {str(e)}")
    
    finally:
        results["processing_time"] = time.time() - start_time
    
    return results

def search_documents(query: str, top_k: int = 10, section_type: Optional[str] = None) -> Dict[str, Any]:
    """Search documents using semantic similarity."""
    vector_store = get_vector_store()
    filters = {"section_type": section_type} if section_type else None
    return vector_store.search(query, top_k, filters)

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics."""
    vector_store = get_vector_store()
    return vector_store.get_stats()

def analyze_coverage_decision(query: str, relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze coverage decision based on relevant chunks."""
    query_lower = query.lower()
    
    coverage_chunks = [c for c in relevant_chunks if c['metadata']['section_type'] == 'coverage']
    exclusion_chunks = [c for c in relevant_chunks if c['metadata']['section_type'] == 'exclusion']
    
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

def process_insurance_query(query: str) -> Dict[str, Any]:
    """Process insurance query with decision logic."""
    search_results = search_documents(query, top_k=5)
    
    relevant_chunks = []
    for result in search_results['results']:
        if result['similarity'] > 0.3:  # Threshold for relevance
            relevant_chunks.append({
                'content': result['content'],
                'source': f"{result['metadata']['filename']} (Page {result['metadata']['page']})",
                'section_type': result['metadata']['section_type'],
                'similarity': result['similarity'],
                'metadata': result['metadata']
            })
    
    decision_logic = analyze_coverage_decision(query, relevant_chunks)
    
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
