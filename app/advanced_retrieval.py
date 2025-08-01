"""
Advanced retrieval strategies for improved accuracy
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
import logging
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retrieval combining semantic search with keyword matching"""
    
    def __init__(self, vectorstore, top_k: int = 15):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.insurance_domain_terms = self._load_domain_terms()
    
    def _load_domain_terms(self) -> Dict[str, List[str]]:
        """Load insurance domain-specific terms for better matching"""
        return {
            "coverage_terms": [
                "coverage", "covered", "benefits", "protection", "insured", 
                "includes", "encompasses", "extends", "applies", "eligible"
            ],
            "exclusion_terms": [
                "exclusion", "excluded", "not covered", "limitation", "restriction",
                "prohibited", "barred", "invalid", "void", "excepted"
            ],
            "claims_terms": [
                "claim", "procedure", "process", "settlement", "reimbursement",
                "documentation", "forms", "submission", "approval"
            ],
            "financial_terms": [
                "premium", "deductible", "copay", "limit", "amount", "sum",
                "cost", "fee", "charge", "payment"
            ],
            "condition_terms": [
                "condition", "requirement", "eligibility", "criteria", "terms",
                "prerequisites", "mandatory", "must", "shall"
            ],
            "time_terms": [
                "period", "duration", "waiting", "term", "validity", "expiry",
                "renewal", "effective", "before", "after"
            ]
        }
    
    def _extract_key_concepts(self, query: str) -> Dict[str, List[str]]:
        """Extract key concepts from query for targeted search"""
        query_lower = query.lower()
        found_concepts = defaultdict(list)
        
        for category, terms in self.insurance_domain_terms.items():
            for term in terms:
                if term in query_lower:
                    found_concepts[category].append(term)
        
        # Extract specific entities
        # Numbers (amounts, percentages, ages)
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        if numbers:
            found_concepts['numbers'] = numbers
        
        # Time periods
        time_patterns = [
            r'\d+\s*(?:years?|months?|days?|weeks?)',
            r'annually?|monthly|quarterly|yearly'
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                found_concepts['time_periods'].extend(matches)
        
        return dict(found_concepts)
    
    def _rerank_by_query_concepts(self, 
                                  results: List[Tuple[Document, float]], 
                                  query_concepts: Dict[str, List[str]]) -> List[Tuple[Document, float]]:
        """Rerank results based on concept matching"""
        
        reranked_results = []
        
        for doc, score in results:
            content_lower = doc.page_content.lower()
            metadata = doc.metadata
            
            # Calculate concept boost
            concept_boost = 0.0
            concept_matches = 0
            
            # Check for concept matches
            for category, terms in query_concepts.items():
                for term in terms:
                    if term in content_lower:
                        concept_matches += 1
                        # Different weights for different concept types
                        if category in ['coverage_terms', 'exclusion_terms']:
                            concept_boost += 0.1  # High importance
                        elif category in ['claims_terms', 'financial_terms']:
                            concept_boost += 0.08
                        elif category in ['numbers', 'time_periods']:
                            concept_boost += 0.05  # Exact matches are valuable
                        else:
                            concept_boost += 0.03
            
            # Section type matching bonus
            section_type = metadata.get('section_type', '').lower()
            query_lower = ' '.join([term for terms in query_concepts.values() for term in terms]).lower()
            
            if section_type and section_type in query_lower:
                concept_boost += 0.15
            
            # Adjust score (lower is better for similarity)
            adjusted_score = score - concept_boost
            
            reranked_results.append((doc, adjusted_score, concept_matches))
        
        # Sort by adjusted score, then by concept matches
        reranked_results.sort(key=lambda x: (x[1], -x[2]))
        
        return [(doc, score) for doc, score, _ in reranked_results]
    
    def _diversify_results(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Diversify results to include different sections and sources"""
        
        diversified = []
        seen_sections = set()
        seen_sources = set()
        section_counts = defaultdict(int)
        
        # First pass: prioritize diversity
        for doc, score in results:
            section_type = doc.metadata.get('section_type', 'unknown')
            source = doc.metadata.get('filename', 'unknown')
            
            # Include if it's a new section type or source, or if we haven't hit limits
            include = False
            
            if section_type not in seen_sections:
                include = True
                seen_sections.add(section_type)
            elif source not in seen_sources:
                include = True
                seen_sources.add(source)
            elif section_counts[section_type] < 2:  # Allow up to 2 per section
                include = True
            elif len(diversified) < self.top_k // 2:  # Always include best matches
                include = True
            
            if include:
                diversified.append((doc, score))
                section_counts[section_type] += 1
                
                if len(diversified) >= self.top_k:
                    break
        
        # Second pass: fill remaining slots with best remaining results
        if len(diversified) < self.top_k:
            remaining_slots = self.top_k - len(diversified)
            diversified_docs = {doc.metadata.get('chunk_id', id(doc)) for doc, _ in diversified}
            
            for doc, score in results:
                chunk_id = doc.metadata.get('chunk_id', id(doc))
                if chunk_id not in diversified_docs:
                    diversified.append((doc, score))
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
        
        return diversified
    
    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Advanced hybrid retrieval"""
        
        # Extract key concepts from query
        query_concepts = self._extract_key_concepts(query)
        logger.info(f"Extracted concepts: {dict(query_concepts)}")
        
        # Primary semantic search
        semantic_results = self.vectorstore.similarity_search_with_score(
            query, k=min(self.top_k * 2, 30)  # Get more results for reranking
        )
        
        # Rerank based on concept matching
        reranked_results = self._rerank_by_query_concepts(semantic_results, query_concepts)
        
        # Diversify results
        diversified_results = self._diversify_results(reranked_results)
        
        # Additional targeted searches for specific concepts
        additional_results = []
        
        # Search for specific numbers or amounts mentioned in query
        if 'numbers' in query_concepts:
            for number in query_concepts['numbers'][:2]:  # Top 2 numbers
                number_results = self.vectorstore.similarity_search_with_score(
                    f"amount {number} sum {number}", k=3
                )
                additional_results.extend(number_results)
        
        # Search for specific terms with high weight
        important_terms = []
        if 'coverage_terms' in query_concepts:
            important_terms.extend(query_concepts['coverage_terms'][:2])
        if 'exclusion_terms' in query_concepts:
            important_terms.extend(query_concepts['exclusion_terms'][:2])
        
        for term in important_terms:
            term_results = self.vectorstore.similarity_search_with_score(term, k=2)
            additional_results.extend(term_results)
        
        # Combine and deduplicate results
        all_results = list(diversified_results)
        seen_chunks = {doc.metadata.get('chunk_id', id(doc)) for doc, _ in diversified_results}
        
        for doc, score in additional_results:
            chunk_id = doc.metadata.get('chunk_id', id(doc))
            if chunk_id not in seen_chunks and len(all_results) < self.top_k:
                all_results.append((doc, score))
                seen_chunks.add(chunk_id)
        
        # Final sort and limit
        all_results.sort(key=lambda x: x[1])  # Sort by score
        final_results = all_results[:self.top_k]
        
        logger.info(f"Retrieved {len(final_results)} results with hybrid approach")
        
        return final_results


class ContextualRetriever:
    """Retriever that maintains context across multiple queries"""
    
    def __init__(self, vectorstore, top_k: int = 15):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.query_history = []
        self.max_history = 5
    
    def add_query_to_history(self, query: str, results: List[Document]):
        """Add query and its results to history for context"""
        self.query_history.append({
            'query': query,
            'results': results,
            'concepts': self._extract_concepts(query)
        })
        
        # Keep only recent history
        if len(self.query_history) > self.max_history:
            self.query_history.pop(0)
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
        return [word for word in words if word not in ['what', 'when', 'where', 'which', 'does']]
    
    def _get_contextual_expansion(self, current_query: str) -> str:
        """Expand current query with context from history"""
        if not self.query_history:
            return current_query
        
        current_concepts = set(self._extract_concepts(current_query))
        
        # Find related concepts from history
        related_concepts = set()
        for past_query in self.query_history[-3:]:  # Last 3 queries
            past_concepts = set(past_query['concepts'])
            # Add concepts that appear with current concepts
            if current_concepts & past_concepts:  # If there's overlap
                related_concepts.update(past_concepts)
        
        # Limit expansion
        if related_concepts:
            expansion_terms = list(related_concepts - current_concepts)[:3]
            if expansion_terms:
                expanded_query = f"{current_query} {' '.join(expansion_terms)}"
                return expanded_query
        
        return current_query
    
    def retrieve_with_context(self, query: str) -> List[Tuple[Document, float]]:
        """Retrieve with contextual awareness"""
        
        # Expand query with context
        expanded_query = self._get_contextual_expansion(query)
        
        if expanded_query != query:
            logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
        
        # Use hybrid retrieval
        hybrid_retriever = HybridRetriever(self.vectorstore, self.top_k)
        results = hybrid_retriever.retrieve(expanded_query)
        
        # Add to history
        result_docs = [doc for doc, _ in results]
        self.add_query_to_history(query, result_docs)
        
        return results


def create_enhanced_retriever(vectorstore, retrieval_strategy: str = "hybrid", top_k: int = 15):
    """Factory function to create enhanced retrievers"""
    
    if retrieval_strategy == "hybrid":
        return HybridRetriever(vectorstore, top_k)
    elif retrieval_strategy == "contextual":
        return ContextualRetriever(vectorstore, top_k)
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")
