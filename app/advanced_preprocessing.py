"""
Advanced text preprocessing and chunking strategies for better embeddings
"""

import re
import spacy
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class AdvancedTextPreprocessor:
    """Advanced text preprocessing for better embeddings"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_insurance_text(self, text: str) -> str:
        """Clean and normalize insurance document text"""
        
        # Remove page numbers and headers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove policy numbers and reference codes
        text = re.sub(r'Policy\s+No\.?\s*:?\s*[\w\d/-]+', 'Policy Number', text, flags=re.IGNORECASE)
        text = re.sub(r'UIN\s*:?\s*[\w\d/-]+', 'UIN Number', text, flags=re.IGNORECASE)
        
        # Normalize currency amounts for better matching
        text = re.sub(r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)', r'INR \1', text)
        text = re.sub(r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)', r'INR \1', text)
        
        # Normalize percentages
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
        
        # Clean up whitespace and formatting
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        
        # Remove bullet points and numbering for better semantic flow
        text = re.sub(r'^\s*[•·▪▫◦‣⁃]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[a-z]\)\s*', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key insurance terms for better context"""
        
        insurance_terms = {
            'coverage': ['coverage', 'covered', 'benefit', 'protection', 'insured'],
            'exclusion': ['exclusion', 'excluded', 'not covered', 'limitation', 'restriction'],
            'claim': ['claim', 'settlement', 'reimbursement', 'compensation'],
            'condition': ['condition', 'requirement', 'eligibility', 'prerequisite'],
            'amount': ['amount', 'limit', 'sum', 'premium', 'deductible'],
            'time': ['period', 'duration', 'waiting', 'term', 'validity']
        }
        
        found_terms = []
        text_lower = text.lower()
        
        for category, terms in insurance_terms.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append(f"{category}:{term}")
        
        return found_terms
    
    def enhance_with_context(self, text: str, metadata: Dict[str, Any]) -> str:
        """Enhance text with contextual information"""
        
        context_prefix = []
        
        # Add section type context
        if 'section_type' in metadata:
            context_prefix.append(f"[{metadata['section_type'].upper()}]")
        
        # Add document context
        if 'filename' in metadata:
            # Extract policy type from filename
            filename = metadata['filename'].lower()
            if 'health' in filename or 'medical' in filename:
                context_prefix.append("[HEALTH_INSURANCE]")
            elif 'life' in filename:
                context_prefix.append("[LIFE_INSURANCE]")
            elif 'vehicle' in filename or 'motor' in filename:
                context_prefix.append("[VEHICLE_INSURANCE]")
        
        # Add key terms context
        key_terms = self.extract_key_terms(text)
        if key_terms:
            context_prefix.extend(key_terms[:3])  # Top 3 key terms
        
        if context_prefix:
            enhanced_text = f"{' '.join(context_prefix)} {text}"
        else:
            enhanced_text = text
        
        return enhanced_text


class SemanticChunkSplitter:
    """Semantic-aware chunk splitting for better embeddings"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preprocessor = AdvancedTextPreprocessor()
        
        # Sentence-aware splitter
        self.sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            keep_separator=True
        )
    
    def split_by_semantic_sections(self, text: str) -> List[str]:
        """Split text by semantic sections"""
        
        # Insurance document section patterns
        section_patterns = [
            r'(?i)^(COVERAGE|BENEFITS?|INCLUSIONS?)',
            r'(?i)^(EXCLUSIONS?|LIMITATIONS?|RESTRICTIONS?)',
            r'(?i)^(CLAIMS?|PROCEDURES?|PROCESS)',
            r'(?i)^(CONDITIONS?|TERMS?|DEFINITIONS?)',
            r'(?i)^(PREMIUMS?|COSTS?|PAYMENTS?)',
        ]
        
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts a new section
            is_new_section = any(re.match(pattern, line) for pattern in section_patterns)
            
            if is_new_section and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def create_overlapping_chunks(self, sections: List[str]) -> List[str]:
        """Create overlapping chunks from sections"""
        
        chunks = []
        
        for i, section in enumerate(sections):
            # Process individual section
            section_chunks = self.sentence_splitter.split_text(section)
            
            for chunk in section_chunks:
                # Add context from adjacent sections
                enhanced_chunk = chunk
                
                # Add previous section context if available
                if i > 0 and len(sections[i-1]) < 200:
                    enhanced_chunk = f"Previous context: {sections[i-1][-100:]}...\n\n{enhanced_chunk}"
                
                # Add next section context if available
                if i < len(sections) - 1 and len(sections[i+1]) < 200:
                    enhanced_chunk = f"{enhanced_chunk}\n\nNext context: {sections[i+1][:100]}..."
                
                chunks.append(enhanced_chunk)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using semantic awareness"""
        
        all_chunks = []
        
        for doc in documents:
            # Clean and preprocess text
            cleaned_text = self.preprocessor.clean_insurance_text(doc.page_content)
            
            # Split by semantic sections
            sections = self.split_by_semantic_sections(cleaned_text)
            
            # Create overlapping chunks
            chunks = self.create_overlapping_chunks(sections)
            
            # Create chunk documents
            for i, chunk_text in enumerate(chunks):
                # Enhance with context
                enhanced_text = self.preprocessor.enhance_with_context(chunk_text, doc.metadata)
                
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    'chunk_id': f"{doc.metadata.get('filename', 'doc')}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_length': len(doc.page_content),
                    'chunk_length': len(enhanced_text)
                })
                
                chunk_doc = Document(
                    page_content=enhanced_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(documents)} documents")
        return all_chunks


# Optimized configuration for better embeddings
OPTIMIZED_CHUNKING_CONFIG = {
    'chunk_size': 800,  # Smaller chunks for focused semantics
    'chunk_overlap': 200,  # Significant overlap for context preservation
    'use_semantic_splitting': True,
    'enhance_with_context': True,
    'clean_insurance_text': True
}
