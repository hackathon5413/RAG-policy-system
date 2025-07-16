import pypdf
import re
from typing import List, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str

# Insurance section patterns
SECTION_PATTERNS = [
    r'(?i)^[A-Z\s]+(?:Cover|Coverage|Benefit)',
    r'(?i)^(?:Section|Clause|Article)\s*[0-9]+',
    r'(?i)^We will (?:not )?cover',
    r'(?i)^(?:Exclusions?|Limitations?|Conditions?)',
    r'(?i)^(?:Claims?|Reimbursement)'
]

def extract_text_from_pdf(file_path: str) -> List[Tuple[str, int]]:
    """Extract text from PDF file page by page."""
    pages = []
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                pages.append((text, i + 1))
    return pages

def identify_sections(text: str) -> List[str]:
    """Split text into semantic sections based on insurance patterns."""
    lines = text.split('\n')
    sections = []
    current_section = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        is_section_header = any(re.match(pattern, line) for pattern in SECTION_PATTERNS)
        
        if is_section_header and current_section:
            sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    return sections

def classify_section(text: str) -> str:
    """Classify section type based on content."""
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in ["exclusion", "not cover", "will not"]):
        return "exclusion"
    elif any(keyword in text_lower for keyword in ["cover", "benefit", "we will cover"]):
        return "coverage"
    elif any(keyword in text_lower for keyword in ["claim", "reimbursement", "payable"]):
        return "claims"
    elif any(keyword in text_lower for keyword in ["condition", "requirement", "provided that"]):
        return "conditions"
    else:
        return "general"

def create_chunks_from_pages(pages: List[Tuple[str, int]], filename: str) -> List[DocumentChunk]:
    """Create document chunks from extracted pages."""
    chunks = []
    
    for page_text, page_num in pages:
        sections = identify_sections(page_text)
        
        for i, section in enumerate(sections):
            if len(section.strip()) < 50:
                continue
                
            chunk_id = f"{filename}_p{page_num}_s{i}"
            section_type = classify_section(section)
            
            metadata = {
                "filename": filename,
                "page": page_num,
                "section_index": i,
                "section_type": section_type,
                "word_count": len(section.split())
            }
            
            chunks.append(DocumentChunk(
                content=section,
                metadata=metadata,
                chunk_id=chunk_id
            ))
    
    return chunks

def process_pdf_document(file_path: str) -> List[DocumentChunk]:
    """Process a single PDF document and return chunks."""
    filename = Path(file_path).stem
    pages = extract_text_from_pdf(file_path)
    chunks = create_chunks_from_pages(pages, filename)
    return chunks

def process_multiple_pdfs(pdf_paths: List[str]) -> List[DocumentChunk]:
    """Process multiple PDF documents."""
    all_chunks = []
    for pdf_path in pdf_paths:
        chunks = process_pdf_document(pdf_path)
        all_chunks.extend(chunks)
    return all_chunks
