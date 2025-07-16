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
        
        # Skip empty lines and common headers/footers
        if not line or len(line) < 10:
            continue
            
        # Skip common header/footer patterns
        if any(pattern in line.lower() for pattern in [
            'page ', 'bajaj allianz', 'insurance co', 'policy wordings',
            'uin-', 'airport road', 'yerawada', 'pune -', 'www.'
        ]):
            continue
            
        is_section_header = any(re.match(pattern, line) for pattern in SECTION_PATTERNS)
        
        if is_section_header and current_section:
            section_content = '\n'.join(current_section)
            if len(section_content.strip()) > 100:  # Only keep substantial sections
                sections.append(section_content)
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        section_content = '\n'.join(current_section)
        if len(section_content.strip()) > 100:  # Only keep substantial sections
            sections.append(section_content)
    
    return sections

def classify_section(text: str) -> str:
    """Classify section type based on content."""
    text_lower = text.lower()
    
    # Check for exclusions first (more specific)
    if any(keyword in text_lower for keyword in [
        "exclusion", "not cover", "will not", "shall not", "excluded", 
        "limitations", "restrictions", "not payable", "not applicable"
    ]):
        return "exclusion"
    
    # Check for coverage sections
    elif any(keyword in text_lower for keyword in [
        "cover", "benefit", "we will cover", "we will pay", "payable", 
        "covered", "eligible", "reimbursement", "compensation"
    ]):
        return "coverage"
    
    # Check for claims sections
    elif any(keyword in text_lower for keyword in [
        "claim", "claims procedure", "how to claim", "settlement", 
        "documentation", "submit", "filing"
    ]):
        return "claims"
    
    # Check for conditions sections
    elif any(keyword in text_lower for keyword in [
        "condition", "requirement", "provided that", "subject to", 
        "terms", "definitions", "waiting period", "deductible"
    ]):
        return "conditions"
    
    # Check for specific benefits (air ambulance, maternity, etc.)
    elif any(keyword in text_lower for keyword in [
        "air ambulance", "emergency transport", "maternity", "dental", 
        "optical", "wellness", "preventive", "vaccination"
    ]):
        return "coverage"  # Specific benefits are usually coverage
    
    else:
        return "general"

def create_chunks_from_pages(pages: List[Tuple[str, int]], filename: str) -> List[DocumentChunk]:
    """Create document chunks from extracted pages."""
    chunks = []
    
    for page_text, page_num in pages:
        sections = identify_sections(page_text)
        
        for i, section in enumerate(sections):
            # Skip very short sections
            if len(section.strip()) < 100:
                continue
                
            # Skip sections that are mostly headers/footers
            section_lower = section.lower()
            if any(pattern in section_lower for pattern in [
                'bajaj allianz general insurance',
                'policy wordings/page',
                'airport road, yerawada'
            ]):
                continue
                
            chunk_id = f"{filename}_p{page_num}_s{i}"
            section_type = classify_section(section)
            
            # Skip if section type is still 'general' (likely header/footer)
            if section_type == 'general' and len(section.strip()) < 200:
                continue
            
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
