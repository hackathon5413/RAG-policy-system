import PyPDF2
import re
from typing import List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict
    chunk_id: str

class DocumentProcessor:
    def __init__(self):
        self.section_patterns = [
            r'(?i)^[A-Z\s]+(?:Cover|Coverage|Benefit)',
            r'(?i)^(?:Section|Clause|Article)\s*[0-9]+',
            r'(?i)^We will (?:not )?cover',
            r'(?i)^(?:Exclusions?|Limitations?|Conditions?)',
            r'(?i)^(?:Claims?|Reimbursement)'
        ]
    
    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        pages = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages.append((text, i + 1))
        return pages
    
    def identify_sections(self, text: str) -> List[str]:
        lines = text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_section_header = any(re.match(pattern, line) for pattern in self.section_patterns)
            
            if is_section_header and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def create_chunks(self, pages: List[Tuple[str, int]], filename: str) -> List[DocumentChunk]:
        chunks = []
        
        for page_text, page_num in pages:
            sections = self.identify_sections(page_text)
            
            for i, section in enumerate(sections):
                if len(section.strip()) < 50:
                    continue
                    
                chunk_id = f"{filename}_p{page_num}_s{i}"
                
                section_type = self._classify_section(section)
                
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
    
    def _classify_section(self, text: str) -> str:
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
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        filename = Path(file_path).stem
        pages = self.extract_text_from_pdf(file_path)
        chunks = self.create_chunks(pages, filename)
        return chunks
