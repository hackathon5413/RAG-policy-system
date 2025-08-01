import re
from typing import Dict, List, Any

def extract_and_validate_numbers(text: str, answer: str) -> str:
    """Extract and validate numerical data consistency"""
    
    text_numbers = set(re.findall(r'\d+(?:\.\d+)?', text))
    answer_numbers = set(re.findall(r'\d+(?:\.\d+)?', answer))
    
    missing_numbers = text_numbers - answer_numbers
    if missing_numbers and len(missing_numbers) <= 3:
        for num in missing_numbers:
            context_pattern = rf'.{{0,50}}{re.escape(num)}.{{0,50}}'
            context_match = re.search(context_pattern, text, re.IGNORECASE)
            if context_match and any(kw in context_match.group().lower() for kw in 
                ['percent', '%', 'rs', 'rupees', 'days', 'months', 'years', 'limit']):
                answer += f" {context_match.group().strip()}"
    
    return answer

def resolve_conflicts(sources: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    """Prioritize sources to resolve conflicts"""
    
    if 'waiting period' in question.lower():
        return sorted(sources, key=lambda x: ('specific' in x['content'].lower(), 
                                            'table' in x['content'].lower()), reverse=True)
    
    if any(term in question.lower() for term in ['limit', 'coverage', 'amount']):
        return sorted(sources, key=lambda x: ('rs' in x['content'].lower() or 
                                            'inr' in x['content'].lower() or
                                            '%' in x['content']), reverse=True)
    
    return sorted(sources, key=lambda x: x['similarity'], reverse=True)

def enhance_answer_completeness(question: str, answer: str, sources: List[Dict[str, Any]]) -> str:
    """Ensure answer addresses all question components"""
    
    question_lower = question.lower()
    
    if 'what' in question_lower and 'condition' in question_lower:
        conditions = []
        for source in sources:
            content = source['content'].lower()
            if any(cond in content for cond in ['provided', 'subject to', 'must', 'shall', 'required']):
                condition_match = re.search(r'(provided[^.]+|subject to[^.]+|must[^.]+|shall[^.]+)', 
                                          source['content'], re.IGNORECASE)
                if condition_match:
                    conditions.append(condition_match.group(1))
        
        if conditions and len(conditions) <= 2:
            answer += f" Conditions: {'. '.join(conditions)}"
    
    if re.search(r'(does|is|are).*(cover|covered)', question_lower):
        for source in sources:
            if any(term in source['content'].lower() for term in ['excluded', 'not covered', 'limitation']):
                exclusion_match = re.search(r'(excluded?[^.]+|not covered[^.]+|limitation[^.]+)', 
                                          source['content'], re.IGNORECASE)
                if exclusion_match and 'exclusion' not in answer.lower():
                    answer += f" Note: {exclusion_match.group(1)}"
                break
    
    return answer
