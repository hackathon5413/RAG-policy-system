
import logging
import json
from .rag_core import call_gemini
from config import config

logger = logging.getLogger(__name__)



def _fallback_classify(question: str) -> str:

    q = question.lower()
    
    # Verification patterns
    verification_words = ["can i", "am i eligible", "customer for", "been covered", 
                         "years", "is covered", "covered?", "does cover", 
                         "allowed?", "eligible?", "remaining", "balance"]
    
    if any(word in q for word in verification_words):
        logger.info("🔄 Fallback: FACT_VERIFICATION")
        return "FACT_VERIFICATION"
    
    # Question patterns  
    elif any(q.startswith(word) for word in ["what", "when", "how", "why", "where", "which"]):
        logger.info("🔄 Fallback: QUESTION_ANSWERING")
        return "QUESTION_ANSWERING"
    
    # Default
    else:
        logger.info("🔄 Fallback: RETRIEVAL_QUERY")
        return "RETRIEVAL_QUERY"

def get_task_and_queries(question: str) -> dict:
    try:
        expansion_count = getattr(config, 'query_expansion_count', 3)
        
        prompt = f"""Analyze this question and provide both task classification and query expansion:

QUESTION: "{question}"

Provide response as JSON:
{{
  "task_type": "QUESTION_ANSWERING|FACT_VERIFICATION|RETRIEVAL_QUERY",
  "expanded_questions": ["original question", "variant1", "variant2"]
}}

TASK TYPES:
- QUESTION_ANSWERING: What/When/How questions, document lists, procedures
- FACT_VERIFICATION: Can I/Am I/Is covered questions, eligibility scenarios
- RETRIEVAL_QUERY: General search, invalid questions

GENERATE {expansion_count} total questions (original + {expansion_count-1} variants) with:
- Domain-specific terminology versions
- Different aspects (what/why/how/when)
- Synonyms and related terms

JSON:"""
        
        response = call_gemini(prompt).strip()
        
        try:
            result = json.loads(response)
            task_type = result.get('task_type', '').upper()
            expanded_questions = result.get('expanded_questions', [question])
            
            valid_types = ["QUESTION_ANSWERING", "FACT_VERIFICATION", "RETRIEVAL_QUERY"]
            if task_type not in valid_types:
                task_type = _fallback_classify(question)
            
            if not expanded_questions or len(expanded_questions) == 0:
                expanded_questions = [question]
                
            logger.info(f"🎯 Combined: {task_type}, {len(expanded_questions)} queries")
            
            return {
                "task_type": task_type,
                "expanded_questions": expanded_questions
            }
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, using fallback")
            return {
                "task_type": _fallback_classify(question),
                "expanded_questions": [question]
            }
            
    except Exception as e:
        logger.error(f"Combined classification failed: {e}")
        return {
            "task_type": _fallback_classify(question),
            "expanded_questions": [question]
        }
