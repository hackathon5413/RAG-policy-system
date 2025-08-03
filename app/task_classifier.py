
import logging
from .rag_core import call_gemini

logger = logging.getLogger(__name__)

def get_optimal_task_type(question: str) -> str:
  
    try:
     
        prompt = f"""Task type for this question:

QUESTION_ANSWERING: What/When/How questions, document lists, procedures
FACT_VERIFICATION: Can I/Am I/Is covered questions, eligibility scenarios  
RETRIEVAL_QUERY: General search, invalid questions

Q: "{question}"

Type:"""

        response = call_gemini(prompt).strip().upper()
        
        valid_types = ["QUESTION_ANSWERING", "FACT_VERIFICATION", "RETRIEVAL_QUERY"]
        
        for valid_type in valid_types:
            if valid_type in response or response in valid_type:
                logger.info(f"🎯 Classified as {valid_type}")
                return valid_type
        
        # If no match, use fallback
        logger.warning(f"⚠️ Invalid response '{response}', using fallback")
        return _fallback_classify(question)
            
    except Exception as e:
        logger.error(f"❌ Classification failed: {e}")
        return _fallback_classify(question)

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
