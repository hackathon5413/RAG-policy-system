import json
import logging
from dataclasses import dataclass
from enum import Enum

from jinja2 import Environment, FileSystemLoader

from .rag_core import call_gemini

logger = logging.getLogger(__name__)

jinja_env = Environment(loader=FileSystemLoader("prompts"))


class TaskType(Enum):
    """All supported Google embedding task types"""

    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


class QueryTransformationType(Enum):
    """Query transformation techniques"""

    NONE = "none"
    HYDE = "hyde"
    MULTI_STEP = "multi_step"
    COMBINED = "combined"


@dataclass
class QueryClassificationResult:
    """Enhanced classification result with transformation"""

    original_question: str
    task_type: str
    transformation_type: str
    transformed_queries: list[str]
    sub_questions: list[str] | None = None


def get_batch_task_classifications(questions: list[str]) -> list[dict]:
    """
    Enhanced batch classification with unified query transformation

    Performs classification and multi-step transformation in a SINGLE API call
    """
    try:
        # Use unified template for all tasks
        template = jinja_env.get_template("task_classifier.j2")
        prompt = template.render(questions=questions)

        response = call_gemini(prompt).strip()

        # Clean response
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        result = json.loads(response)
        results = result.get("results", [])

        enhanced_results = []
        valid_types = [task.value for task in TaskType]

        for i, question in enumerate(questions):
            if i < len(results):
                item = results[i]
                task_type = item.get("task_type", "RETRIEVAL_QUERY").upper()
                sub_questions = item.get("sub_questions", [])

                # Validate task type
                if task_type not in valid_types:
                    task_type = "RETRIEVAL_QUERY"

                # Build all transformed queries
                all_queries = {question}  # Start with original

                # Add sub-questions if available
                if sub_questions:
                    for sub_q in sub_questions:
                        if sub_q and sub_q.strip():
                            all_queries.add(sub_q)

                # Convert to list with original first
                final_queries = [question] + [q for q in all_queries if q != question]

                enhanced_result = {
                    "question": question,
                    "task_type": task_type,
                    "transformed_queries": final_queries,
                }

                enhanced_results.append(enhanced_result)
            else:
                # Fallback for missing results
                raise Exception(f"Missing result for question {i + 1}: '{question}'")

        logger.info(
            f"🎯 Unified classification+transformation completed for {len(enhanced_results)} questions in SINGLE API call"
        )
        return enhanced_results

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse unified classification response: {e}")
    except Exception as e:
        logger.error(f"Unified classification failed: {e}")
        raise Exception(f"Unified task classification failed: {e}")
