import json
import logging
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


def _fallback_classify(question: str) -> TaskType:
    q = question.lower()

    # Code-related patterns
    code_words = [
        "function",
        "code",
        "script",
        "programming",
        "debug",
        "algorithm",
        "api",
    ]
    if any(word in q for word in code_words):
        logger.info("🔄 Fallback: CODE_RETRIEVAL_QUERY")
        return TaskType.CODE_RETRIEVAL_QUERY

    # Verification patterns
    verification_words = [
        "can i",
        "am i eligible",
        "customer for",
        "been covered",
        "years",
        "is covered",
        "covered?",
        "does cover",
        "allowed?",
        "eligible?",
        "remaining",
        "balance",
        "verify",
        "check if",
        "confirm",
    ]
    if any(word in q for word in verification_words):
        logger.info("🔄 Fallback: FACT_VERIFICATION")
        return TaskType.FACT_VERIFICATION

    # Classification patterns
    classification_words = ["classify", "category", "type of", "kind of", "classify as"]
    if any(word in q for word in classification_words):
        logger.info("🔄 Fallback: CLASSIFICATION")
        return TaskType.CLASSIFICATION

    # Clustering/grouping patterns
    clustering_words = ["group", "similar", "cluster", "organize", "categorize"]
    if any(word in q for word in clustering_words):
        logger.info("🔄 Fallback: CLUSTERING")
        return TaskType.CLUSTERING

    # Similarity patterns
    similarity_words = ["similar to", "like", "compare", "similarity", "related"]
    if any(word in q for word in similarity_words):
        logger.info("🔄 Fallback: SEMANTIC_SIMILARITY")
        return TaskType.SEMANTIC_SIMILARITY

    # Question patterns
    elif any(
        q.startswith(word) for word in ["what", "when", "how", "why", "where", "which"]
    ):
        logger.info("🔄 Fallback: QUESTION_ANSWERING")
        return TaskType.QUESTION_ANSWERING

    # Default
    else:
        logger.info("🔄 Fallback: RETRIEVAL_QUERY")
        return TaskType.RETRIEVAL_QUERY


def get_batch_task_classifications(questions: list[str]) -> list[dict]:
    try:
        template = jinja_env.get_template("task_classifier.j2")
        prompt = template.render(questions=questions, expansion_enabled=False)

        response = call_gemini(prompt).strip()

        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            result = json.loads(response)
            results = result.get("results", [])

            classifications = []
            valid_types = [task.value for task in TaskType]  # Use enum values

            for i, question in enumerate(questions):
                if i < len(results):
                    item = results[i]
                    task_type = item.get("task_type", "").upper()
                    if task_type in valid_types:
                        task_enum = TaskType(task_type)
                    else:
                        task_enum = _fallback_classify(question)
                else:
                    task_enum = _fallback_classify(question)

                classifications.append(
                    {
                        "question": question,
                        "task_type": task_enum.value,  # Return string value for compatibility
                    }
                )

            logger.info(f"🎯 Batch classified {len(classifications)} questions")
            return classifications

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse batch JSON: {e}. Using fallback.")
            return [
                {"question": q, "task_type": _fallback_classify(q).value}
                for q in questions
            ]

    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        return [
            {"question": q, "task_type": _fallback_classify(q).value} for q in questions
        ]
