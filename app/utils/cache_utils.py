"""Cache management utilities."""

import json
import os
from pathlib import Path

URL_CACHE_FILE = "./data/url_cache.json"
QUESTION_CACHE_FILE = "./data/question_cache.json"
os.makedirs(os.path.dirname(URL_CACHE_FILE), exist_ok=True)


def load_url_cache() -> dict[str, bool]:
    """Load URL cache from file"""
    try:
        with open(URL_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_url_cache(cache: dict[str, bool]):
    """Save URL cache to file"""
    with open(URL_CACHE_FILE, "w") as f:
        json.dump(cache, f)


class QuestionCache:
    """Cache for storing question-answer pairs"""

    def __init__(self):
        self.cache_file = Path(QUESTION_CACHE_FILE)
        self.cache_file.parent.mkdir(exist_ok=True)
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        try:
            if self.cache_file.exists():
                return json.loads(self.cache_file.read_text())
            else:
                return {}
        except Exception:
            return {}

    def _save_cache(self, cache_data: dict | None = None):
        data = cache_data or self._cache
        self.cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def get(self, question: str) -> str | None:
        return self._cache.get(question)

    def set(self, question: str, answer: str):
        self._cache[question] = answer
        self._save_cache()


# Global instance for backward compatibility
question_cache = QuestionCache()
