import json
from pathlib import Path

CACHE_FILE = "./data/question_cache.json"


class QuestionCache:
    def __init__(self):
        self.cache_file = Path(CACHE_FILE)
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


question_cache = QuestionCache()
