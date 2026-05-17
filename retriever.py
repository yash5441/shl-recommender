"""
retriever.py - Lightweight TF-IDF retriever for SHL catalog.
Uses scikit-learn only — fits comfortably in 512MB RAM.
"""

import json
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from catalog import load_catalog, build_search_document

JOB_LEVEL_ALIASES: Dict[str, str] = {
    "entry":       "Entry-Level",
    "junior":      "Entry-Level",
    "graduate":    "Graduate",
    "intern":      "Entry-Level",
    "mid":         "Mid-Professional",
    "mid-level":   "Mid-Professional",
    "senior":      "Professional Individual Contributor",
    "experienced": "Professional Individual Contributor",
    "manager":     "Manager",
    "lead":        "Front Line Manager",
    "supervisor":  "Supervisor",
    "director":    "Director",
    "executive":   "Executive",
    "vp":          "Executive",
    "c-suite":     "Executive",
    "ceo":         "Executive",
}


class CatalogRetriever:
    def __init__(self):
        self.assessments: List[Dict] = []
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            max_features=10000,
        )
        self._matrix = None
        self._loaded = False
        self._url_index: Dict[str, Dict] = {}
        self._name_index: Dict[str, Dict] = {}

    def load(self) -> bool:
        try:
            self.assessments = load_catalog()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

        if not self.assessments:
            print("WARNING: Catalog is empty.")
            return False

        print(f"Building TF-IDF index for {len(self.assessments)} assessments...")

        documents = [build_search_document(a) for a in self.assessments]
        self._matrix = self._vectorizer.fit_transform(documents)

        for a in self.assessments:
            self._url_index[a["url"]] = a
            self._name_index[a["name"].lower()] = a

        self._loaded = True
        print(f"TF-IDF index ready. {len(documents)} documents indexed.")
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_types: Optional[List[str]] = None,
        filter_job_level: Optional[str] = None,
        require_remote: bool = False,
        require_adaptive: bool = False,
    ) -> List[Dict]:
        if not self._loaded or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1]

        output = []
        for idx in top_indices:
            if len(output) >= top_k * 2:
                break
            a = self.assessments[idx]

            if require_remote and not a["remote"]:
                continue
            if require_adaptive and not a["adaptive"]:
                continue

            item = {**a, "relevance_score": round(float(scores[idx]), 4)}
            output.append(item)

        # Soft boost by type
        if filter_types:
            def sort_type(x):
                has_type = any(t in x["test_types"] for t in filter_types)
                return (0 if has_type else 1, -x["relevance_score"])
            output.sort(key=sort_type)

        # Soft boost by job level
        if filter_job_level:
            def sort_level(x):
                has_level = filter_job_level in x["job_levels"]
                return (0 if has_level else 1, -x["relevance_score"])
            output.sort(key=sort_level)

        return output[:top_k]

    def get_by_url(self, url: str) -> Optional[Dict]:
        return self._url_index.get(url)

    def get_by_name(self, name: str) -> Optional[Dict]:
        name_lower = name.lower().strip()
        if name_lower in self._name_index:
            return self._name_index[name_lower]
        for key, val in self._name_index.items():
            if name_lower in key or key in name_lower:
                return val
        return None

    def get_valid_urls(self) -> set:
        return set(self._url_index.keys())

    def is_loaded(self) -> bool:
        return self._loaded

    def count(self) -> int:
        return len(self.assessments)

    @staticmethod
    def normalize_job_level(text: str) -> Optional[str]:
        lower = text.lower()
        for alias, canonical in JOB_LEVEL_ALIASES.items():
            if alias in lower:
                return canonical
        return None


retriever = CatalogRetriever()