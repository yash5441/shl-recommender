"""
retriever.py - Semantic retriever using Google's embedding model.
Embeddings are generated via API (no local model, minimal RAM).
Falls back to TF-IDF if embedding API is unavailable.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

from catalog import load_catalog, build_search_document

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "models/embedding-001"  # Google's best free embedding model

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

SYNONYMS = {
    "coding":        ["programming", "software", "developer", "development"],
    "programming":   ["coding", "software", "developer"],
    "cognitive":     ["ability", "aptitude", "reasoning", "numerical", "verbal"],
    "soft skills":   ["personality", "behaviour", "interpersonal", "behavioural"],
    "psychometric":  ["personality", "behaviour", "opq"],
    "numerical":     ["quantitative", "data", "arithmetic", "number"],
    "verbal":        ["language", "comprehension", "reading", "written"],
    "leadership":    ["management", "competencies", "managerial"],
    "sales":         ["customer", "persuasion", "commercial"],
    "communication": ["written", "verbal", "language", "email"],
    "aptitude":      ["ability", "cognitive", "reasoning"],
    "behavioural":   ["personality", "behaviour", "opq"],
    "emotional":     ["personality", "behaviour", "resilience"],
}


class CatalogRetriever:
    def __init__(self):
        self.assessments: List[Dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._documents: List[str] = []
        self._use_semantic = False

        # TF-IDF fallback
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            max_features=10000,
        )
        self._tfidf_matrix = None
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

        # Build search documents
        self._documents = [build_search_document(a) for a in self.assessments]

        # Build indexes
        for a in self.assessments:
            self._url_index[a["url"]] = a
            self._name_index[a["name"].lower()] = a

        # Always build TF-IDF as fallback
        print("Building TF-IDF fallback index...")
        self._tfidf_matrix = self._vectorizer.fit_transform(self._documents)
        print("TF-IDF index ready.")

        # Try to build semantic embeddings
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self._build_semantic_index()
        else:
            print("No API key — using TF-IDF only.")

        self._loaded = True
        return True

    def _build_semantic_index(self):
        """Build semantic embeddings using Google's embedding API."""
        print(f"Building semantic index for {len(self._documents)} assessments...")
        try:
            embeddings = []
            # Process in batches of 20 to avoid rate limits
            batch_size = 20
            for i in range(0, len(self._documents), batch_size):
                batch = self._documents[i:i + batch_size]
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                )
                
                embeddings.extend(result["embedding"])
                print(f"  Embedded {min(i + batch_size, len(self._documents))}/{len(self._documents)}")

            self._embeddings = np.array(embeddings)
            self._use_semantic = True
            print(f"Semantic index ready. Shape: {self._embeddings.shape}")

        except Exception as e:
            print(f"Semantic embedding failed: {e}. Falling back to TF-IDF.")
            self._use_semantic = False

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Search using Google embeddings — true semantic understanding."""
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
            )
            query_embedding = np.array(result["embedding"]).reshape(1, -1)

            # Cosine similarity
            scores = cosine_similarity(query_embedding, self._embeddings).flatten()
            top_indices = np.argsort(scores)[::-1]

            output = []
            for idx in top_indices[:top_k * 2]:
                a = self.assessments[idx]
                output.append({
                    **a,
                    "relevance_score": round(float(scores[idx]), 4)
                })
            return output

        except Exception as e:
            print(f"Semantic search failed: {e}. Falling back to TF-IDF.")
            return []

    def _tfidf_search(self, query: str, top_k: int) -> List[Dict]:
        """TF-IDF search with synonym expansion."""
        expanded = self._expand_query(query)
        query_vec = self._vectorizer.transform([expanded])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1]

        output = []
        for idx in top_indices[:top_k * 2]:
            a = self.assessments[idx]
            output.append({
                **a,
                "relevance_score": round(float(scores[idx]), 4)
            })
        return output

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_types: Optional[List[str]] = None,
        filter_job_level: Optional[str] = None,
        require_remote: bool = False,
        require_adaptive: bool = False,
    ) -> List[Dict]:
        if not self._loaded:
            return []

        # Use semantic search if available, else TF-IDF
        if self._use_semantic and self._embeddings is not None:
            results = self._semantic_search(query, top_k)
            if not results:
                results = self._tfidf_search(query, top_k)
        else:
            results = self._tfidf_search(query, top_k)

        # Apply hard filters
        filtered = []
        for item in results:
            if require_remote and not item["remote"]:
                continue
            if require_adaptive and not item["adaptive"]:
                continue
            filtered.append(item)

        # Soft boost by test type
        if filter_types:
            def sort_type(x):
                has_type = any(t in x["test_types"] for t in filter_types)
                return (0 if has_type else 1, -x["relevance_score"])
            filtered.sort(key=sort_type)

        # Soft boost by job level
        if filter_job_level:
            def sort_level(x):
                has_level = filter_job_level in x["job_levels"]
                return (0 if has_level else 1, -x["relevance_score"])
            filtered.sort(key=sort_level)

        return filtered[:top_k]

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

    def is_using_semantic(self) -> bool:
        return self._use_semantic

    @staticmethod
    def _expand_query(query: str) -> str:
        words = query.lower().split()
        expansions = []
        for word in words:
            if word in SYNONYMS:
                expansions.extend(SYNONYMS[word])
        return query + " " + " ".join(expansions) if expansions else query

    @staticmethod
    def normalize_job_level(text: str) -> Optional[str]:
        lower = text.lower()
        for alias, canonical in JOB_LEVEL_ALIASES.items():
            if alias in lower:
                return canonical
        return None


retriever = CatalogRetriever()