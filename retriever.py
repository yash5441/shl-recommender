"""
retriever.py - ChromaDB in-memory vector store for semantic search over the SHL catalog.
Loaded once at startup; all queries served from memory.
"""

import json
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

from catalog import load_catalog, build_search_document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "shl_catalog"

# Job-level normalization: maps user terms -> catalog job level strings
JOB_LEVEL_ALIASES: Dict[str, str] = {
    "entry":        "Entry-Level",
    "junior":       "Entry-Level",
    "graduate":     "Graduate",
    "intern":       "Entry-Level",
    "mid":          "Mid-Professional",
    "mid-level":    "Mid-Professional",
    "senior":       "Professional Individual Contributor",
    "experienced":  "Professional Individual Contributor",
    "manager":      "Manager",
    "lead":         "Front Line Manager",
    "supervisor":   "Supervisor",
    "director":     "Director",
    "executive":    "Executive",
    "vp":           "Executive",
    "c-suite":      "Executive",
    "ceo":          "Executive",
}


class CatalogRetriever:
    def __init__(self):
        self.assessments: List[Dict] = []
        self._client = chromadb.Client()
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = None
        self._loaded = False
        # Fast lookup by URL and by name
        self._url_index: Dict[str, Dict] = {}
        self._name_index: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load catalog from JSON and build ChromaDB index. Returns True on success."""
        try:
            self.assessments = load_catalog()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return False

        if not self.assessments:
            print("WARNING: Catalog is empty.")
            return False

        print(f"Building ChromaDB index for {len(self.assessments)} assessments...")

        # Reset collection
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        documents, ids, metadatas = [], [], []

        for i, a in enumerate(self.assessments):
            doc_text = build_search_document(a)
            documents.append(doc_text)
            ids.append(f"a_{i}")
            metadatas.append({
                "name":             a["name"],
                "url":              a["url"],
                "test_type":        a["test_type"],
                "test_types":       json.dumps(a["test_types"]),
                "test_type_labels": json.dumps(a["test_type_labels"]),
                "job_levels":       json.dumps(a["job_levels"]),
                "duration":         a["duration"],
                "remote":           str(a["remote"]),
                "adaptive":         str(a["adaptive"]),
                "description":      a["description"][:400],
            })

            # Build lookup indexes
            self._url_index[a["url"]] = a
            self._name_index[a["name"].lower()] = a

        self._collection.add(documents=documents, ids=ids, metadatas=metadatas)
        self._loaded = True
        print(f"Index ready. {len(documents)} documents embedded.")
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_types: Optional[List[str]] = None,
        filter_job_level: Optional[str] = None,
        require_remote: bool = False,
        require_adaptive: bool = False,
    ) -> List[Dict]:
        """
        Semantic search over catalog.

        Args:
            query:            Natural language search query.
            top_k:            Max results to return.
            filter_types:     Optional list of test type codes to prefer (post-filter).
            filter_job_level: Optional job level string to prefer (post-filter).
            require_remote:   If True, only return remote-compatible assessments.
            require_adaptive: If True, only return adaptive assessments.

        Returns:
            List of assessment dicts sorted by relevance, enriched with relevance_score.
        """
        if not self._loaded or self._collection is None:
            return []

        fetch_n = min(top_k * 3, len(self.assessments))

        results = self._collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["metadatas", "distances"],
        )

        output = []
        seen = set()

        if not results or not results["metadatas"]:
            return []

        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            name = meta["name"]
            if name in seen:
                continue
            seen.add(name)

            test_types = json.loads(meta.get("test_types", "[]"))
            job_levels = json.loads(meta.get("job_levels", "[]"))
            remote = meta.get("remote") == "True"
            adaptive = meta.get("adaptive") == "True"

            # Hard filters
            if require_remote and not remote:
                continue
            if require_adaptive and not adaptive:
                continue

            item = {
                "name":             name,
                "url":              meta["url"],
                "test_type":        meta["test_type"],
                "test_types":       test_types,
                "test_type_labels": json.loads(meta.get("test_type_labels", "[]")),
                "job_levels":       job_levels,
                "duration":         meta.get("duration", ""),
                "remote":           remote,
                "adaptive":         adaptive,
                "description":      meta.get("description", ""),
                "relevance_score":  round(1 - dist, 4),
            }
            output.append(item)

        # Soft boost: bring type-matching items to the front
        if filter_types:
            def sort_key(x):
                has_type = any(t in x["test_types"] for t in filter_types)
                return (0 if has_type else 1, -x["relevance_score"])
            output.sort(key=sort_key)

        # Soft boost: bring job-level-matching items to the front
        if filter_job_level:
            def sort_key_level(x):
                has_level = filter_job_level in x["job_levels"]
                return (0 if has_level else 1, -x["relevance_score"])
            output.sort(key=sort_key_level)

        return output[:top_k]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def get_by_url(self, url: str) -> Optional[Dict]:
        return self._url_index.get(url)

    def get_by_name(self, name: str) -> Optional[Dict]:
        name_lower = name.lower().strip()
        # Exact
        if name_lower in self._name_index:
            return self._name_index[name_lower]
        # Partial
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
        """Map a user's seniority description to an SHL job level string."""
        lower = text.lower()
        for alias, canonical in JOB_LEVEL_ALIASES.items():
            if alias in lower:
                return canonical
        return None


# Singleton — shared across the app
retriever = CatalogRetriever()
