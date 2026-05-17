"""
catalog.py - Loads and normalises the SHL product catalog from the provided JSON file.
No scraping needed — the JSON is bundled with the project.

The JSON has control characters embedded in some description strings (literal newlines
inside quoted values). This module cleans that on load.

Fields in the source JSON:
    entity_id, name, link, scraped_at,
    job_levels (list), languages (list),
    duration, remote, adaptive, description,
    keys (list of test-type strings)
"""

import json
import os
import re
from typing import List, Dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(_HERE, "data", "shl_product_catalog.json")

# ---------------------------------------------------------------------------
# Test-type mapping  (full label -> letter code used in the API response)
# ---------------------------------------------------------------------------
KEY_TO_CODE: Dict[str, str] = {
    "Ability & Aptitude":             "A",
    "Assessment Exercises":           "E",
    "Biodata & Situational Judgment": "B",
    "Competencies":                   "C",
    "Development & 360":              "D",
    "Knowledge & Skills":             "K",
    "Motivation":                     "M",
    "Personality & Behavior":         "P",
    "Simulations":                    "S",
}

CODE_TO_LABEL: Dict[str, str] = {v: k for k, v in KEY_TO_CODE.items()}


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------
def _clean_raw(raw: bytes) -> str:
    """
    The JSON file contains literal newline characters embedded inside some
    string values (e.g. a product name split across two lines).
    Standard json.loads rejects these as invalid control characters.

    Strategy: decode as UTF-8 (replacing bad bytes), then replace any
    bare newline that appears inside a JSON string value with a space.
    """
    text = raw.decode("utf-8", errors="replace")
    fixed = re.sub(
        r'(?<=")((?:[^"\\]|\\.)*)(\n)((?:[^"\\]|\\.)*?)(?=")',
        lambda m: m.group(1) + " " + m.group(3),
        text,
    )
    return fixed


def _keys_to_codes(keys: List[str]) -> List[str]:
    """Convert a list of full key labels to letter codes."""
    codes = []
    for k in keys:
        code = KEY_TO_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return codes or ["A"]


def _normalise(raw: Dict) -> Dict:
    """
    Convert one raw catalog record into the normalised format used internally.

    Output schema:
        name              str   - assessment name
        url               str   - canonical SHL product URL
        test_type         str   - primary letter code (first in keys list)
        test_types        list  - all letter codes
        test_type_labels  list  - full label strings (for search richness)
        job_levels        list  - e.g. ["Mid-Professional", "Manager"]
        languages         list  - e.g. ["English (USA)", "French"]
        duration          str   - e.g. "30 minutes" or "" or "Untimed"
        remote            bool
        adaptive          bool
        description       str   - product description (capped at 1000 chars)
        entity_id         str
    """
    codes = _keys_to_codes(raw.get("keys", []))
    desc = (raw.get("description") or "").strip()
    desc = re.sub(r"\s+", " ", desc)[:1000]

    return {
        "entity_id":        str(raw.get("entity_id", "")),
        "name":             raw.get("name", "").strip(),
        "url":              raw.get("link", "").strip(),
        "test_type":        codes[0],
        "test_types":       codes,
        "test_type_labels": raw.get("keys", []),
        "job_levels":       raw.get("job_levels", []),
        "languages":        raw.get("languages", []),
        "duration":         (raw.get("duration") or "").strip(),
        "remote":           raw.get("remote", "no").lower() == "yes",
        "adaptive":         raw.get("adaptive", "no").lower() == "yes",
        "description":      desc,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_catalog(path: str = CATALOG_PATH) -> List[Dict]:
    """
    Load, clean, and normalise the SHL product catalog.
    Raises FileNotFoundError if the JSON is missing.
    Returns a list of normalised assessment dicts.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Catalog not found at {path}. "
            "Make sure shl_product_catalog.json is in the data/ directory."
        )

    with open(path, "rb") as f:
        raw_bytes = f.read()

    cleaned = _clean_raw(raw_bytes)
    raw_list: List[Dict] = json.loads(cleaned)

    catalog = [
        _normalise(r) for r in raw_list
        if r.get("name") and r.get("link")
    ]
    return catalog


def build_search_document(item: Dict) -> str:
    """
    Build a rich text blob for embedding / semantic search.
    More fields = better retrieval quality.
    """
    parts = [f"Assessment: {item['name']}."]
    if item["test_type_labels"]:
        parts.append(f"Type: {', '.join(item['test_type_labels'])}.")
    if item["job_levels"]:
        parts.append(f"Suitable for: {', '.join(item['job_levels'])}.")
    if item["duration"]:
        parts.append(f"Duration: {item['duration']}.")
    if item["languages"]:
        parts.append(f"Languages: {', '.join(item['languages'][:5])}.")
    if item["remote"]:
        parts.append("Remote testing: yes.")
    if item["adaptive"]:
        parts.append("Adaptive testing: yes.")
    if item["description"]:
        parts.append(item["description"])
    return " ".join(parts)


def get_catalog_summary(catalog: List[Dict], max_items: int = 377) -> str:
    """Compact multi-line summary of catalog for prompt injection."""
    lines = []
    for a in catalog[:max_items]:
        types_str = " | ".join(a["test_type_labels"])
        levels_str = ", ".join(a["job_levels"][:4]) if a["job_levels"] else "all levels"
        dur = f" | {a['duration']}" if a["duration"] else ""
        desc_preview = a["description"][:180] if a["description"] else ""
        lines.append(
            f"• {a['name']}  [{types_str}]  Levels: {levels_str}{dur}\n"
            f"  URL: {a['url']}\n"
            f"  {desc_preview}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Quick CLI check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    catalog = load_catalog()
    print(f"Loaded {len(catalog)} assessments.")

    from collections import Counter
    type_counts = Counter(
        label for a in catalog for label in a["test_type_labels"]
    )
    print("\nTest type distribution:")
    for label, count in type_counts.most_common():
        print(f"  {label}: {count}")

    print("\nSample records:")
    for a in catalog[:3]:
        print(f"  [{a['test_type']}] {a['name']} — {a['url']}")
        print(f"    Levels: {a['job_levels']}")
        print(f"    Duration: {a['duration']} | Remote: {a['remote']} | Adaptive: {a['adaptive']}")
        print(f"    Desc: {a['description'][:120]}")
        print()
