"""
agent.py - Core agent logic.
Classifies intent → retrieves from catalog → calls Gemini → validates → returns response.
"""
import json
import os
import re
from typing import List, Dict, Optional

from dotenv import load_dotenv
import google.generativeai as genai

# IMPORTANT: Load the .env file BEFORE trying to get the key
load_dotenv()

from models import Message, Recommendation, ChatResponse
from catalog import CODE_TO_LABEL
from prompts import SYSTEM_PROMPT, CATALOG_INJECTION_TEMPLATE
from retriever import retriever, JOB_LEVEL_ALIASES

# ---------------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------------
# Now this will successfully grab the key from your .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
print(f"[Startup] GEMINI_API_KEY loaded: {bool(GEMINI_API_KEY)} | length: {len(GEMINI_API_KEY)}")

# You can switch this back to the newer model now!
GEMINI_MODEL = "gemini-2.5-flash"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Intent signals
# ---------------------------------------------------------------------------
# agent.py - add this near the top

REQUIRED_INFO = {
    "job_role":    ["developer", "engineer", "manager", "analyst", "sales", "designer",
                    "executive", "director", "ceo", "cxo", "leader", "consultant",
                    "recruiter", "accountant", "nurse", "teacher", "officer", "rep",
                    "associate", "coordinator", "specialist", "architect", "devops",
                    "data scientist", "product manager", "marketing"],
    "seniority":   ["entry", "junior", "graduate", "mid", "senior", "experienced",
                    "manager", "lead", "director", "executive", "vp", "cxo", "ceo",
                    "intern", "fresher", "years", "level"],
}
COMPARE_SIGNALS = [
    "difference", "compare", "vs", "versus", "better than", "prefer",
    "which one", "what's the diff", "how does", "compared to", "distinguish",
]

OUT_OF_SCOPE_SIGNALS = [
    "salary", "legal advice", "lawsuit", "gdpr", "compliance law",
    "discriminat", "ignore previous", "disregard", "pretend you are",
    "jailbreak", "forget your instructions", "act as", "new persona",
    "competitor", "hire someone", "interview tips", "resume advice",
    "cover letter",
]

REFINE_SIGNALS = [
    "actually", "instead", "add", "remove", "also include", "drop",
    "change", "update", "without", "exclude", "plus", "and also",
    "no personality", "no cognitive", "only", "just the",
]

VAGUE_PATTERNS = [
    r"^i need (an?|some) assessments?\.?$",
    r"^help me find (an?|some) assessments?\.?$",
    r"^what assessment should i use\??$",
    r"^recommend (an?|some) assessments?\.?$",
    r"^can you recommend (an?|some) tests?\??$",
    r"^hello\.?$", r"^hi\.?$", r"^hey\.?$",
]

# Test type keywords used for extraction from conversation
TYPE_KEYWORDS: Dict[str, str] = {
    "personality":   "P", "behaviour":    "P", "behavior":     "P", "opq":         "P",
    "ability":       "A", "aptitude":     "A", "cognitive":    "A", "numerical":   "A",
    "verbal":        "A", "reasoning":    "A", "inductive":    "A", "deductive":   "A",
    "knowledge":     "K", "skills test":  "K", "coding":       "K", "programming": "K",
    "technical":     "K", "java":         "K", "python":       "K", "sql":         "K",
    "motivation":    "M", "drive":        "M",
    "situational":   "B", "sjt":          "B", "biodata":      "B",
    "competenc":     "C", "leadership":   "C",
    "simulation":    "S", "exercise":     "E",
    "360":           "D", "development":  "D", "feedback":     "D",
}

ROLE_HINTS = [
    "engineer", "developer", "manager", "sales", "scientist", "analyst",
    "customer service", "service rep", "support", "technician", "accountant",
    "designer", "administrator", "programmer", "consultant", "nurse",
    "teacher", "data", "software", "backend", "frontend", "java", "python",
]


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------
def has_enough_context(messages: List[Message]) -> tuple[bool, str]:
    """
    Check if the conversation has enough info to make a recommendation.
    Returns (ready, missing_dimension).
    """
    all_user_text = " ".join(
        m.content.lower() for m in messages if m.role == "user"
    )

    has_role = any(word in all_user_text for word in REQUIRED_INFO["job_role"])
    has_seniority = any(word in all_user_text for word in REQUIRED_INFO["seniority"])

    if not has_role:
        return False, "job_role"
    if not has_seniority:
        return False, "seniority"
    return True, ""
def classify_intent(messages: List[Message]) -> str:
    """Returns: 'clarify' | 'recommend' | 'refine' | 'compare' | 'out_of_scope'"""
    if not messages:
        return "clarify"

    last_user_msg = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user_msg = m.content.lower().strip()
            break

    if not last_user_msg:
        return "clarify"

    # Out of scope (highest priority)
    for signal in OUT_OF_SCOPE_SIGNALS:
        if signal in last_user_msg:
            return "out_of_scope"

    # Compare
    for signal in COMPARE_SIGNALS:
        if signal in last_user_msg:
            return "compare"

    # Refine — only if there's a prior recommendation
    has_prior_recs = _has_prior_recommendations(messages)
    if has_prior_recs:
        for signal in REFINE_SIGNALS:
            if signal in last_user_msg:
                return "refine"

    # Vague query
    for pattern in VAGUE_PATTERNS:
        if re.match(pattern, last_user_msg):
            return "clarify"

    if _looks_vague(last_user_msg):
        return "clarify"

    return "recommend"


def _has_prior_recommendations(messages: List[Message]) -> bool:
    """Check if any prior assistant message contained recommendations."""
    for m in messages:
        if m.role == "assistant":
            try:
                data = json.loads(m.content)
                if data.get("recommendations"):
                    return True
            except Exception:
                if '"recommendations"' in m.content and '"url"' in m.content:
                    return True
    return False


# ---------------------------------------------------------------------------
# Context extraction from conversation
# ---------------------------------------------------------------------------
def extract_type_filters(messages: List[Message]) -> List[str]:
    """Extract explicit test type preferences from the full conversation."""
    filters = []
    all_user_text = " ".join(m.content.lower() for m in messages if m.role == "user")
    for keyword, code in TYPE_KEYWORDS.items():
        if keyword in all_user_text and code not in filters:
            filters.append(code)
    return filters


def extract_job_level(messages: List[Message]) -> Optional[str]:
    """Try to extract a canonical job level from conversation."""
    all_user_text = " ".join(m.content.lower() for m in messages if m.role == "user")
    for alias, canonical in JOB_LEVEL_ALIASES.items():
        if alias in all_user_text:
            return canonical
    return None


def _looks_vague(text: str) -> bool:
    """Detect broad requests that need a clarifying question first."""
    generic_markers = (
        "assessment", "assessments", "test", "tests", "recommend",
        "recommendation", "find", "help", "company",
    )
    if not any(marker in text for marker in generic_markers):
        return False

    if any(hint in text for hint in ROLE_HINTS):
        return False

    if any(keyword in text for keyword in TYPE_KEYWORDS):
        return False

    if any(level in text for level in JOB_LEVEL_ALIASES):
        return False

    return True


def _build_clarify_reply(messages: List[Message]) -> str:
    """Ask one focused question to fill the biggest missing gap."""
    all_user_text = " ".join(m.content.lower() for m in messages if m.role == "user")
    if any(hint in all_user_text for hint in ROLE_HINTS):
        return "What seniority level is this role?"
    return "What role or job family are you hiring for?"


def _best_catalog_match(query: str):
    results = retriever.search(query, top_k=3)
    return results[0] if results else None


def _build_compare_reply(messages: List[Message]) -> str:
    """Summarize the compared assessments using catalog facts only."""
    query = build_search_query(messages).lower()
    targets = []

    if "opq" in query:
        opq = _best_catalog_match("OPQ")
        if opq:
            targets.append(opq)

    if "verify" in query:
        verify = _best_catalog_match("Verify")
        if verify:
            targets.append(verify)

    if len(targets) < 2:
        for item in retriever.search(build_search_query(messages), top_k=2):
            if item not in targets:
                targets.append(item)
            if len(targets) == 2:
                break

    if not targets:
        return "I couldn't find matching catalog entries to compare."

    parts = []
    for item in targets[:2]:
        type_labels = ", ".join(item.get("test_type_labels", [])) or item["test_type"]
        levels = ", ".join(item["job_levels"][:3]) if item.get("job_levels") else "all levels"
        duration = item.get("duration") or "duration not listed"
        parts.append(
            f"{item['name']} is a {type_labels} assessment for {levels} ({duration})."
        )

    if len(parts) == 2:
        return (
            f"{parts[0]} {parts[1]} "
            "In catalog terms, the main difference is the assessment type and the job-level fit."
        )
    return parts[0]


def build_search_query(messages: List[Message]) -> str:
    """Build retrieval query from last 3 user messages."""
    user_msgs = [m.content for m in messages if m.role == "user"]
    return " ".join(user_msgs[-3:])


# ---------------------------------------------------------------------------
# Catalog context building
# ---------------------------------------------------------------------------
def build_catalog_context(messages: List[Message], intent: str) -> str:
    """Build the catalog snippet to inject into the LLM prompt."""
    if not retriever.is_loaded():
        return "CATALOG NOT AVAILABLE — do not make any recommendations."

    query = build_search_query(messages)
    type_filters = extract_type_filters(messages)
    job_level = extract_job_level(messages)

    # For compare, fetch broader results targeting the named assessments
    if intent == "compare":
        results = retriever.search(query, top_k=15)
    else:
        results = retriever.search(
            query,
            top_k=10,
            filter_types=type_filters if type_filters else None,
            filter_job_level=job_level,
        )

        # If type filters produced too few results, top up with unfiltered
        if type_filters and len(results) < 5:
            extra = retriever.search(query, top_k=10)
            seen_urls = {r["url"] for r in results}
            results += [r for r in extra if r["url"] not in seen_urls]
            results = results[:10]

    catalog_text = _format_catalog_items(results)
    return CATALOG_INJECTION_TEMPLATE.format(catalog_text=catalog_text)


def _format_catalog_items(items: List[Dict]) -> str:
    """Format retrieval results for prompt injection."""
    lines = []
    for item in items:
        type_labels = " | ".join(item.get("test_type_labels", [item["test_type"]]))
        levels = ", ".join(item["job_levels"][:4]) if item["job_levels"] else "All levels"
        dur = f" | Duration: {item['duration']}" if item.get("duration") else ""
        remote = " | Remote: Yes" if item.get("remote") else ""
        adaptive = " | Adaptive: Yes" if item.get("adaptive") else ""
        desc = item.get("description", "")
        desc_part = f"\n    {desc[:200]}" if desc else ""

        lines.append(
            f"• Name: {item['name']}\n"
            f"  Type: [{type_labels}]\n"
            f"  Levels: {levels}{dur}{remote}{adaptive}\n"
            f"  URL: {item['url']}"
            f"{desc_part}"
        )
    return "\n\n".join(lines) if lines else "No matching assessments found in catalog."


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------
def call_gemini(messages: List[Message], catalog_context: str) -> str:
    """Call Gemini with full conversation + catalog context. Returns raw text."""
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT + "\n\n" + catalog_context,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )

    # Convert to Gemini history format (all but last message)
    gemini_history = []
    for msg in messages[:-1]:
        role = "model" if msg.role == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg.content]})

    last_content = messages[-1].content

    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(last_content)
    return response.text


# ---------------------------------------------------------------------------
# Response parsing + validation
# ---------------------------------------------------------------------------
def parse_and_validate(raw: str) -> ChatResponse:
    clean = raw.strip()
    clean = re.sub(r"^```json\s*", "", clean)
    clean = re.sub(r"^```\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean).strip()

    # Sometimes Gemini wraps in extra text before the JSON
    # Try to extract just the JSON object
    json_match = re.search(r'\{.*\}', clean, re.DOTALL)
    if json_match:
        clean = json_match.group(0)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        print(f"[Parse] Raw Gemini output: {raw[:500]}")
        reply_match = re.search(r'"reply"\s*:\s*"([^"]+)"', clean)
        reply = reply_match.group(1) if reply_match else "Could you please rephrase that?"
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    reply = data.get("reply", "How can I help you find the right SHL assessment?")
    end_of_conversation = bool(data.get("end_of_conversation", False))
    raw_recs = data.get("recommendations", [])

    valid_urls = retriever.get_valid_urls()
    validated: List[Recommendation] = []

    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()
        test_type = str(rec.get("test_type", "A")).strip().upper()

        print(f"[Validate] Checking: '{name}' | '{url}'")

        if not name:
            continue

        if url not in valid_urls:
            match = retriever.get_by_name(name)
            if match:
                url = match["url"]
                test_type = match["test_type"]
                print(f"[Validate] Fixed via name lookup: {url}")
            else:
                print(f"[Validate] Skipping hallucinated: '{name}' | '{url}'")
                continue

        validated.append(Recommendation(name=name, url=url, test_type=test_type))

    return ChatResponse(
        reply=reply,
        recommendations=validated[:10],
        end_of_conversation=end_of_conversation,
    )

# ---------------------------------------------------------------------------
# Fallback: direct retrieval when LLM gives empty recs near turn limit
# ---------------------------------------------------------------------------
def _force_recommendations(messages: List[Message]) -> List[Recommendation]:
    query = build_search_query(messages)
    type_filters = extract_type_filters(messages)
    job_level = extract_job_level(messages)

    results = retriever.search(
        query, top_k=10,
        filter_types=type_filters if type_filters else None,
        filter_job_level=job_level,
    )

    if type_filters:
        seen_urls = {r["url"] for r in results}
        missing_types = [
            code for code in type_filters
            if code not in {r["test_type"] for r in results}
        ]
        extras = []
        for code in missing_types:
            type_hint = CODE_TO_LABEL.get(code, code)
            extra_results = retriever.search(
                f"{query} {type_hint}",
                top_k=20,
                filter_types=[code],
                filter_job_level=job_level,
            )
            for extra in extra_results:
                if extra["test_type"] != code:
                    continue
                if extra["url"] not in seen_urls:
                    extras.append(extra)
                    seen_urls.add(extra["url"])
                    break
        results = extras + [r for r in results if r["url"] in seen_urls]

    return [
        Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
        for r in results[:5]
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_agent(messages: List[Message]) -> ChatResponse:
    if not messages:
        return ChatResponse(
            reply=(
                "Hello! I'm your SHL Assessment Recommender. "
                "Tell me about the role you're hiring for — job title, seniority level, "
                "and the key skills or behaviours you want to assess — and I'll suggest "
                "the right SHL assessments from the catalog."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    if messages[-1].role != "user":
        return ChatResponse(
            reply="Please go ahead — what role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    turn_count = len(messages)
    intent = classify_intent(messages)

    print(f"[Agent] Turn {turn_count} | Intent: {intent}")

    # Hard out-of-scope refusal
    if intent == "out_of_scope":
        return ChatResponse(
            reply=(
                "I can only help with selecting SHL assessments. "
                "I'm not able to assist with general hiring advice, legal questions, "
                "or anything outside the SHL product catalog. "
                "What role are you looking to assess candidates for?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # ---------------------------------------------------------------
    # CORE LOGIC: check if we have enough context to recommend
    # ---------------------------------------------------------------
    ready, missing = has_enough_context(messages)

    # Force clarify if not ready (unless near turn limit)
    if not ready and turn_count < 7:
        intent = "clarify"
        print(f"[Agent] Not enough context — missing: {missing}. Forcing clarify.")

    # Force recommend if near turn limit regardless
    if turn_count >= 7:
        intent = "recommend"
        ready = True
        print("[Agent] Turn limit approaching — forcing recommend.")

    catalog_context = build_catalog_context(messages, intent)

    try:
        raw = call_gemini(messages, catalog_context)
    except Exception as e:
        print(f"[Agent] Gemini error: {e}")
        forced = _force_recommendations(messages)
        return ChatResponse(
            reply="I couldn't reach the AI service, so I'm using the catalog directly.",
            recommendations=forced,
            end_of_conversation=False,
        )

    response = parse_and_validate(raw)

    # ---------------------------------------------------------------
    # HARD ENFORCEMENT: if not ready, always wipe recommendations
    # No matter what Gemini returns
    # ---------------------------------------------------------------
    if not ready:
        if response.recommendations:
            print(f"[Agent] Wiped {len(response.recommendations)} premature recs — context incomplete.")
        response.recommendations = []
        response.end_of_conversation = False

    # ---------------------------------------------------------------
    # HARD ENFORCEMENT: if reply has a question, wipe recommendations
    # Cannot ask and recommend simultaneously
    # ---------------------------------------------------------------
    if "?" in response.reply:
        if response.recommendations:
            print(f"[Agent] Wiped recs — reply contains a question.")
        response.recommendations = []
        response.end_of_conversation = False

    # Safety net at turn limit
    if turn_count >= 7 and not response.recommendations:
        forced = _force_recommendations(messages)
        if forced:
            response.recommendations = forced
            response.reply += "\n\nBased on what you've shared, here are my best recommendations:"

    print(
        f"[Agent] Reply: {response.reply[:80]}... | "
        f"Recs: {len(response.recommendations)} | EOC: {response.end_of_conversation}"
    )
    return response