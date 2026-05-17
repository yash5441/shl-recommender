"""
agent.py - Core agent logic.
"""

import json
import os
import re
import time
from typing import List, Dict, Optional, Tuple

import google.generativeai as genai

from models import Message, Recommendation, ChatResponse
from prompts import SYSTEM_PROMPT, CATALOG_INJECTION_TEMPLATE
from retriever import retriever

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-lite"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Required info before recommending
# ---------------------------------------------------------------------------
ROLE_KEYWORDS = [
    "developer", "engineer", "manager", "analyst", "sales", "designer",
    "executive", "director", "ceo", "cxo", "leader", "consultant",
    "recruiter", "accountant", "nurse", "teacher", "officer", "rep",
    "associate", "coordinator", "specialist", "architect", "devops",
    "scientist", "marketing", "java", "python", "software", "finance",
    "hr", "operations", "support", "service", "administrator"
]

SENIORITY_KEYWORDS = [
    "entry", "junior", "graduate", "mid", "senior", "experienced",
    "manager", "lead", "director", "executive", "vp", "cxo", "ceo",
    "intern", "years", "level", "fresher", "experienced"
]

OUT_OF_SCOPE = [
    "salary", "legal", "lawsuit", "gdpr", "discriminat",
    "ignore previous", "disregard", "pretend", "jailbreak",
    "forget instructions", "act as", "competitor", "interview tips",
    "resume", "cover letter"
]

COMPARE_SIGNALS = [
    "difference", "compare", "vs", "versus", "better than",
    "which one", "how does", "compared to", "distinguish"
]

REFINE_SIGNALS = [
    "actually", "instead", "add", "remove", "also include", "drop",
    "change", "update", "without", "exclude", "plus", "and also"
]

JOB_LEVEL_ALIASES = {
    "entry": "Entry-Level", "junior": "Entry-Level",
    "graduate": "Graduate", "intern": "Entry-Level",
    "mid": "Mid-Professional", "mid-level": "Mid-Professional",
    "senior": "Professional Individual Contributor",
    "experienced": "Professional Individual Contributor",
    "manager": "Manager", "lead": "Front Line Manager",
    "supervisor": "Supervisor", "director": "Director",
    "executive": "Executive", "vp": "Executive",
    "ceo": "Executive", "cxo": "Executive",
}

TYPE_KEYWORDS = {
    "personality": "P", "behaviour": "P", "behavior": "P", "opq": "P",
    "ability": "A", "aptitude": "A", "cognitive": "A", "numerical": "A",
    "verbal": "A", "reasoning": "A", "inductive": "A",
    "knowledge": "K", "coding": "K", "programming": "K",
    "technical": "K", "java": "K", "python": "K", "sql": "K",
    "motivation": "M", "drive": "M",
    "situational": "B", "sjt": "B",
    "competenc": "C", "leadership": "C",
    "simulation": "S", "360": "D",
}


# ---------------------------------------------------------------------------
# Context checks
# ---------------------------------------------------------------------------
def has_enough_context(messages: List[Message]) -> Tuple[bool, str]:
    all_text = " ".join(m.content.lower() for m in messages if m.role == "user")
    has_role = any(w in all_text for w in ROLE_KEYWORDS)
    has_seniority = any(w in all_text for w in SENIORITY_KEYWORDS)
    if not has_role:
        return False, "job_role"
    if not has_seniority:
        return False, "seniority"
    return True, ""


def classify_intent(messages: List[Message]) -> str:
    if not messages:
        return "clarify"
    last = next((m.content.lower() for m in reversed(messages) if m.role == "user"), "")
    for s in OUT_OF_SCOPE:
        if s in last:
            return "out_of_scope"
    for s in COMPARE_SIGNALS:
        if s in last:
            return "compare"
    has_prior = any(
        m.role == "assistant" and "recommendations" in m.content
        for m in messages
    )
    if has_prior:
        for s in REFINE_SIGNALS:
            if s in last:
                return "refine"
    return "recommend"


def extract_type_filters(messages: List[Message]) -> List[str]:
    filters = []
    all_text = " ".join(m.content.lower() for m in messages if m.role == "user")
    for kw, code in TYPE_KEYWORDS.items():
        if kw in all_text and code not in filters:
            filters.append(code)
    return filters


def extract_job_level(messages: List[Message]) -> Optional[str]:
    all_text = " ".join(m.content.lower() for m in messages if m.role == "user")
    for alias, canonical in JOB_LEVEL_ALIASES.items():
        if alias in all_text:
            return canonical
    return None


def build_search_query(messages: List[Message]) -> str:
    user_msgs = [m.content for m in messages if m.role == "user"]
    return " ".join(user_msgs[-3:])


# ---------------------------------------------------------------------------
# Catalog context - SMALLER to avoid Gemini overflow
# ---------------------------------------------------------------------------
def build_catalog_context(messages: List[Message], intent: str) -> str:
    if not retriever.is_loaded():
        return "CATALOG NOT AVAILABLE."

    query = build_search_query(messages)
    type_filters = extract_type_filters(messages)
    job_level = extract_job_level(messages)

    # Fetch only 8 results to keep prompt small and avoid Gemini overflow
    results = retriever.search(
        query,
        top_k=8,
        filter_types=type_filters if type_filters else None,
        filter_job_level=job_level,
    )

    # If type filters gave too few, top up
    if type_filters and len(results) < 4:
        extra = retriever.search(query, top_k=8)
        seen = {r["url"] for r in results}
        results += [r for r in extra if r["url"] not in seen]
        results = results[:8]

    lines = []
    for item in results:
        types = " | ".join(item.get("test_type_labels", [item["test_type"]]))
        desc = item.get("description", "")[:150]
        lines.append(
            f"Name: {item['name']}\n"
            f"Type: {types}\n"
            f"URL: {item['url']}\n"
            f"Info: {desc}"
        )

    catalog_text = "\n---\n".join(lines)
    return CATALOG_INJECTION_TEMPLATE.format(catalog_text=catalog_text)


# ---------------------------------------------------------------------------
# Gemini call - simplified, no JSON mode (causes issues on some keys)
# ---------------------------------------------------------------------------
def call_gemini(messages: List[Message], catalog_context: str) -> str:
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT + "\n\n" + catalog_context,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=800,
        ),
    )

    gemini_history = []
    for msg in messages[:-1]:
        role = "model" if msg.role == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg.content]})

    last_content = messages[-1].content
    chat = model.start_chat(history=gemini_history)

    for attempt in range(2):
        try:
            response = chat.send_message(last_content)
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt == 0:
                print(f"[Agent] Rate limited, waiting 30s...")
                time.sleep(30)
                continue
            raise


# ---------------------------------------------------------------------------
# Parse Gemini response
# ---------------------------------------------------------------------------
def parse_and_validate(raw: str) -> ChatResponse:
    print(f"[Parse] Raw output: {raw[:300]}")

    clean = raw.strip()
    # Strip markdown fences
    clean = re.sub(r"^```json\s*", "", clean)
    clean = re.sub(r"^```\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean).strip()

    # Extract JSON object
    json_match = re.search(r'\{.*\}', clean, re.DOTALL)
    if json_match:
        clean = json_match.group(0)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"[Parse] JSON error: {e} | Raw: {raw[:300]}")
        reply_match = re.search(r'"reply"\s*:\s*"([^"]+)"', clean)
        reply = reply_match.group(1) if reply_match else "Could you tell me more about the role you are hiring for?"
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    reply = data.get("reply", "What role are you hiring for?")
    end_of_conversation = bool(data.get("end_of_conversation", False))
    raw_recs = data.get("recommendations", [])

    valid_urls = retriever.get_valid_urls()
    validated = []

    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name", "")).strip()
        url = str(rec.get("url", "")).strip()
        test_type = str(rec.get("test_type", "A")).strip().upper()

        print(f"[Validate] '{name}' | '{url}'")

        if not name:
            continue

        if url not in valid_urls:
            match = retriever.get_by_name(name)
            if match:
                url = match["url"]
                test_type = match["test_type"]
                print(f"[Validate] Fixed via name: {url}")
            else:
                print(f"[Validate] Skipped hallucinated: '{name}'")
                continue

        validated.append(Recommendation(name=name, url=url, test_type=test_type))

    return ChatResponse(
        reply=reply,
        recommendations=validated[:10],
        end_of_conversation=end_of_conversation,
    )


# ---------------------------------------------------------------------------
# Force recommendations fallback
# ---------------------------------------------------------------------------
def force_recommendations(messages: List[Message]) -> List[Recommendation]:
    query = build_search_query(messages)
    type_filters = extract_type_filters(messages)
    job_level = extract_job_level(messages)
    results = retriever.search(
        query, top_k=5,
        filter_types=type_filters if type_filters else None,
        filter_job_level=job_level,
    )
    return [
        Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
        for r in results
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_agent(messages: List[Message]) -> ChatResponse:
    if not messages:
        return ChatResponse(
            reply="Hello! I'm your SHL Assessment Recommender. What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    if messages[-1].role != "user":
        return ChatResponse(
            reply="What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    turn_count = len(messages)
    intent = classify_intent(messages)
    ready, missing = has_enough_context(messages)

    print(f"[Agent] Turn {turn_count} | Intent: {intent} | Ready: {ready} | Missing: {missing}")

    # Out of scope
    if intent == "out_of_scope":
        return ChatResponse(
            reply="I can only help with SHL assessment selection. What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Not enough context yet — force clarify
    if not ready and turn_count < 7:
        intent = "clarify"

    # Near turn limit — force recommend
    if turn_count >= 7:
        intent = "recommend"
        ready = True

    catalog_context = build_catalog_context(messages, intent)

    try:
        raw = call_gemini(messages, catalog_context)
    except Exception as e:
        print(f"[Agent] Gemini error: {e}")
        recs = force_recommendations(messages)
        return ChatResponse(
            reply="I couldn't reach the AI service, using catalog directly.",
            recommendations=recs,
            end_of_conversation=False,
        )

    response = parse_and_validate(raw)

    # Hard rule 1: not ready = no recommendations
    if not ready:
        response.recommendations = []
        response.end_of_conversation = False

    # Hard rule 2: reply has question = no recommendations
    if "?" in response.reply:
        response.recommendations = []
        response.end_of_conversation = False

    # Hard rule 3: ready but Gemini returned empty = force from retriever
    if ready and not response.recommendations and "?" not in response.reply:
        print("[Agent] Ready but empty recs — forcing from retriever.")
        response.recommendations = force_recommendations(messages)

    # Safety net at turn limit
    if turn_count >= 7 and not response.recommendations:
        response.recommendations = force_recommendations(messages)

    print(f"[Agent] Recs: {len(response.recommendations)} | EOC: {response.end_of_conversation}")
    return response