"""Treat retrieved text as untrusted data: detect and remove injected instructions."""

import re

from src.retrieval import SearchHit

# Phrases that try to give the assistant new orders rather than state a policy.
# Simple patterns catch the known example; they will miss reworded attacks.
INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(prior|previous|above|earlier)\s+instructions",
        r"disregard\s+(all\s+)?(prior|previous|above|earlier)\s+(instructions|rules)",
        r"reveal\s+(the\s+)?(system\s+prompt|api\s+key|password|secret)",
    ]
]


def find_injection(text: str) -> str | None:
    """Return the first suspicious phrase found in text, or None if it looks clean."""
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def filter_evidence(hits: list[SearchHit]) -> tuple[list[SearchHit], list[str]]:
    """Drop chunks containing injected instructions; return (clean_hits, safety_events)."""
    clean, events = [], []
    for hit in hits:
        phrase = find_injection(hit.chunk.text)
        if phrase:
            events.append(f"removed {hit.chunk.chunk_id}: suspicious instruction '{phrase}'")
        else:
            clean.append(hit)
    return clean, events
