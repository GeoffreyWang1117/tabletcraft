"""Input classifier: determine text type and appropriate processing mode."""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List


class InputType(Enum):
    NAME = "name"               # Personal name or short proper noun
    SHORT_PHRASE = "short"      # Short sentence suitable for experience mode
    HISTORICAL = "historical"   # Historical-style text, suitable for full pipeline
    MODERN_HEAVY = "modern"     # Dense modern concepts, needs warning
    ANOMALOUS = "anomalous"     # Garbage, injection, unsupported


@dataclass
class Classification:
    input_type: InputType
    confidence: float  # 0-1
    warnings: List[str]
    suggested_mode: str  # "experience", "educational", "research"


# Modern concept indicators that have no Akkadian equivalent
_MODERN_TERMS = re.compile(
    r"\b(internet|computer|phone|email|democracy|parliament|vaccine|"
    r"electricity|airplane|satellite|bitcoin|algorithm|software|"
    r"photograph|television|radio|nuclear|quantum|robot|AI|"
    r"capitalism|communism|fascism|socialism)\b", re.I
)

# Historical/archaic style indicators
_HISTORICAL_MARKERS = re.compile(
    r"\b(king|lord|temple|priest|god|goddess|palace|throne|"
    r"silver|gold|barley|sheep|ox|tablet|seal|merchant|"
    r"decree|oath|offering|servant|slave|donkey|caravan)\b", re.I
)

# Anomalous input patterns
_ANOMALOUS_PATTERNS = [
    re.compile(r"<[^>]+>"),           # HTML tags
    re.compile(r"\{[^}]+\}"),         # Code-like braces
    re.compile(r"[\x00-\x08\x0e-\x1f]"),  # Control chars
    re.compile(r"(.)\1{10,}"),        # 10+ char repetition
]


def classify(text: str) -> Classification:
    """Classify input text to determine processing mode.

    Returns a Classification with type, confidence, warnings, and suggested mode.
    """
    if not text or not text.strip():
        return Classification(InputType.ANOMALOUS, 1.0, ["Empty input"], "experience")

    text = text.strip()
    warnings = []

    # Check for anomalous input
    for pat in _ANOMALOUS_PATTERNS:
        if pat.search(text):
            return Classification(
                InputType.ANOMALOUS, 0.9,
                ["Input contains unsupported characters or patterns"],
                "experience",
            )

    # Very long input
    if len(text) > 2000:
        return Classification(
            InputType.ANOMALOUS, 0.8,
            ["Input exceeds maximum length (2000 chars)"],
            "experience",
        )

    # Emoji / non-Latin heavy
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.3:
        warnings.append("High proportion of non-ASCII characters")

    words = text.split()
    word_count = len(words)

    # Name detection: 1-3 words, capitalized, no verbs
    if word_count <= 3 and all(w[0].isupper() for w in words if w):
        return Classification(InputType.NAME, 0.9, warnings, "experience")

    # Short phrase: <= 10 words
    if word_count <= 10:
        # Check for modern concepts
        modern_hits = len(_MODERN_TERMS.findall(text))
        if modern_hits > 0:
            warnings.append(
                f"Contains {modern_hits} modern concept(s) with no direct Akkadian equivalent"
            )
            return Classification(InputType.MODERN_HEAVY, 0.7, warnings, "educational")

        return Classification(InputType.SHORT_PHRASE, 0.85, warnings, "experience")

    # Longer text: check historical vs modern
    hist_hits = len(_HISTORICAL_MARKERS.findall(text))
    modern_hits = len(_MODERN_TERMS.findall(text))

    if modern_hits > hist_hits and modern_hits >= 2:
        warnings.append(
            f"Text contains {modern_hits} modern concepts vs {hist_hits} historical terms"
        )
        return Classification(InputType.MODERN_HEAVY, 0.7, warnings, "educational")

    if hist_hits >= 2:
        return Classification(InputType.HISTORICAL, 0.8, warnings, "educational")

    # Default: treat as short phrase if short enough, else historical
    if word_count <= 20:
        return Classification(InputType.SHORT_PHRASE, 0.6, warnings, "experience")

    return Classification(InputType.HISTORICAL, 0.5, warnings, "research")
