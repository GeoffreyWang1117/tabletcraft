"""Output validator: check transliteration quality before rendering.

The validator sits between model output and cuneiform conversion.
It catches unreliable outputs and triggers fallback instead of
confidently rendering wrong cuneiform.
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    valid: bool
    score: float  # 0-1, higher = more trustworthy
    issues: List[str] = field(default_factory=list)
    suggestion: str = ""  # "render", "render_with_caveat", "fallback"


# Known Akkadian syllable patterns (common CV, CVC, VC patterns)
_AKKADIAN_SYLLABLE = re.compile(
    r"^[a-zšṣṭḫāēīū][a-zšṣṭḫāēīū]*$", re.I
)

# Common logograms
_LOGOGRAMS = {
    "LUGAL", "DINGIR", "KUR", "URU", "É", "DUMU", "MAN", "EN",
    "KÙ", "BABBAR", "GÍN", "ITU", "KAM", "MEŠ", "ŠÀ", "GIŠ",
    "TÚG", "ANŠE", "GÚ", "KIŠIB", "IGI", "ÌR", "NAM",
}


def validate(transliteration: str, source_text: str = "") -> ValidationResult:
    """Validate a model-generated transliteration before cuneiform conversion.

    Args:
        transliteration: Model output (Akkadian transliteration)
        source_text: Original input (for length ratio check)

    Returns:
        ValidationResult with pass/fail, confidence score, and issues list.
    """
    if not transliteration or not transliteration.strip():
        return ValidationResult(
            valid=False, score=0.0,
            issues=["Empty transliteration"],
            suggestion="fallback",
        )

    text = transliteration.strip()
    issues = []
    score = 1.0

    # --- Check 1: Length anomaly ---
    if source_text:
        src_words = len(source_text.split())
        tgt_words = len(text.split())
        ratio = tgt_words / max(src_words, 1)
        if ratio > 5.0:
            issues.append(f"Output is {ratio:.1f}x longer than input (expansion anomaly)")
            score -= 0.3
        elif ratio < 0.1 and src_words > 3:
            issues.append(f"Output is {ratio:.1f}x shorter than input (collapse anomaly)")
            score -= 0.3

    # --- Check 2: Excessive repetition ---
    words = text.split()
    if len(words) >= 4:
        # Check for repeated bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        unique_ratio = len(set(bigrams)) / max(len(bigrams), 1)
        if unique_ratio < 0.3:
            issues.append(f"High repetition detected ({1-unique_ratio:.0%} repeated bigrams)")
            score -= 0.4

    # --- Check 3: Non-Akkadian characters ---
    # After stripping determinatives, Akkadian transliteration should be
    # mostly Latin letters + diacritics + hyphens + digits + spaces
    stripped = re.sub(r"\{[^}]+\}", "", text)  # Remove determinatives
    stripped = re.sub(r"[<>]", "", stripped)     # Remove gap markers
    suspicious = re.findall(r"[^\w\s\-₀₁₂₃₄₅₆₇₈₉.,;:šṣṭḫŠṢṬḪāēīūÂÊÎÛàèìùáéíú]", stripped)
    if len(suspicious) > len(words) * 0.2:
        issues.append(f"Contains {len(suspicious)} unexpected characters: {''.join(set(suspicious)[:10])}")
        score -= 0.2

    # --- Check 4: Task prefix leakage ---
    if re.match(r"translate (Akkadian|English) to", text, re.I):
        issues.append("Output contains task prefix (model leakage)")
        score -= 0.5

    # --- Check 5: English leakage ---
    # If output looks more like English than Akkadian
    english_words = {"the", "of", "and", "to", "in", "is", "was", "for", "that", "with"}
    word_set = set(w.lower() for w in words)
    english_overlap = len(word_set & english_words)
    if english_overlap >= 3 and english_overlap / max(len(words), 1) > 0.3:
        issues.append(f"Output appears to be English rather than Akkadian ({english_overlap} common English words)")
        score -= 0.4

    # --- Determine suggestion ---
    score = max(score, 0.0)

    if score >= 0.7:
        suggestion = "render"
    elif score >= 0.4:
        suggestion = "render_with_caveat"
    else:
        suggestion = "fallback"

    return ValidationResult(
        valid=score >= 0.4,
        score=score,
        issues=issues,
        suggestion=suggestion,
    )
