#!/usr/bin/env python3
"""
Tier 1C: Post-processing optimization for Akkadian-English translation.

Implements:
1. Sliding-window phrase deduplication (from chunky-v1.5 top solution)
2. Fragment trimming (remove trailing incomplete text)
3. Gap marker cleanup
4. Repeated word/phrase removal
5. Vectorized batch processing

Can be applied on top of any model output to improve BLEU.
"""

import re
from typing import List


def remove_phrase_repeats(text: str, min_phrase_len: int = 3, max_phrase_len: int = 8) -> str:
    """
    Sliding-window phrase deduplication.
    Detects and removes consecutively repeated multi-word phrases.

    E.g., "two talents of gold, two talents of gold, two talents of gold"
    → "two talents of gold,"
    """
    words = text.split()
    if len(words) < min_phrase_len * 2:
        return text

    # Check from longest phrases to shortest, iterate until stable
    changed = True
    while changed:
        changed = False
        for phrase_len in range(max_phrase_len, min_phrase_len - 1, -1):
            i = 0
            result_words = []
            while i < len(words):
                if i + phrase_len * 2 <= len(words):
                    phrase = words[i:i + phrase_len]
                    next_phrase = words[i + phrase_len:i + phrase_len * 2]
                    if phrase == next_phrase:
                        # Found a repeat - keep one copy, skip all duplicates
                        result_words.extend(phrase)
                        i += phrase_len
                        while i + phrase_len <= len(words) and words[i:i + phrase_len] == phrase:
                            i += phrase_len
                        changed = True
                        continue
                result_words.append(words[i])
                i += 1
            words = result_words

    return ' '.join(words)


def remove_word_repeats(text: str) -> str:
    """Remove immediately repeated words: 'destroyed destroyed destroyed' → 'destroyed'."""
    # Handle word repeats with optional trailing punctuation (3+ repeats only):
    # "destroyed, destroyed, destroyed," → "destroyed,"
    text = re.sub(r'\b(\w+)([,;]?\s+\1[,;]?){2,}\b', r'\1', text)
    # "destroyed destroyed destroyed" → "destroyed" (3+ repeats only)
    text = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', text)
    # Clean up resulting double commas or spaces
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text


def trim_fragment(text: str, min_length: int = 50) -> str:
    """
    Remove trailing incomplete sentence fragments.
    If text is long enough and doesn't end with proper punctuation,
    trim back to the last sentence boundary.
    """
    text = text.strip()
    if not text:
        return text

    # If text ends with proper punctuation, it's fine
    if text[-1] in '.!?)"\'':
        return text

    # Only trim if text is long enough (short texts might legitimately lack punctuation)
    if len(text) < min_length:
        return text

    # Find last sentence-ending punctuation
    last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    last_paren = text.rfind(')')

    # Use the later of period or closing paren
    cut_point = max(last_period, last_paren)

    if cut_point > len(text) * 0.5:  # Only trim if we keep at least 50% of text
        return text[:cut_point + 1]

    return text


def clean_gap_markers(text: str) -> str:
    """Clean up gap markers and normalize spacing."""
    # Remove leftover gap markers
    text = text.replace('<big_gap>', '...')
    text = text.replace('<gap>', '...')
    # Normalize multiple periods
    text = re.sub(r'\.{4,}', '...', text)
    # Clean multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def clean_special_chars(text: str) -> str:
    """Remove/normalize special characters that shouldn't appear in English output."""
    # Remove subscript numbers that leaked from transliteration
    text = re.sub(r'[₂₃₄₅₆₇₈₉]', '', text)
    # Remove cuneiform brackets that leaked
    text = re.sub(r'[⸢⸣⌈⌉⌊⌋]', '', text)
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    # Clean up resulting double spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def fix_punctuation(text: str) -> str:
    """Fix common punctuation issues."""
    # Remove space before punctuation (but NOT before ...)
    text = re.sub(r'\s+([,;:!?])', r'\1', text)
    # Add space after punctuation if missing (but not for decimals or ellipsis)
    text = re.sub(r'([,;:!?])([A-Za-z])', r'\1 \2', text)
    # Fix double punctuation (but preserve ... ellipsis)
    text = re.sub(r'([,;:!?])\1+', r'\1', text)
    # Fix space inside parentheses
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    return text


def postprocess_single(text: str) -> str:
    """Apply all post-processing steps to a single translation."""
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Step 1: Remove phrase repeats (longest patterns first)
    text = remove_phrase_repeats(text)

    # Step 2: Remove word repeats (3+ only)
    text = remove_word_repeats(text)

    # Step 3: Clean gap markers
    text = clean_gap_markers(text)

    # Step 4: Clean special characters
    text = clean_special_chars(text)

    # Step 5: Fix punctuation (preserves ellipsis)
    text = fix_punctuation(text)

    # NOTE: trim_fragment disabled - with beam=4, lp=1.05, BP is already ~1.0
    # Trimming would only reduce BP and hurt BLEU

    return text


def postprocess_batch(translations: List[str]) -> List[str]:
    """Apply post-processing to a batch of translations."""
    return [postprocess_single(t) for t in translations]


def evaluate_postprocessing(translations: List[str], references: List[str]):
    """Evaluate BLEU before and after post-processing."""
    from sacrebleu import corpus_bleu

    # Before
    bleu_before = corpus_bleu(translations, [references])
    print(f"Before postprocessing: {bleu_before}")

    # After
    processed = postprocess_batch(translations)
    bleu_after = corpus_bleu(processed, [references])
    print(f"After postprocessing:  {bleu_after}")
    print(f"Improvement: {bleu_after.score - bleu_before.score:+.2f}")

    # Show examples that changed
    changed = 0
    for i in range(len(translations)):
        if translations[i] != processed[i]:
            changed += 1
    print(f"Changed: {changed}/{len(translations)} ({100*changed/len(translations):.1f}%)")

    return bleu_before.score, bleu_after.score


if __name__ == "__main__":
    # Test with some examples
    examples = [
        "destroyed, destroyed, destroyed, devastated, (and) burned with fire.",
        "(As for) the water of the water of the water of the gardens",
        "two talents of gold, two talents of gold, two talents of gold, two talents of gold",
        "the king king of Assyria, the great king king king",
        "I brought out his land 40 (and) 5 days after his city, and (thus) I broug",
        "The goddess Ea, lord of everything, the one who loves everything, the one who loves everything, the goddess Ištar",
        "normal translation without issues.",
    ]

    print("Post-processing examples:")
    print("=" * 70)
    for ex in examples:
        processed = postprocess_single(ex)
        if ex != processed:
            print(f"BEFORE: {ex}")
            print(f"AFTER:  {processed}")
            print()
        else:
            print(f"UNCHANGED: {ex}")
            print()
