"""Cuneiform sign conversion: transliteration <-> Unicode cuneiform."""

import re
import json
from pathlib import Path
from typing import Dict, Optional


class CuneiformConverter:
    """Convert between Akkadian transliteration and Unicode cuneiform signs.

    Uses a mapping of 14,240+ transliteration values to cuneiform Unicode (U+12000-U+1254F).
    """

    def __init__(self, mapping_path: Optional[str] = None):
        if mapping_path is None:
            # Try default locations
            candidates = [
                Path(__file__).parent.parent.parent / "knowledge" / "sign_tables" / "transliteration_mapping.json",
                Path(__file__).parent.parent.parent / "dictionaries" / "processed" / "transliteration_mapping.json",
                Path.home() / ".cuneiscribe" / "transliteration_mapping.json",
            ]
            for p in candidates:
                if p.exists():
                    mapping_path = str(p)
                    break

        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path, encoding="utf-8") as f:
                data = json.load(f)
            self._t2u: Dict[str, str] = data.get("transliteration_to_unicode", {})
            self._u2t: Dict[str, str] = data.get("unicode_to_transliteration", {})
        else:
            self._t2u = {}
            self._u2t = {}

        # Build reverse map if not provided
        if not self._u2t and self._t2u:
            self._u2t = {}
            for k, v in self._t2u.items():
                if v not in self._u2t:
                    self._u2t[v] = k

    @property
    def num_signs(self) -> int:
        return len(self._t2u)

    def to_cuneiform(self, transliteration: str) -> str:
        """Convert Akkadian transliteration to cuneiform Unicode signs.

        Args:
            transliteration: e.g. "šar-ru-um LUGAL dan-nu"

        Returns:
            Cuneiform Unicode string, e.g. "𒊬𒊒𒌝 𒈗 𒁕𒀭𒉡"
        """
        if not transliteration or not transliteration.strip():
            return ""

        # Strip determinatives: {d}, {m}, {f}, {URU}, {KUR}, etc.
        cleaned = re.sub(r"\{[^}]+\}", "", transliteration)

        words = cleaned.strip().split()
        result_words = []

        for word in words:
            # Handle special tokens
            if word in ("<gap>", "<big_gap>", "..."):
                result_words.append("…")
                continue

            # Split by hyphens (syllable boundaries)
            syllables = re.split(r"[-.]", word)
            signs = []
            for syl in syllables:
                if not syl:
                    continue
                sign = self._lookup_sign(syl)
                signs.append(sign)

            result_words.append("".join(signs))

        return " ".join(result_words)

    def _lookup_sign(self, syllable: str) -> str:
        """Look up a single syllable/logogram in the sign table."""
        # Try exact match (case-sensitive for logograms like LUGAL)
        if syllable in self._t2u:
            return self._t2u[syllable]

        # Try lowercase
        lower = syllable.lower()
        if lower in self._t2u:
            return self._t2u[lower]

        # Try stripping subscript numbers (e.g., u₂ → u)
        stripped = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]", "", lower)
        if stripped and stripped in self._t2u:
            return self._t2u[stripped]

        # Try ASCII subscript notation (e.g., u2 → u)
        stripped_ascii = re.sub(r"\d+$", "", lower)
        if stripped_ascii and stripped_ascii in self._t2u:
            return self._t2u[stripped_ascii]

        # Not found - return original in brackets
        return f"[{syllable}]"

    def from_cuneiform(self, cuneiform: str) -> str:
        """Convert cuneiform Unicode signs back to transliteration (best-effort).

        Args:
            cuneiform: Unicode cuneiform string

        Returns:
            Transliteration string (approximate)
        """
        result = []
        for char in cuneiform:
            if char in self._u2t:
                result.append(self._u2t[char])
            elif char == " ":
                result.append(" ")
            elif char == "…":
                result.append("...")
            else:
                # Check if it's in the cuneiform Unicode range
                cp = ord(char)
                if 0x12000 <= cp <= 0x1254F or 0x12400 <= cp <= 0x1247F:
                    result.append(f"[U+{cp:05X}]")
                else:
                    result.append(char)
        return "-".join(result) if result else ""

    def get_sign_info(self, syllable: str) -> Optional[Dict]:
        """Get information about a cuneiform sign."""
        sign = self._lookup_sign(syllable)
        if sign.startswith("["):
            return None
        return {
            "transliteration": syllable,
            "cuneiform": sign,
            "unicode": f"U+{ord(sign[0]):05X}" if sign else None,
            "name": f"CUNEIFORM SIGN {syllable.upper()}",
        }
