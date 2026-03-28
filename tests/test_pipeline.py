"""Regression tests for TabletCraft pipeline.

These tests verify the confidence gating pipeline works correctly
WITHOUT requiring a loaded model (tests classifier + validator + converter).
"""

import pytest
from tabletcraft.pipeline.classifier import classify, InputType
from tabletcraft.pipeline.validator import validate
from tabletcraft.knowledge.cuneiform import CuneiformConverter


# ============================================================
# Input Classifier Tests
# ============================================================

class TestClassifier:
    def test_name(self):
        r = classify("John")
        assert r.input_type == InputType.NAME

    def test_name_multi_word(self):
        r = classify("Marcus Aurelius")
        assert r.input_type == InputType.NAME

    def test_short_phrase(self):
        r = classify("The king rules the land")
        assert r.input_type == InputType.SHORT_PHRASE

    def test_modern_concept_warning(self):
        r = classify("The internet connects all computers worldwide")
        assert r.input_type == InputType.MODERN_HEAVY
        assert any("modern" in w.lower() for w in r.warnings)

    def test_historical_text(self):
        r = classify("The king made an offering of silver and gold to the temple of the god")
        assert r.input_type in (InputType.HISTORICAL, InputType.SHORT_PHRASE)

    def test_empty_input(self):
        r = classify("")
        assert r.input_type == InputType.ANOMALOUS

    def test_html_injection(self):
        r = classify("<script>alert('xss')</script>")
        assert r.input_type == InputType.ANOMALOUS

    def test_very_long(self):
        r = classify("word " * 500)
        assert r.input_type == InputType.ANOMALOUS

    def test_repeated_chars(self):
        r = classify("aaaaaaaaaaaaaaaaaaaaa")
        assert r.input_type == InputType.ANOMALOUS


# ============================================================
# Validator Tests
# ============================================================

class TestValidator:
    def test_valid_transliteration(self):
        r = validate("šar-ru dan-nu", "The mighty king")
        assert r.valid
        assert r.score >= 0.7
        assert r.suggestion == "render"

    def test_empty_output(self):
        r = validate("", "Hello world")
        assert not r.valid
        assert r.suggestion == "fallback"

    def test_task_prefix_leak(self):
        r = validate("translate Akkadian to English: šar-ru", "king")
        assert "task prefix" in " ".join(r.issues).lower()
        assert r.score < 0.7

    def test_english_leak(self):
        r = validate("the king of the land was in the palace", "some input")
        assert any("english" in i.lower() for i in r.issues)

    def test_extreme_expansion(self):
        r = validate("a " * 100, "Hi")
        assert r.score < 0.7

    def test_normal_akkadian(self):
        r = validate("LUGAL dan-nu LUGAL KUR aš-šur", "Strong king, king of Assyria")
        assert r.valid
        assert r.suggestion == "render"


# ============================================================
# Cuneiform Converter Tests
# ============================================================

class TestCuneiformConverter:
    @pytest.fixture
    def conv(self):
        return CuneiformConverter()

    def test_basic_logogram(self, conv):
        if conv.num_signs == 0:
            pytest.skip("No sign table loaded")
        result = conv.to_cuneiform("LUGAL")
        assert result == "𒈗"

    def test_syllabic(self, conv):
        if conv.num_signs == 0:
            pytest.skip("No sign table loaded")
        result = conv.to_cuneiform("a-na")
        assert result == "𒀀𒈾"

    def test_determinative_stripping(self, conv):
        if conv.num_signs == 0:
            pytest.skip("No sign table loaded")
        r1 = conv.to_cuneiform("{d}aš-šur")
        r2 = conv.to_cuneiform("aš-šur")
        assert r1 == r2  # Determinative should be stripped

    def test_gap_marker(self, conv):
        result = conv.to_cuneiform("...")
        assert "…" in result

    def test_empty(self, conv):
        assert conv.to_cuneiform("") == ""

    def test_sign_info(self, conv):
        if conv.num_signs == 0:
            pytest.skip("No sign table loaded")
        info = conv.get_sign_info("LUGAL")
        assert info is not None
        assert info["cuneiform"] == "𒈗"


# ============================================================
# Integration: Classifier + Validator together
# ============================================================

class TestGatingIntegration:
    def test_name_passes_all(self):
        c = classify("Alexander")
        assert c.input_type == InputType.NAME
        v = validate("a-lik-sa-an-dar", "Alexander")
        assert v.valid

    def test_modern_gets_warning(self):
        c = classify("The computer runs the algorithm")
        assert c.input_type == InputType.MODERN_HEAVY
        assert len(c.warnings) > 0

    def test_garbage_rejected(self):
        c = classify("\x00\x01\x02\x03")
        assert c.input_type == InputType.ANOMALOUS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
