"""Tests for batch processing and metrics."""

import json
import tempfile
from pathlib import Path

from tabletcraft.pipeline.batch import process_batch, read_input, write_output
from tabletcraft.pipeline.metrics import SessionMetrics, Timer


class TestBatchProcessing:
    def test_cuneiform_only_batch(self):
        inputs = ["LUGAL dan-nu", "a-na", "šar-ru"]
        results = process_batch(inputs, translator=None, direction="ak2en")
        assert len(results) == 3
        for r in results:
            assert r["cuneiform"] != ""

    def test_en2ak_no_model(self):
        results = process_batch(["The king"], translator=None, direction="en2ak")
        assert results[0]["status"] == "skipped_no_model"

    def test_anomalous_rejected(self):
        results = process_batch(["<script>alert(1)</script>"], direction="en2ak")
        assert results[0]["status"] == "rejected"

    def test_read_write_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"text": "LUGAL dan-nu"}\n')
            f.write('{"text": "a-na"}\n')
            path = f.name
        texts = read_input(path)
        assert len(texts) == 2
        assert texts[0] == "LUGAL dan-nu"

        results = process_batch(texts, direction="ak2en")
        out_path = path.replace(".jsonl", "_out.jsonl")
        write_output(results, out_path)
        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        Path(path).unlink()
        Path(out_path).unlink()

    def test_read_write_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("text\n")
            f.write("LUGAL\n")
            f.write("a-na\n")
            path = f.name
        texts = read_input(path)
        assert len(texts) == 2

        results = process_batch(texts, direction="ak2en")
        out_path = path.replace(".csv", "_out.csv")
        write_output(results, out_path)
        Path(path).unlink()
        Path(out_path).unlink()

    def test_read_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("LUGAL\na-na\nšar-ru\n")
            path = f.name
        texts = read_input(path)
        assert len(texts) == 3
        Path(path).unlink()


class TestMetrics:
    def test_session_metrics(self):
        m = SessionMetrics()
        m.record("name", "render", 0.9, signs_total=5, signs_unknown=0, latency_ms=10)
        m.record("modern", "fallback", 0.3, signs_total=0, signs_unknown=0, latency_ms=50)
        m.record("anomalous", "fallback", 0.0)

        assert m.total_requests == 3
        assert m.fallback_rate == 2 / 3
        assert m.rejection_rate == 1 / 3
        assert m.unknown_mapping_rate == 0.0
        assert m.avg_validation_score > 0

        s = m.summary()
        assert "fallback_rate" in s
        assert s["total_requests"] == 3

    def test_timer(self):
        with Timer() as t:
            _ = sum(range(10000))
        assert t.elapsed_ms > 0
