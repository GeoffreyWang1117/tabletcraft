"""Batch processing: CSV/JSONL input → structured output with logs."""

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

from cuneiscribe.pipeline.classifier import classify
from cuneiscribe.pipeline.validator import validate
from cuneiscribe.knowledge.cuneiform import CuneiformConverter

logger = logging.getLogger("cuneiscribe.batch")


def process_batch(
    inputs: List[str],
    translator=None,
    converter: Optional[CuneiformConverter] = None,
    direction: str = "en2ak",
) -> List[dict]:
    """Process a batch of texts with full pipeline logging.

    Args:
        inputs: List of input texts
        translator: AkkadianTranslator instance (None = cuneiform-only mode)
        converter: CuneiformConverter instance
        direction: "en2ak" (English→Akkadian→cuneiform) or "ak2en" (Akkadian→English)

    Returns:
        List of result dicts with all intermediate layers
    """
    if converter is None:
        converter = CuneiformConverter()

    results = []
    stats = {"total": len(inputs), "rendered": 0, "fallback": 0, "rejected": 0}

    for i, text in enumerate(inputs):
        row = {"index": i, "input": text}

        if direction == "en2ak":
            # Classify
            cls = classify(text)
            row["input_type"] = cls.input_type.value
            row["mode"] = cls.suggested_mode
            row["classify_confidence"] = cls.confidence
            row["classify_warnings"] = cls.warnings

            if cls.input_type.value == "anomalous":
                row["status"] = "rejected"
                row["akkadian"] = ""
                row["cuneiform"] = ""
                stats["rejected"] += 1
                results.append(row)
                continue

            # Translate
            if translator is not None:
                akkadian = translator.to_akkadian(text)
            else:
                row["status"] = "skipped_no_model"
                row["akkadian"] = ""
                row["cuneiform"] = ""
                results.append(row)
                continue

            row["akkadian"] = akkadian

            # Validate
            val = validate(akkadian, text)
            row["valid"] = val.valid
            row["validate_score"] = val.score
            row["validate_issues"] = val.issues
            row["suggestion"] = val.suggestion

            if val.suggestion == "fallback":
                row["status"] = "fallback"
                row["cuneiform"] = ""
                stats["fallback"] += 1
            else:
                row["cuneiform"] = converter.to_cuneiform(akkadian)
                row["status"] = val.suggestion  # "render" or "render_with_caveat"
                stats["rendered"] += 1

        elif direction == "ak2en":
            if translator is not None:
                row["english"] = translator.to_english(text)
                row["status"] = "translated"
            else:
                row["english"] = ""
                row["status"] = "skipped_no_model"
            row["cuneiform"] = converter.to_cuneiform(text)
            stats["rendered"] += 1

        results.append(row)

    logger.info(
        f"Batch complete: {stats['total']} items, "
        f"{stats['rendered']} rendered, {stats['fallback']} fallback, "
        f"{stats['rejected']} rejected"
    )
    return results


def read_input(path: str) -> List[str]:
    """Read input from CSV or JSONL file."""
    p = Path(path)
    texts = []
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                texts.append(obj.get("text", obj.get("input", "")))
    elif p.suffix == ".csv":
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col = None
            for name in ["text", "input", "transliteration", "english"]:
                if name in reader.fieldnames:
                    col = name
                    break
            if col is None:
                col = reader.fieldnames[0]
            for row in reader:
                texts.append(row[col])
    else:
        # Plain text, one per line
        with open(p, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    return texts


def write_output(results: List[dict], path: str):
    """Write results to CSV or JSONL."""
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif p.suffix == ".csv":
        if not results:
            return
        keys = results[0].keys()
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                # Flatten lists to strings for CSV
                flat = {}
                for k, v in r.items():
                    flat[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                writer.writerow(flat)
    else:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Output written to {path} ({len(results)} rows)")
