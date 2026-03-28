"""Observability: structured logging and runtime metrics.

Tracks key reliability indicators:
- Input type distribution
- Unknown mapping rate
- Fallback rate
- Validation score distribution
"""

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger("tabletcraft.metrics")


@dataclass
class SessionMetrics:
    """Accumulates metrics for a session / batch / deployment period."""
    total_requests: int = 0
    input_types: Counter = field(default_factory=Counter)
    suggestions: Counter = field(default_factory=Counter)  # render / render_with_caveat / fallback
    rejected: int = 0
    total_signs_converted: int = 0
    unknown_signs: int = 0
    validation_scores: list = field(default_factory=list)
    latencies_ms: list = field(default_factory=list)

    def record(self, input_type: str, suggestion: str, validation_score: float,
               signs_total: int = 0, signs_unknown: int = 0, latency_ms: float = 0):
        self.total_requests += 1
        self.input_types[input_type] += 1
        self.suggestions[suggestion] += 1
        if input_type == "anomalous":
            self.rejected += 1
        self.validation_scores.append(validation_score)
        self.total_signs_converted += signs_total
        self.unknown_signs += signs_unknown
        if latency_ms > 0:
            self.latencies_ms.append(latency_ms)

    @property
    def fallback_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.suggestions.get("fallback", 0) / self.total_requests

    @property
    def rejection_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.rejected / self.total_requests

    @property
    def unknown_mapping_rate(self) -> float:
        if self.total_signs_converted == 0:
            return 0.0
        return self.unknown_signs / self.total_signs_converted

    @property
    def avg_validation_score(self) -> float:
        if not self.validation_scores:
            return 0.0
        return sum(self.validation_scores) / len(self.validation_scores)

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "input_types": dict(self.input_types),
            "suggestions": dict(self.suggestions),
            "fallback_rate": round(self.fallback_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "unknown_mapping_rate": round(self.unknown_mapping_rate, 4),
            "avg_validation_score": round(self.avg_validation_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }

    def log_summary(self):
        s = self.summary()
        logger.info(f"Session metrics: {json.dumps(s)}")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)


# Global session metrics (singleton for simple use)
_session = SessionMetrics()


def get_session() -> SessionMetrics:
    return _session


def reset_session():
    global _session
    _session = SessionMetrics()


class Timer:
    """Context manager for timing operations."""
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
