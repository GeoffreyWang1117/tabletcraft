"""CuneiScribe - Turn any text into cuneiform clay tablets, reliably."""

__version__ = "0.2.0"

from cuneiscribe.core import CuneiScribe, CraftResult
from cuneiscribe.models.translator import AkkadianTranslator
from cuneiscribe.knowledge.cuneiform import CuneiformConverter
from cuneiscribe.interfaces.renderer import TabletRenderer
from cuneiscribe.pipeline.classifier import classify, InputType
from cuneiscribe.pipeline.validator import validate

__all__ = [
    "CuneiScribe", "CraftResult",
    "AkkadianTranslator", "CuneiformConverter", "TabletRenderer",
    "classify", "InputType", "validate",
]
