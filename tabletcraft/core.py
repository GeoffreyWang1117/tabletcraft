"""TabletCraft: end-to-end pipeline with confidence gating.

The core orchestrator routes inputs through classification → translation →
validation → conversion → rendering, with graceful degradation at each step.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from tabletcraft.pipeline.classifier import classify, InputType
from tabletcraft.pipeline.validator import validate
from tabletcraft.models.translator import AkkadianTranslator
from tabletcraft.knowledge.cuneiform import CuneiformConverter
from tabletcraft.interfaces.renderer import TabletRenderer


@dataclass
class CraftResult:
    """Full result with intermediate layers and confidence info."""
    input_text: str
    input_type: str
    akkadian: str
    cuneiform: str
    image: Optional[str]  # SVG string or PNG bytes
    confidence: float
    warnings: List[str] = field(default_factory=list)
    mode: str = "experience"  # "experience", "educational", "research"
    suggestion: str = "render"  # "render", "render_with_caveat", "fallback"


class TabletCraft:
    """End-to-end pipeline with confidence gating.

    Usage:
        tc = TabletCraft(model_path="models/byt5-base-akkadian")
        result = tc.craft("The king rules the land")

        # result.confidence > 0.7 → safe to display
        # result.warnings → list of caveats
        # result.suggestion → "render" / "render_with_caveat" / "fallback"
        # result.image → SVG string (if rendered)
        # result.akkadian → transliteration (always present)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
    ):
        self.translator = AkkadianTranslator(model_path) if model_path else None
        self.converter = CuneiformConverter(mapping_path)
        self.renderer = TabletRenderer()

    def craft(
        self,
        english_text: str,
        format: str = "svg",
        force_render: bool = False,
        output_path: Optional[str] = None,
    ) -> CraftResult:
        """Full pipeline with confidence gating.

        Args:
            english_text: Input text
            format: "svg" or "png"
            force_render: Skip validation and render regardless (research mode)
            output_path: Save image to file

        Returns:
            CraftResult with all intermediate layers and confidence info
        """
        # Step 1: Classify input
        classification = classify(english_text)
        warnings = list(classification.warnings)

        # Step 2: Handle anomalous input
        if classification.input_type == InputType.ANOMALOUS:
            return CraftResult(
                input_text=english_text, input_type="anomalous",
                akkadian="", cuneiform="", image=None, confidence=0.0,
                warnings=warnings + ["Input rejected: " + "; ".join(classification.warnings)],
                mode=classification.suggested_mode, suggestion="fallback",
            )

        # Step 3: Translate
        if self.translator is None:
            return CraftResult(
                input_text=english_text, input_type=classification.input_type.value,
                akkadian="", cuneiform="", image=None, confidence=0.0,
                warnings=["No translation model loaded"],
                mode=classification.suggested_mode, suggestion="fallback",
            )

        akkadian = self.translator.to_akkadian(english_text)

        # Step 4: Validate transliteration
        validation = validate(akkadian, english_text)
        warnings.extend(validation.issues)

        # Step 5: Confidence gating — fallback if unreliable
        if not force_render and validation.suggestion == "fallback":
            return CraftResult(
                input_text=english_text, input_type=classification.input_type.value,
                akkadian=akkadian, cuneiform="", image=None,
                confidence=validation.score,
                warnings=warnings + ["Output did not pass validation; showing transliteration only"],
                mode=classification.suggested_mode, suggestion="fallback",
            )

        # Step 6: Convert to cuneiform
        cuneiform = self.converter.to_cuneiform(akkadian)

        # Step 7: Render tablet
        title = english_text[:80]
        if validation.suggestion == "render_with_caveat":
            title = f"[approximate] {title}"

        if format == "svg":
            image = self.renderer.render_svg(cuneiform, title=title)
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(image)
        else:
            image = self.renderer.render_png(cuneiform, title=title, output_path=output_path)

        warnings.append("Machine-generated Akkadian — approximate rendering, not authentic ancient text")

        return CraftResult(
            input_text=english_text, input_type=classification.input_type.value,
            akkadian=akkadian, cuneiform=cuneiform, image=image,
            confidence=validation.score, warnings=warnings,
            mode=classification.suggested_mode, suggestion=validation.suggestion,
        )

    def transliterate_and_render(
        self, transliteration: str, format: str = "svg",
        title: Optional[str] = None, output_path: Optional[str] = None,
    ) -> CraftResult:
        """Convert known transliteration → cuneiform → tablet (no NMT, no gating)."""
        cuneiform = self.converter.to_cuneiform(transliteration)
        t = title or transliteration[:80]
        if format == "svg":
            image = self.renderer.render_svg(cuneiform, title=t)
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(image)
        else:
            image = self.renderer.render_png(cuneiform, title=t, output_path=output_path)
        return CraftResult(
            input_text=transliteration, input_type="transliteration",
            akkadian=transliteration, cuneiform=cuneiform, image=image,
            confidence=1.0, warnings=[], mode="experience", suggestion="render",
        )
