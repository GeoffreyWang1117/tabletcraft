"""Render cuneiform text as clay tablet images (SVG/PNG)."""

import math
import textwrap
from pathlib import Path
from typing import Optional, Tuple

try:
    import svgwrite
except ImportError:
    svgwrite = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None


# Clay tablet color palette
CLAY_BG = "#C4A882"  # Warm clay
CLAY_DARK = "#8B7355"  # Aged clay edges
CLAY_LIGHT = "#D4C4A8"  # Highlight
TEXT_COLOR = "#3D2B1F"  # Dark brown ink/impression
BORDER_COLOR = "#6B5B45"


class TabletRenderer:
    """Render cuneiform text as a clay tablet image.

    Generates SVG or PNG images that look like ancient Mesopotamian clay tablets
    with cuneiform inscriptions.
    """

    def __init__(
        self,
        width: int = 600,
        height: Optional[int] = None,
        chars_per_line: int = 20,
        font_size: int = 28,
        padding: int = 40,
        style: str = "classic",
    ):
        self.width = width
        self.chars_per_line = chars_per_line
        self.font_size = font_size
        self.padding = padding
        self.style = style
        self._height = height

    def render_svg(self, cuneiform_text: str, title: Optional[str] = None) -> str:
        """Render cuneiform text as an SVG string.

        Args:
            cuneiform_text: Unicode cuneiform string
            title: Optional title/label below the tablet

        Returns:
            SVG XML string
        """
        if svgwrite is None:
            raise ImportError("svgwrite is required: pip install svgwrite")

        lines = self._wrap_text(cuneiform_text)
        line_height = self.font_size * 1.6
        text_height = len(lines) * line_height
        height = self._height or int(text_height + self.padding * 3 + (40 if title else 0))

        dwg = svgwrite.Drawing(size=(self.width, height))

        # Tablet background with rounded rectangle
        tablet_x = self.padding // 2
        tablet_y = self.padding // 2
        tablet_w = self.width - self.padding
        tablet_h = height - self.padding - (30 if title else 0)

        # Outer shadow
        dwg.add(dwg.rect(
            insert=(tablet_x + 3, tablet_y + 3),
            size=(tablet_w, tablet_h),
            rx=15, ry=15,
            fill="#7A6B55", opacity=0.4,
        ))

        # Main tablet body
        dwg.add(dwg.rect(
            insert=(tablet_x, tablet_y),
            size=(tablet_w, tablet_h),
            rx=12, ry=12,
            fill=CLAY_BG,
            stroke=BORDER_COLOR,
            stroke_width=2,
        ))

        # Inner highlight (subtle clay texture effect)
        dwg.add(dwg.rect(
            insert=(tablet_x + 6, tablet_y + 6),
            size=(tablet_w - 12, tablet_h - 12),
            rx=8, ry=8,
            fill="none",
            stroke=CLAY_LIGHT,
            stroke_width=1,
            opacity=0.6,
        ))

        # Horizontal ruling lines (like real tablets)
        for i in range(len(lines)):
            y = self.padding + i * line_height + line_height * 0.85
            dwg.add(dwg.line(
                start=(self.padding, y),
                end=(self.width - self.padding, y),
                stroke=CLAY_DARK,
                stroke_width=0.5,
                opacity=0.3,
            ))

        # Cuneiform text
        for i, line in enumerate(lines):
            y = self.padding + i * line_height + line_height * 0.7
            dwg.add(dwg.text(
                line,
                insert=(self.padding + 5, y),
                fill=TEXT_COLOR,
                font_size=self.font_size,
                font_family="Noto Sans Cuneiform, Akkadian, serif",
            ))

        # Title/label below tablet
        if title:
            dwg.add(dwg.text(
                title,
                insert=(self.width // 2, height - 8),
                fill="#555",
                font_size=12,
                font_family="Georgia, serif",
                text_anchor="middle",
                font_style="italic",
            ))

        return dwg.tostring()

    def render_png(
        self,
        cuneiform_text: str,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Optional[bytes]:
        """Render cuneiform text as a PNG image using Pillow.

        Args:
            cuneiform_text: Unicode cuneiform string
            title: Optional title below tablet
            output_path: If set, save to this path

        Returns:
            PNG bytes if output_path is None, else None (saves to file)
        """
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")

        lines = self._wrap_text(cuneiform_text)
        line_height = int(self.font_size * 1.8)
        text_height = len(lines) * line_height
        height = self._height or int(text_height + self.padding * 3 + (40 if title else 0))

        img = Image.new("RGB", (self.width, height), color=CLAY_BG)
        draw = ImageDraw.Draw(img)

        # Try to load cuneiform font, fall back to default
        try:
            font = ImageFont.truetype("NotoSansCuneiform-Regular.ttf", self.font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCuneiform-Regular.ttf", self.font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Draw tablet border
        border = self.padding // 2
        draw.rounded_rectangle(
            [border, border, self.width - border, height - border - (30 if title else 0)],
            radius=12,
            outline=BORDER_COLOR,
            width=2,
        )

        # Draw ruling lines
        for i in range(len(lines)):
            y = self.padding + i * line_height + line_height
            draw.line(
                [(self.padding, y), (self.width - self.padding, y)],
                fill=CLAY_DARK,
                width=1,
            )

        # Draw cuneiform text
        for i, line in enumerate(lines):
            y = self.padding + i * line_height + 5
            draw.text((self.padding + 5, y), line, fill=TEXT_COLOR, font=font)

        # Draw title
        if title:
            try:
                title_font = ImageFont.truetype("Georgia", 12)
            except (OSError, IOError):
                title_font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), title, font=title_font)
            tw = bbox[2] - bbox[0]
            draw.text(
                ((self.width - tw) // 2, height - 25),
                title,
                fill="#555555",
                font=title_font,
            )

        if output_path:
            img.save(output_path)
            return None

        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _wrap_text(self, text: str) -> list:
        """Wrap cuneiform text into lines."""
        if not text:
            return [""]

        # Split by spaces (word boundaries)
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 > self.chars_per_line:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = f"{current_line} {word}" if current_line else word

        if current_line:
            lines.append(current_line)

        return lines or [""]
