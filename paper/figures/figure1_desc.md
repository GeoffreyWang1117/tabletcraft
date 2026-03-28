# TabletCraft Pipeline Architecture

TabletCraft is an end-to-end pipeline that transforms modern English text into cuneiform clay tablet images through three stages:

## Stage 1: Neural Machine Translation (ByT5-base)
- Input: English text, e.g., "The mighty king rules the land"
- Model: Fine-tuned ByT5-base (581M params), trained on 116K bidirectional parallel sentences
- Output: Akkadian transliteration, e.g., "šar-ru dan-nu ma-a-tam i-be-el"
- Supports both English→Akkadian and Akkadian→English directions

## Stage 2: Cuneiform Sign Conversion
- Input: Akkadian transliteration tokens
- Lookup table: 14,240 transliteration-to-cuneiform mappings (95.3% coverage)
- Each syllable/logogram maps to Unicode cuneiform characters (U+12000-U+1254F)
- Example: LUGAL → 𒈗 (king), dan → 𒆗, nu → 𒉡

## Stage 3: Clay Tablet Rendering
- Input: Cuneiform Unicode text
- Output: SVG/PNG image styled like an authentic Mesopotamian clay tablet
- Features: clay-colored background, horizontal ruling lines, Noto Sans Cuneiform font
- No GPU required, <10ms per tablet

The pipeline bridges a 4,000-year cultural gap. This is a system architecture diagram showing the three-stage pipeline flowing left to right, with example data at each stage. Style: clean academic diagram suitable for ACL/NeurIPS paper, horizontal flow, rounded boxes for each stage, arrows connecting stages, sample inputs/outputs shown.
