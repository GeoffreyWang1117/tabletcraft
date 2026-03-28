# TabletCraft

**Bridge a 4,000-year cultural gap — read and write in cuneiform.**

TabletCraft lets you interact with humanity's oldest writing system. Translate between English and Akkadian, convert to cuneiform Unicode signs, and render clay tablet images — with built-in confidence gating that tells you when results are unreliable.

```
$ tabletcraft cuneiform "LUGAL dan-nu LUGAL KUR aš-šur"
𒈗 𒆗𒉡 𒈗 𒆳 𒀸𒋩

$ tabletcraft classify "The computer sends an email"
Type:       modern
Confidence: 0.70
Warnings:
  - Contains 2 modern concept(s) with no direct Akkadian equivalent
```

> All outputs are machine-generated approximations. Consult Assyriological expertise for research or public-facing use.

## Install

```bash
pip install tabletcraft
```

## Quick Start

### CLI

```bash
# Convert transliteration to cuneiform
tabletcraft cuneiform "LUGAL dan-nu"        # → 𒈗 𒆗𒉡

# Classify input before processing
tabletcraft classify "The king rules"       # → short, 0.85, experience mode

# Render as clay tablet
tabletcraft render "šar kiš-ša-ti" -o tablet.svg

# Full pipeline with confidence gating (requires model)
tabletcraft craft "The king rules" --model models/byt5-base-akkadian --json

# Look up a sign
tabletcraft info LUGAL                      # → 𒈗, U+12217
```

### Python API

```python
from tabletcraft import TabletCraft, classify

# Check input first
result = classify("I love pizza")
print(result.input_type)   # "short"
print(result.warnings)     # []

# Full pipeline with gating
tc = TabletCraft(model_path="models/byt5-base-akkadian")
result = tc.craft("The mighty king")

print(result.akkadian)     # Transliteration
print(result.cuneiform)    # Unicode cuneiform
print(result.confidence)   # 0.0-1.0
print(result.suggestion)   # "render" / "render_with_caveat" / "fallback"
print(result.warnings)     # List of caveats

# No model needed for transliteration → cuneiform
result = tc.transliterate_and_render("LUGAL dan-nu", output_path="tablet.svg")
```

### Web Demo

```bash
pip install tabletcraft[serve]
python -m tabletcraft.interfaces.demo --model models/byt5-base-akkadian --share
```

## How It Works

```
User Input
    │
    ▼
Input Classifier ──→ anomalous? → REJECT
    │
    ▼
ByT5 Translation (En→Ak)
    │
    ▼
Output Validator ──→ unreliable? → FALLBACK (transliteration only)
    │
    ▼
Cuneiform Converter (14,240 mappings)
    │
    ▼
Tablet Renderer (SVG/PNG)
    │
    ▼
Result + Confidence + Warnings
```

The confidence gating pipeline ensures the system **never confidently renders wrong cuneiform**. When output quality is uncertain, it degrades gracefully to transliteration-only with a warning.

## Features

| Feature | Description |
|---------|-------------|
| Confidence Gating | Input classification + output validation before rendering |
| Cuneiform Converter | 14,240 transliteration→Unicode mappings, 95.3% coverage |
| Clay Tablet Renderer | SVG/PNG with authentic Mesopotamian styling, <10ms |
| Bidirectional NMT | English→Akkadian and Akkadian→English (ByT5-base, 49.1 BLEU) |
| CLI | `cuneiform`, `render`, `craft`, `classify`, `info` commands |
| Web Demo | Gradio interface with 4-panel display |

## Architecture

```
tabletcraft/
├── pipeline/      ← Confidence gating (classifier + validator)
├── models/        ← ByT5 bidirectional translator
├── knowledge/     ← Sign tables + cuneiform converter
└── interfaces/    ← CLI, web demo, SVG renderer
```

Four decoupled layers. Swap the model, update sign tables, or add a dialect without breaking interfaces.

## Limitations

- English→Akkadian produces **approximate modern transliterations**, not authentic ancient text
- Trained on Neo-Assyrian/Old Babylonian data; other Akkadian dialects (e.g., Old Assyrian commercial texts) may have substantially lower quality
- Even with 14,240 sign mappings and a 17K-lemma dictionary, **domain-specific data matters more than model size**
- Currently English-only

See [ROADMAP.md](ROADMAP.md) for the engineering roadmap.

## Citation

```bibtex
@inproceedings{tabletcraft2026,
  title={TabletCraft: Bridging a 4,000-Year Cultural Gap with Bidirectional Akkadian NMT and Cuneiform Rendering},
  author={Wang, Geoffrey},
  booktitle={Proceedings of the 4th Workshop on Cross-Cultural Considerations in NLP (C3NLP)},
  year={2026}
}
```

## License

Apache 2.0
