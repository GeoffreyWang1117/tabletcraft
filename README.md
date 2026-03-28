# TabletCraft

**Turn any text into cuneiform clay tablets.**

Write like an ancient Mesopotamian scribe! TabletCraft translates text to Akkadian cuneiform and renders it as a clay tablet — bridging 4,000 years of human writing.

```
English: "The mighty king rules the land"
    -> Akkadian: šar-ru dan-nu ma-a-tam i-be-el
    -> Cuneiform: 𒊬𒊒 𒆗𒉡 𒈠𒀀𒌓 𒄿𒁁𒂖
    -> [Clay Tablet Image]
```

## Quick Start

```bash
pip install tabletcraft
```

### CLI

```bash
# Convert transliteration to cuneiform signs
tabletcraft cuneiform "LUGAL dan-nu LUGAL KUR aš-šur"
# Output: 𒈗 𒆗𒉡 𒈗 𒆳 𒀸𒋩

# Render as a clay tablet SVG
tabletcraft render "LUGAL dan-nu LUGAL KUR aš-šur" -o tablet.svg

# Look up a cuneiform sign
tabletcraft info LUGAL
# Sign: LUGAL
# Cuneiform: 𒈗
# Unicode: U+12217

# Full pipeline: English -> clay tablet (requires model)
tabletcraft craft "The king of Assyria" -o tablet.svg --model path/to/model
```

### Python API

```python
from tabletcraft import CuneiformConverter, TabletRenderer

# Convert transliteration -> cuneiform
conv = CuneiformConverter()
cuneiform = conv.to_cuneiform("LUGAL dan-nu LUGAL KUR aš-šur")
print(cuneiform)  # 𒈗 𒆗𒉡 𒈗 𒆳 𒀸𒋩

# Render as clay tablet
renderer = TabletRenderer()
svg = renderer.render_svg(cuneiform, title="King of Assyria")
with open("tablet.svg", "w") as f:
    f.write(svg)

# Full pipeline with translation (requires model)
from tabletcraft import TabletCraft
tc = TabletCraft(model_path="path/to/model")
tc.craft("The mighty king", output_path="tablet.svg")
```

### Web Demo

```bash
pip install tabletcraft[serve]
python -m tabletcraft.demo --share
```

## Features

| Feature | Description |
|---------|-------------|
| Cuneiform Converter | 14,240+ transliteration-to-Unicode mappings |
| Clay Tablet Renderer | SVG/PNG with authentic Mesopotamian styling |
| Bidirectional NMT | English -> Akkadian and Akkadian -> English |
| CLI Tool | One-command cuneiform conversion and rendering |
| Web Demo | Gradio interface, deployable to HF Spaces |
| Dictionary | 6,634 Akkadian lemmas with meanings and forms |

## How It Works

```
     English Text
          |
    [ByT5 NMT Model]     <- Fine-tuned on 50K+ parallel sentences
          |
  Akkadian Transliteration
          |
  [Sign Lookup Table]     <- 14,240 mappings from academic sources
          |
   Cuneiform Unicode
          |
   [SVG Renderer]         <- Clay tablet styling with ruling lines
          |
    Tablet Image
```

## About

TabletCraft was built during the [Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) — a Kaggle competition to translate 4,000-year-old Old Assyrian business records. It packages the competition's NMT models and cuneiform data into a reusable toolkit.

The cuneiform sign mappings cover the Unicode Cuneiform block (U+12000-U+1254F) and are sourced from ORACC, CDLI, and the Akkademia project.

## License

Apache 2.0. Academic data used under respective project licenses.

## Citation

```bibtex
@software{tabletcraft2026,
  title={TabletCraft: Turning Text into Cuneiform Clay Tablets},
  author={Wang, Geoffrey},
  year={2026},
  url={https://github.com/geoffreywang1117/tabletcraft}
}
```
