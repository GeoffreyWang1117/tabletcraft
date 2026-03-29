# CuneiScribe

Cuneiform clay tablet toolkit with confidence gating. Translates English↔Akkadian, converts to cuneiform Unicode, renders SVG/PNG tablets.

## Architecture

4-layer decoupled design:
```
cuneiscribe/
├── pipeline/          # Input classifier + output validator (confidence gating)
│   ├── classifier.py  # Classify: name/short/historical/modern/anomalous
│   └── validator.py   # Validate transliteration before rendering
├── models/            # ByT5 bidirectional translator
│   └── translator.py  # Ak→En and En→Ak
├── knowledge/         # Sign tables + cuneiform converter
│   └── cuneiform.py   # 14,240 transliteration→Unicode mappings
├── interfaces/        # CLI, Gradio demo, SVG renderer
│   ├── cli.py         # Entry point: cuneiscribe cuneiform/render/craft/classify/info
│   ├── demo.py        # Gradio web app
│   └── renderer.py    # SVG/PNG clay tablet generation
└── core.py            # Orchestrator: classify → translate → validate → render/fallback
```

## Key files

- `knowledge/sign_tables/transliteration_mapping.json` — 14,240 sign mappings (DO NOT hand-edit)
- `knowledge/dictionaries/` — Akkadian dictionaries (6.6K + 17K lemmas)
- `models/byt5-base-akkadian/` — Model weights (BLEU 49.1, not in git, use git-lfs)
- `ROADMAP.md` — Engineering roadmap with phased milestones

## Commands

```bash
# CLI
python -m cuneiscribe.interfaces.cli cuneiform "LUGAL dan-nu"
python -m cuneiscribe.interfaces.cli classify "The king rules"
python -m cuneiscribe.interfaces.cli render "šar kiš-ša-ti" -o tablet.svg
python -m cuneiscribe.interfaces.cli craft "The king rules" --model models/byt5-base-akkadian --json

# Tests
python -m pytest tests/ -v

# Web demo
python -m cuneiscribe.interfaces.demo --model models/byt5-base-akkadian
```

## Design principles

1. **Reliability > Capability** — The system knows when it's unreliable and degrades gracefully
2. **Show intermediate layers** — Always expose transliteration, not just the final tablet
3. **Caveat by default** — All outputs labeled as "approximate, machine-generated"
4. **Layers decoupled** — Swapping model, updating sign tables, or adding a dialect should not break interfaces

## Do NOT

- Merge sign table changes without running `python -m pytest tests/test_pipeline.py`
- Remove the confidence gating pipeline (it's the core reliability mechanism)
- Claim outputs are "authentic ancient text" in any user-facing string
- Commit model.safetensors to git (use git-lfs or HF Hub)
