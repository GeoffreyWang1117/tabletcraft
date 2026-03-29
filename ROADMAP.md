# CuneiScribe Engineering Roadmap

> Core principle: A reliable tool knows when it is unreliable — and degrades gracefully.

## Product Positioning

CuneiScribe is **not** an authoritative ancient language translator. It is a **cultural engagement tool** that makes cuneiform accessible while being transparent about its limitations.

Three operating modes:

| Mode | Audience | Behavior |
|------|----------|----------|
| **Experience** | Public, museum visitors | Names and short sentences → tablet. Cached templates, fast, caveats minimal but present. |
| **Educational** | Students, teachers | Show all intermediate layers: input → transliteration → sign mappings → tablet. Highlight uncertain mappings. |
| **Research** | Scholars, NLP researchers | Batch processing, export transliterations + mapping logs + failure reports. Default: "requires human review." |

## Architecture (4 Layers, Decoupled)

```
┌─────────────────────────────────────────────────┐
│  Interfaces: CLI / Python API / Web / REST API  │
├─────────────────────────────────────────────────┤
│  Pipeline: input classification → confidence    │
│  gating → validator → fallback → renderer       │
├─────────────────────────────────────────────────┤
│  Model Inference: ByT5 Ak↔En translation        │
├─────────────────────────────────────────────────┤
│  Knowledge: sign tables, dictionaries, rules,   │
│  dialect metadata (all versioned)               │
└─────────────────────────────────────────────────┘
```

Layers are **decoupled**: swapping a model, updating sign tables, or adding a dialect should not break the interfaces above.

## Confidence Gating (The Key Reliability Mechanism)

```
User Input
    │
    ▼
Input Classifier
    ├─ Name / short phrase → Experience mode (cache hit?)
    ├─ Historical-style text → Full pipeline + intermediate display
    ├─ Modern-concept-heavy → ⚠ Warning + degraded output
    └─ Anomalous input → Reject + explanation
    │
    ▼
Model generates transliteration
    │
    ▼
Validator checks:
    ├─ Unknown tokens?
    ├─ Low-frequency sign combinations?
    ├─ Sign mapping gaps?
    ├─ Excessive repetition?
    └─ Abnormal length expansion?
    │
    ├─ PASS → Cuneiform conversion → Renderer → Full output
    └─ FAIL → Transliteration-only + risk notice (safe fallback)
```

A tool that says "I'm not sure about this word" is 100x more reliable than one that confidently renders wrong cuneiform.

## Version Management

Every release binds **5 artifact versions**:

| Artifact | Example | Why |
|----------|---------|-----|
| Training corpus | `corpus-v1.0-116k-bidir` | Data changes affect model behavior |
| Preprocessing rules | `rules-v1.2` | Normalization affects inputs |
| Model checkpoint | `byt5-base-ckpt44000` | Model weights |
| Sign table | `signs-v1.0-14240` | Mapping changes affect rendering |
| Renderer | `render-v1.0` | Visual output changes |

If a user reports "this sentence looked different last month," we can trace which artifact changed.

## Evaluation (4 Types)

| Type | Purpose |
|------|---------|
| **Regression set** | Fixed inputs (names, short phrases, museum examples), must pass on every release |
| **Mapping integrity** | Sign table coverage ≥ 95.3%, logogram/syllabic breakdown stable |
| **Adversarial inputs** | Emoji, HTML, ultra-long, mixed-language, prompt injection — must not crash or hallucinate |
| **Human review benchmark** | Domain experts rate outputs as "acceptable for display" / "acceptable for teaching" / "not displayable" |

## Phased Roadmap

### Phase 1: Reliable Minimal Product
- [ ] Confidence gating pipeline (input classifier + validator)
- [ ] Experience mode: names and short sentences with cached templates
- [ ] Default caveats on all outputs ("approximate rendering")
- [ ] 4-panel display: input → transliteration → cuneiform → tablet + confidence
- [ ] Regression test suite (50+ fixed test cases)
- [ ] Adversarial input handling (graceful rejection)
- [ ] pip-installable with CLI working end-to-end
- [ ] Gradio demo deployable to HF Spaces

### Phase 2: Research Batch Mode
- [ ] CSV/JSONL input → batch translation + export
- [ ] Transliteration + mapping log + failure report per item
- [ ] Human-in-the-loop: expert correction → review queue → knowledge feedback
- [ ] Observability: input distribution, unknown mapping rate, fallback rate
- [ ] REST API with rate limiting

### Phase 3: Multi-Dialect & Extensions
- [ ] Dialect/genre router (royal inscriptions vs. commercial vs. literary)
- [ ] Sumerian support
- [ ] Cuneiform OCR integration
- [ ] Multilingual source (beyond English)
- [ ] Community-driven sign table expansion

## Deployment Strategy

| Stack | Use case | Characteristics |
|-------|----------|-----------------|
| **Public demo** | HF Spaces / lightweight hosting | Experience mode, short text cache, CPU renderer, rate limited |
| **Research/education** | Local or private deployment | Full pipeline, GPU for model, exportable results |

Renderer is CPU-only (<10ms/tablet). GPU only needed for translation model. Common short texts can be cached to avoid model inference entirely.

## Domain Gap: The Fundamental Constraint

Akkadian is not one language — it is a spectrum of dialects, genres, and periods. A model trained on Neo-Assyrian royal inscriptions will struggle with Old Assyrian trade letters, regardless of BLEU score. Engineering response:

- **Do not** pretend one model handles everything
- **Do** build toward router + specialist assets
- **Do** tell users which text genre the model was trained on
- **Do** flag when input style doesn't match training distribution

This is the single most important insight from the Deep Past competition: all three gold-medal teams emphasized data quality and domain adaptation over model architecture.
