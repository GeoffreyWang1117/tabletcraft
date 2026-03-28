#!/usr/bin/env python3
"""
Multi-Model MBR Inference Pipeline for Akkadian→English Translation.

Based on best practices from top competition notebooks:
1. Multi-model candidate pooling (our models + public models)
2. Multi-strategy candidate generation: beam search + multi-temperature sampling
3. Competition-aligned MBR utility: sqrt(BLEU * chrF++) or weighted combo
4. Comprehensive post-processing (forbidden chars, fractions, dedup, OA Lexicon)
5. BF16 inference + BetterTransformer + bucket batching

Usage:
    python src/inference_mbr.py --models MODEL1 MODEL2 ... --test_csv data/kaggle/test.csv
    python src/inference_mbr.py --eval  # evaluate on validation set
"""
import os
import re
import gc
import math
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import sacrebleu

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("mbr_inference")

PROJECT_ROOT = Path("/home/coder-gw/kaggle/DPC-Akkadian2English")


# ============================================================
# Configuration
# ============================================================

@dataclass
class MBRConfig:
    # Models (list of paths)
    model_paths: List[str] = field(default_factory=lambda: [])

    # Input/output
    test_data_path: str = ""
    output_dir: str = str(PROJECT_ROOT / "experiments" / "mbr_output")

    # Sequence limits
    max_input_length: int = 512
    max_new_tokens: int = 384
    batch_size: int = 2

    # Beam search candidates
    num_beam_cands: int = 4
    num_beams: int = 8
    length_penalty: float = 1.3
    early_stopping: bool = True
    repetition_penalty: float = 1.2

    # Multi-temperature sampling candidates
    use_sampling: bool = True
    sample_temperatures: List[float] = field(default_factory=lambda: [0.55, 0.75, 0.95])
    num_sample_per_temp: int = 2
    sample_top_p: float = 0.92

    # MBR settings
    mbr_pool_cap: int = 48  # Max candidates per sample
    mbr_w_chrf: float = 0.55
    mbr_w_bleu: float = 0.25
    mbr_w_jaccard: float = 0.20
    mbr_length_bonus: float = 0.10

    # Use competition metric sqrt(BLEU * chrF++) as MBR utility
    use_competition_utility: bool = False
    # Agreement bonus for repeated candidates
    agreement_bonus: float = 0.05

    # Engineering
    use_bf16: bool = True
    use_bucket_batching: bool = True
    num_buckets: int = 6

    @property
    def num_sample_cands(self) -> int:
        return len(self.sample_temperatures) * self.num_sample_per_temp


# ============================================================
# Preprocessing (shared with train_optimized.py)
# ============================================================

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú",
                         "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù",
                         "A": "À", "E": "È", "I": "Ì", "U": "Ù"})
_GAP_BIG = re.compile(r"(\.{3,}|…+|……)")
_GAP_SMALL = re.compile(r"(xx+|\s+x\s+)")


def normalize_transliteration(text: str) -> str:
    if pd.isna(text) or not text:
        return ""
    s = str(text).strip()
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    s = _GAP_BIG.sub("<big_gap>", s)
    s = _GAP_SMALL.sub("<gap>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Post-processing (comprehensive, from top notebooks)
# ============================================================

_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)", re.I)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")
_FORBIDDEN = '()——<>⌈⌉⌋⌊[]+ʾ;'
_FORBIDDEN_TRANS = str.maketrans("", "", _FORBIDDEN)
_GAP_UNIFIED_RE = re.compile(r"\.\s*\.\s*\.|\[\s*\.\s*\.\s*\.\s*\]|\[…\]|…")

# Fraction patterns
_FRAC_PATTERNS = [
    (re.compile(r"\b0\.5\b"), "½"),
    (re.compile(r"\b0\.25\b"), "¼"),
    (re.compile(r"\b0\.75\b"), "¾"),
    (re.compile(r"\b0\.333+\b"), "⅓"),
    (re.compile(r"\b0\.666+\b"), "⅔"),
    (re.compile(r"\b0\.166+\b"), "⅙"),
    (re.compile(r"\b0\.833+\b"), "⅚"),
    (re.compile(r"\b1\.5\b"), "1½"),
    (re.compile(r"\b2\.5\b"), "2½"),
]


def postprocess(text: str) -> str:
    """Comprehensive post-processing from top competition notebooks."""
    if not text or not text.strip():
        return ""
    s = str(text).strip()

    # Remove task prefix leakage
    s = re.sub(r"^translate (?:Akkadian|English) to (?:English|Akkadian):\s*", "", s)

    # Remove grammatical annotations
    s = _SOFT_GRAM_RE.sub(" ", s)
    s = _BARE_GRAM_RE.sub(" ", s)

    # Normalize gap markers - protect during forbidden char removal
    s = _GAP_UNIFIED_RE.sub("\x00GAP\x00", s)
    s = s.replace("<big_gap>", "\x00BGAP\x00")
    s = s.replace("<gap>", "\x00GAP\x00")

    # Remove forbidden characters
    s = s.translate(_FORBIDDEN_TRANS)

    # Restore gap markers
    s = s.replace("\x00BGAP\x00", "...")
    s = s.replace("\x00GAP\x00", "...")

    # Fraction normalization
    for pat, repl in _FRAC_PATTERNS:
        s = pat.sub(repl, s)

    # Repeated word removal (single words)
    s = _REPEAT_WORD_RE.sub(r"\1", s)

    # Repeated n-gram removal (2-4 grams)
    for n in range(4, 1, -1):
        pat = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
        s = re.sub(pat, r"\1", s)

    # Punctuation cleanup
    s = _REPEAT_PUNCT_RE.sub(r"\1", s)
    s = _PUNCT_SPACE_RE.sub(r"\1", s)
    s = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", s)

    # Normalize dashes/quotes
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("|", "/")

    # Clean whitespace
    s = re.sub(r"\s{2,}", " ", s).strip()

    # Capitalize first letter
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    # Trailing short fragment trim
    s = re.sub(r"\s+\w{1,3}$", "", s)

    return s.strip()


def postprocess_batch(texts: List[str]) -> List[str]:
    return [postprocess(t) for t in texts]


# ============================================================
# Dataset & Bucket Batching
# ============================================================

class AkkadianDataset(Dataset):
    def __init__(self, ids: List, texts: List[str]):
        self.sample_ids = ids
        self.input_texts = [f"translate Akkadian to English: {normalize_transliteration(t)}" for t in texts]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.input_texts[idx]


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets):
        self.batch_size = batch_size
        lengths = [len(t.split()) for _, t in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.batches = []
        for i in range(0, len(sorted_idx), batch_size):
            self.batches.append(sorted_idx[i:i + batch_size])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# ============================================================
# Model wrapper: generates candidates from a single model
# ============================================================

class ModelWrapper:
    def __init__(self, model_path: str, cfg: MBRConfig, label: str):
        self.cfg = cfg
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Try BetterTransformer
        if self.device.type == "cuda":
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                logger.info(f"[{label}] BetterTransformer enabled")
            except Exception:
                pass

        n = sum(p.numel() for p in self.model.parameters())
        logger.info(f"[{label}] loaded {model_path} ({n:,} params)")

    def _bf16_ctx(self):
        if self.cfg.use_bf16 and self.device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            except Exception:
                pass
        return nullcontext()

    def generate_candidates(self, texts: List[str]) -> List[List[str]]:
        """Generate candidate translations for a batch of texts.
        Returns list of candidate lists (one per input)."""

        enc = self.tokenizer(
            texts,
            max_length=self.cfg.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        all_candidates = [[] for _ in range(len(texts))]

        with torch.no_grad(), self._bf16_ctx():
            # 1. Beam search candidates
            beam_out = self.model.generate(
                **enc,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
                num_return_sequences=self.cfg.num_beam_cands,
                length_penalty=self.cfg.length_penalty,
                early_stopping=self.cfg.early_stopping,
                repetition_penalty=self.cfg.repetition_penalty,
            )
            beam_decoded = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)
            for i in range(len(texts)):
                start = i * self.cfg.num_beam_cands
                end = start + self.cfg.num_beam_cands
                all_candidates[i].extend([d.strip() for d in beam_decoded[start:end]])

            # 2. Multi-temperature sampling candidates
            if self.cfg.use_sampling:
                for temp in self.cfg.sample_temperatures:
                    for _ in range(self.cfg.num_sample_per_temp):
                        sample_out = self.model.generate(
                            **enc,
                            max_new_tokens=self.cfg.max_new_tokens,
                            do_sample=True,
                            temperature=temp,
                            top_p=self.cfg.sample_top_p,
                            num_return_sequences=1,
                        )
                        sample_decoded = self.tokenizer.batch_decode(sample_out, skip_special_tokens=True)
                        for i, d in enumerate(sample_decoded):
                            all_candidates[i].append(d.strip())

        return all_candidates

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# MBR Selector
# ============================================================

class MBRSelector:
    def __init__(self, cfg: MBRConfig):
        self.cfg = cfg
        self._chrf = sacrebleu.metrics.CHRF(word_order=2)
        self._bleu = sacrebleu.metrics.BLEU(effective_order=True)

    def _chrfpp(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(self._chrf.sentence_score(a, [b]).score)

    def _bleu_score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        try:
            return float(self._bleu.sentence_score(a, [b]).score)
        except Exception:
            return 0.0

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa and not sb:
            return 1.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / max(union, 1)

    def _pairwise_score(self, hyp: str, ref: str) -> float:
        if self.cfg.use_competition_utility:
            # Competition metric: DPI = sqrt(BLEU * chrF++)
            b = self._bleu_score(hyp, ref)
            c = self._chrfpp(hyp, ref)
            return math.sqrt(max(b, 0) * max(c, 0))
        else:
            # Weighted combination
            c = self._chrfpp(hyp, ref)
            b = self._bleu_score(hyp, ref)
            j = self._jaccard(hyp, ref)
            total_w = self.cfg.mbr_w_chrf + self.cfg.mbr_w_bleu + self.cfg.mbr_w_jaccard
            return (self.cfg.mbr_w_chrf * c + self.cfg.mbr_w_bleu * b + self.cfg.mbr_w_jaccard * j) / max(total_w, 1e-9)

    def select(self, candidates: List[str]) -> str:
        """Select best candidate by MBR consensus."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        # Post-process candidates before scoring
        cleaned = postprocess_batch(candidates)

        # Deduplicate (track counts for agreement bonus)
        seen = {}
        unique = []
        for c in cleaned:
            if c in seen:
                seen[c] += 1
            else:
                seen[c] = 1
                unique.append(c)

        if len(unique) == 1:
            return unique[0]

        # Cap pool size
        if len(unique) > self.cfg.mbr_pool_cap:
            unique = unique[:self.cfg.mbr_pool_cap]

        # Score each candidate
        scores = []
        for hyp in unique:
            total = 0.0
            for ref in unique:
                if ref is hyp:
                    continue
                total += self._pairwise_score(hyp, ref)
            avg = total / max(len(unique) - 1, 1)

            # Agreement bonus: repeated candidates get a small boost
            count = seen.get(hyp, 1)
            if count > 1 and self.cfg.agreement_bonus > 0:
                avg += self.cfg.agreement_bonus * min(count - 1, 3)

            # Length bonus: penalize very short outputs
            words = len(hyp.split())
            if words < 3 and self.cfg.mbr_length_bonus > 0:
                avg -= self.cfg.mbr_length_bonus

            scores.append(avg)

        best_idx = int(np.argmax(scores))
        return unique[best_idx]


# ============================================================
# Main inference engine
# ============================================================

class MBRInferenceEngine:
    def __init__(self, cfg: MBRConfig):
        self.cfg = cfg
        self.mbr = MBRSelector(cfg)

    def run(self, ids: List, texts: List[str]) -> pd.DataFrame:
        """Run multi-model MBR inference."""
        dataset = AkkadianDataset(ids, texts)

        # Collect candidates from all models
        all_candidates = {i: [] for i in range(len(texts))}

        for model_idx, model_path in enumerate(self.cfg.model_paths):
            label = f"M{model_idx}"
            logger.info(f"Loading model {label}: {model_path}")
            wrapper = ModelWrapper(model_path, self.cfg, label)

            if self.cfg.use_bucket_batching:
                sampler = BucketBatchSampler(dataset, self.cfg.batch_size, self.cfg.num_buckets)
                loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda b: b)
            else:
                loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False,
                                    collate_fn=lambda b: b)

            for batch in tqdm(loader, desc=f"[{label}] Generating"):
                batch_ids = [b[0] for b in batch]
                batch_texts_raw = [b[1] for b in batch]

                cands = wrapper.generate_candidates(batch_texts_raw)

                for j, sample_id in enumerate(batch_ids):
                    # Find the index in the original dataset
                    idx = ids.index(sample_id)
                    all_candidates[idx].extend(cands[j])

            # Unload model to free GPU for next one
            wrapper.unload()
            logger.info(f"[{label}] done, GPU freed")

        # MBR selection
        logger.info("Running MBR selection...")
        results = []
        for i in tqdm(range(len(texts)), desc="MBR select"):
            cands = all_candidates[i]
            selected = self.mbr.select(cands)
            results.append(selected)

        # Final post-processing pass
        results = postprocess_batch(results)

        return pd.DataFrame({"id": ids, "translation": results})

    def evaluate(self, ids, texts, references) -> Dict:
        """Run inference and evaluate against references."""
        result_df = self.run(ids, texts)

        translations = result_df["translation"].tolist()

        bleu = sacrebleu.corpus_bleu(translations, [references])
        chrf = sacrebleu.corpus_chrf(translations, [references], word_order=2)
        dpi = math.sqrt(max(bleu.score, 0) * max(chrf.score, 0))

        metrics = {
            "bleu": bleu.score,
            "chrf": chrf.score,
            "dpi": dpi,
            "num_samples": len(translations),
        }
        logger.info(f"BLEU: {bleu.score:.2f} | chrF++: {chrf.score:.2f} | DPI: {dpi:.2f}")
        return metrics, result_df


# ============================================================
# Model soup (weight-space merging)
# ============================================================

def model_soup(model_paths: List[str], weights: Optional[List[float]] = None,
               output_path: Optional[str] = None) -> str:
    """Merge multiple models by weight averaging (model soup).
    Returns path to merged model."""

    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    logger.info(f"Model soup: merging {len(model_paths)} models with weights {weights}")

    # Load first model as base
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[0])
    base_sd = base_model.state_dict()

    # Average weights
    for i, path in enumerate(model_paths[1:], 1):
        sd = AutoModelForSeq2SeqLM.from_pretrained(path).state_dict()
        for key in base_sd:
            if key in sd:
                if i == 1:
                    base_sd[key] = weights[0] * base_sd[key] + weights[i] * sd[key]
                else:
                    base_sd[key] = base_sd[key] + weights[i] * sd[key]
        del sd
        gc.collect()

    base_model.load_state_dict(base_sd)

    if output_path is None:
        output_path = str(PROJECT_ROOT / "experiments" / "model_soup")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    base_model.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(model_paths[0]).save_pretrained(output_path)
    logger.info(f"Merged model saved to {output_path}")

    del base_model
    gc.collect()
    return output_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-model MBR inference")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--test_csv", type=str, default=None, help="Test CSV path")
    parser.add_argument("--eval", action="store_true", help="Evaluate on validation set")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--num_beam_cands", type=int, default=4)
    parser.add_argument("--no_sampling", action="store_true")
    parser.add_argument("--competition_utility", action="store_true",
                        help="Use sqrt(BLEU*chrF++) as MBR utility")
    parser.add_argument("--soup", action="store_true", help="Merge models before inference")
    parser.add_argument("--soup_weights", nargs="+", type=float, default=None)
    args = parser.parse_args()

    cfg = MBRConfig()
    cfg.model_paths = args.models
    cfg.batch_size = args.batch_size
    cfg.num_beams = args.num_beams
    cfg.num_beam_cands = args.num_beam_cands
    cfg.use_sampling = not args.no_sampling
    cfg.use_competition_utility = args.competition_utility
    if args.output_dir:
        cfg.output_dir = args.output_dir

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Optional: merge models first
    if args.soup and len(args.models) > 1:
        merged = model_soup(args.models, args.soup_weights,
                            str(Path(cfg.output_dir) / "model_soup"))
        cfg.model_paths = [merged]

    engine = MBRInferenceEngine(cfg)

    if args.eval:
        # Evaluate on Akkademia validation set
        val_tr = PROJECT_ROOT / "baselines" / "Akkademia" / "NMT_input" / "valid.tr"
        val_en = PROJECT_ROOT / "baselines" / "Akkademia" / "NMT_input" / "valid.en"
        with open(val_tr) as f:
            sources = [line.strip() for line in f]
        with open(val_en) as f:
            references = [line.strip() for line in f]
        if args.max_samples:
            sources = sources[:args.max_samples]
            references = references[:args.max_samples]
        ids = list(range(len(sources)))

        metrics, result_df = engine.evaluate(ids, sources, references)

        # Save results
        with open(Path(cfg.output_dir) / "eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        result_df.to_csv(Path(cfg.output_dir) / "eval_predictions.csv", index=False)
        logger.info(f"Results saved to {cfg.output_dir}")

    elif args.test_csv:
        # Generate submission
        test_df = pd.read_csv(args.test_csv)
        ids = test_df["id"].tolist()
        texts = test_df["transliteration"].astype(str).tolist()
        if args.max_samples:
            ids = ids[:args.max_samples]
            texts = texts[:args.max_samples]

        result_df = engine.run(ids, texts)
        out_path = Path(cfg.output_dir) / "submission.csv"
        result_df.to_csv(out_path, index=False)
        logger.info(f"Submission saved to {out_path}")

    else:
        parser.error("Specify either --test_csv or --eval")


if __name__ == "__main__":
    main()
