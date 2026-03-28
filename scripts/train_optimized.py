#!/usr/bin/env python3
"""
Optimized ByT5 Training for Akkadian→English Translation.

Improvements over train_akkademia.py based on top competition notebooks:
1. Combined data: Akkademia 50K + competition train.csv + sentences_parallel.csv
2. Sentence alignment splitting (doc→sentence pairs)
3. Bidirectional training (Ak→En + En→Ak) as data augmentation
4. Competition-aligned eval metric: DPI = sqrt(BLEU * chrF++)
5. FP32 training (ByT5 NaN with FP16), BF16 on Ampere+
6. Adafactor optimizer (memory efficient for large models)
7. Label smoothing 0.2 (from starter notebook)
8. Supports ByT5-small/base/large
"""
import os
import re
import json
import math
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset, concatenate_datasets
import evaluate

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/coder-gw/kaggle/DPC-Akkadian2English")
AKKADEMIA_DIR = PROJECT_ROOT / "baselines" / "Akkademia" / "NMT_input"
KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle"


# ============================================================
# Preprocessing (from top notebooks)
# ============================================================

# Diacritic normalization (from ngyzly/better-candidate-diversity)
_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú",
                         "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù",
                         "A": "À", "E": "È", "I": "Ì", "U": "Ù"})
_GAP_BIG = re.compile(r"(\.{3,}|…+|……)")
_GAP_SMALL = re.compile(r"(xx+|\s+x\s+)")


def normalize_transliteration(text: str) -> str:
    """Normalize Akkadian transliteration with diacritics and gap markers."""
    if pd.isna(text) or not text:
        return ""
    s = str(text).strip()
    # ASCII diacritics → Unicode
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    # Gap markers
    s = _GAP_BIG.sub("<big_gap>", s)
    s = _GAP_SMALL.sub("<gap>", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# Sentence alignment (from takamichitoda/dpc-starter-train)
# ============================================================

def sentence_align(sources: List[str], targets: List[str]) -> Tuple[List[str], List[str]]:
    """
    Split multi-sentence document pairs into aligned sentence pairs.
    Heuristic: if #English sentences == #Akkadian lines, treat as 1:1 aligned.
    """
    aligned_src, aligned_tgt = [], []
    for src, tgt in zip(sources, targets):
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]

        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned_src.append(s)
                    aligned_tgt.append(t)
        else:
            aligned_src.append(src)
            aligned_tgt.append(tgt)

    return aligned_src, aligned_tgt


# ============================================================
# Data loading
# ============================================================

def load_akkademia(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load Akkademia parallel corpus (.tr and .en files)."""
    tr_path = AKKADEMIA_DIR / f"{split}.tr"
    en_path = AKKADEMIA_DIR / f"{split}.en"
    with open(tr_path, encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(en_path, encoding="utf-8") as f:
        targets = [line.strip() for line in f]
    assert len(sources) == len(targets), f"Mismatch: {len(sources)} vs {len(targets)}"
    logger.info(f"Akkademia {split}: {len(sources)} pairs")
    return sources, targets


def load_competition_train() -> Tuple[List[str], List[str]]:
    """Load competition train.csv."""
    df = pd.read_csv(KAGGLE_DIR / "train.csv")
    sources = df["transliteration"].astype(str).tolist()
    targets = df["translation"].astype(str).tolist()
    logger.info(f"Competition train: {len(sources)} pairs")
    return sources, targets


def load_sentences_parallel() -> Tuple[List[str], List[str]]:
    """Load sentences_parallel.csv (additional sentence-level pairs)."""
    path = KAGGLE_DIR / "sentences_parallel.csv"
    if not path.exists():
        logger.info("sentences_parallel.csv not found, skipping")
        return [], []
    df = pd.read_csv(path)
    sources = df["transliteration"].astype(str).tolist()
    targets = df["translation"].astype(str).tolist()
    # Filter out very short/empty pairs
    filtered_src, filtered_tgt = [], []
    for s, t in zip(sources, targets):
        if len(s.strip()) > 3 and len(t.strip()) > 3:
            filtered_src.append(s.strip())
            filtered_tgt.append(t.strip())
    logger.info(f"Sentences parallel: {len(filtered_src)} pairs (after filtering)")
    return filtered_src, filtered_tgt


def build_training_data(
    use_sentence_align: bool = True,
    use_bidirectional: bool = True,
    use_competition_data: bool = True,
    use_sentences_parallel: bool = True,
) -> Tuple[HFDataset, HFDataset]:
    """Build combined training and validation datasets."""

    # === Training data ===
    all_src, all_tgt = [], []

    # 1. Akkademia 50K
    akk_src, akk_tgt = load_akkademia("train")
    if use_sentence_align:
        akk_src, akk_tgt = sentence_align(akk_src, akk_tgt)
        logger.info(f"After sentence alignment: {len(akk_src)} pairs")
    all_src.extend(akk_src)
    all_tgt.extend(akk_tgt)

    # 2. Competition train.csv
    if use_competition_data:
        comp_src, comp_tgt = load_competition_train()
        if use_sentence_align:
            comp_src, comp_tgt = sentence_align(comp_src, comp_tgt)
            logger.info(f"Competition after alignment: {len(comp_src)} pairs")
        all_src.extend(comp_src)
        all_tgt.extend(comp_tgt)

    # 3. sentences_parallel.csv
    if use_sentences_parallel:
        sp_src, sp_tgt = load_sentences_parallel()
        all_src.extend(sp_src)
        all_tgt.extend(sp_tgt)

    logger.info(f"Total forward pairs: {len(all_src)}")

    # Normalize transliterations
    all_src = [normalize_transliteration(s) for s in all_src]

    # Build forward dataset (Ak→En)
    fwd_inputs = [f"translate Akkadian to English: {s}" for s in all_src]
    fwd_targets = all_tgt

    if use_bidirectional:
        # Reverse dataset (En→Ak) — doubles training data
        rev_inputs = [f"translate English to Akkadian: {t}" for t in all_tgt]
        rev_targets = all_src  # Raw transliterations as targets
        all_inputs = fwd_inputs + rev_inputs
        all_targets = fwd_targets + rev_targets
        logger.info(f"With bidirectional: {len(all_inputs)} total pairs")
    else:
        all_inputs = fwd_inputs
        all_targets = fwd_targets

    train_dataset = HFDataset.from_dict({"source": all_inputs, "target": all_targets})
    train_dataset = train_dataset.shuffle(seed=42)

    # === Validation data (forward only) ===
    val_src, val_tgt = load_akkademia("valid")
    val_src = [normalize_transliteration(s) for s in val_src]
    val_inputs = [f"translate Akkadian to English: {s}" for s in val_src]
    valid_dataset = HFDataset.from_dict({"source": val_inputs, "target": val_tgt})

    logger.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    return train_dataset, valid_dataset


# ============================================================
# Tokenization
# ============================================================

def tokenize_fn(examples, tokenizer, max_src_len, max_tgt_len):
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_src_len,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples["target"],
        max_length=max_tgt_len,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================================
# Competition-aligned evaluation metric: DPI = sqrt(BLEU * chrF++)
# ============================================================

def safe_decode(tokenizer, token_ids):
    max_valid_id = tokenizer.vocab_size - 1
    token_ids = np.clip(token_ids, 0, max_valid_id)
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except (ValueError, OverflowError):
        valid_ids = [int(t) for t in token_ids if 0 <= t <= max_valid_id]
        try:
            return tokenizer.decode(valid_ids, skip_special_tokens=True)
        except Exception:
            return ""


def make_compute_metrics(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if hasattr(predictions, "ndim") and predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
        predictions = predictions.astype(np.int64)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [safe_decode(tokenizer, p).strip() for p in predictions]
        decoded_labels = [safe_decode(tokenizer, l).strip() for l in labels]

        # BLEU
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )
        bleu_score = bleu_result["score"]

        # chrF++
        chrf_result = chrf_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
            word_order=2,
        )
        chrf_score = chrf_result["score"]

        # DPI = sqrt(BLEU * chrF++)
        dpi = math.sqrt(max(bleu_score, 0) * max(chrf_score, 0))

        return {
            "bleu": bleu_score,
            "chrf": chrf_score,
            "dpi": dpi,
        }

    return compute_metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized ByT5 training for Akkadian NMT")
    parser.add_argument("--model_name", type=str, default="google/byt5-base",
                        help="Model: google/byt5-small, google/byt5-base, google/byt5-large")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device batch size (auto if not set)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (auto if not set)")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beams for eval generation (lower=faster)")
    parser.add_argument("--no_bidirectional", action="store_true")
    parser.add_argument("--no_sentence_align", action="store_true")
    parser.add_argument("--no_competition_data", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Auto-configure batch size based on model size
    model_size = "small" if "small" in args.model_name else \
                 "base" if "base" in args.model_name else "large"

    if args.batch_size is None:
        args.batch_size = {"small": 8, "base": 4, "large": 1}[model_size]
    if args.grad_accum is None:
        # Target effective batch ~32
        args.grad_accum = max(1, 32 // (args.batch_size * 2))  # 2 GPUs

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "experiments" / f"byt5_{model_size}_optimized")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Model:           {args.model_name} ({model_size})")
    logger.info(f"Batch:           {args.batch_size} x {args.grad_accum} x 2gpu = {args.batch_size * args.grad_accum * 2}")
    logger.info(f"LR:              {args.learning_rate}")
    logger.info(f"Epochs:          {args.num_epochs}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info(f"Bidirectional:   {not args.no_bidirectional}")
    logger.info(f"Sentence align:  {not args.no_sentence_align}")
    logger.info(f"Output:          {args.output_dir}")
    logger.info("=" * 60)

    # Load model & tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Gradient checkpointing for large models
    if model_size in ("base", "large"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {params:,}")

    # Build data
    train_dataset, valid_dataset = build_training_data(
        use_sentence_align=not args.no_sentence_align,
        use_bidirectional=not args.no_bidirectional,
        use_competition_data=not args.no_competition_data,
    )

    # Tokenize
    logger.info("Tokenizing...")
    tok_kwargs = dict(tokenizer=tokenizer, max_src_len=args.max_length, max_tgt_len=args.max_length)
    train_dataset = train_dataset.map(
        lambda x: tokenize_fn(x, **tok_kwargs),
        batched=True, remove_columns=["source", "target"], desc="Tokenize train",
    )
    valid_dataset = valid_dataset.map(
        lambda x: tokenize_fn(x, **tok_kwargs),
        batched=True, remove_columns=["source", "target"], desc="Tokenize valid",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    # Determine precision: FP32 for ByT5 (NaN with FP16), BF16 if available
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    logger.info(f"Precision: {'BF16' if use_bf16 else 'FP32'}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing_factor=args.label_smoothing,
        # ByT5 is unstable with FP16, use BF16 or FP32
        fp16=False,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="dpi",  # Competition-aligned metric
        greater_is_better=True,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_length,
        report_to="none",
        save_total_limit=5,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Multi-GPU
        ddp_find_unused_parameters=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    logger.info("Starting training...")
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save
    logger.info(f"Saving to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save config
    config_dict = vars(args)
    config_dict["total_train_samples"] = len(train_dataset)
    config_dict["total_valid_samples"] = len(valid_dataset)
    config_dict["model_params"] = params
    config_dict["precision"] = "bf16" if use_bf16 else "fp32"
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Final eval
    logger.info("Final evaluation...")
    results = trainer.evaluate()
    logger.info(f"Final DPI: {results.get('eval_dpi', 'N/A'):.2f}")
    logger.info(f"  BLEU: {results.get('eval_bleu', 'N/A'):.2f}")
    logger.info(f"  chrF++: {results.get('eval_chrf', 'N/A'):.2f}")

    with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
