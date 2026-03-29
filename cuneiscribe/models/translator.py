"""Neural machine translation: English <-> Akkadian transliteration."""

import re
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


_DEFAULT_MODEL = "google/byt5-small"  # Placeholder; will be replaced with HF hub model


class AkkadianTranslator:
    """Bidirectional English <-> Akkadian translator using fine-tuned ByT5.

    Usage:
        translator = AkkadianTranslator("path/to/model")
        english = translator.to_english("šar-ru dan-nu")
        akkadian = translator.to_akkadian("The mighty king")
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or _DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self._model.to(self.device).eval()

    def _preprocess_akkadian(self, text: str) -> str:
        """Normalize Akkadian transliteration."""
        s = str(text).strip()
        s = re.sub(r"(\.{3,}|…+)", "<big_gap>", s)
        s = re.sub(r"(xx+|\s+x\s+)", "<gap>", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def to_english(self, akkadian: str, num_beams: int = 4, max_length: int = 512) -> str:
        """Translate Akkadian transliteration to English.

        Args:
            akkadian: Akkadian transliteration text
            num_beams: Beam search width
            max_length: Maximum output length

        Returns:
            English translation
        """
        self._load()
        source = f"translate Akkadian to English: {self._preprocess_akkadian(akkadian)}"
        return self._generate(source, num_beams, max_length)

    def to_akkadian(self, english: str, num_beams: int = 4, max_length: int = 512) -> str:
        """Translate English to Akkadian transliteration.

        Args:
            english: English text
            num_beams: Beam search width
            max_length: Maximum output length

        Returns:
            Akkadian transliteration
        """
        self._load()
        source = f"translate English to Akkadian: {english.strip()}"
        return self._generate(source, num_beams, max_length)

    def _generate(self, source: str, num_beams: int, max_length: int) -> str:
        enc = self._tokenizer(
            source, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_length,
                num_beams=num_beams,
                length_penalty=1.3,
                early_stopping=True,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def translate_batch(
        self, texts: List[str], direction: str = "ak2en", batch_size: int = 8
    ) -> List[str]:
        """Translate a batch of texts.

        Args:
            texts: List of input texts
            direction: "ak2en" or "en2ak"
            batch_size: Batch size for inference

        Returns:
            List of translated texts
        """
        self._load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if direction == "ak2en":
                prefix = "translate Akkadian to English: "
                batch = [prefix + self._preprocess_akkadian(t) for t in batch]
            else:
                prefix = "translate English to Akkadian: "
                batch = [prefix + t.strip() for t in batch]

            enc = self._tokenizer(
                batch, return_tensors="pt", max_length=512, truncation=True, padding=True
            ).to(self.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc, max_new_tokens=512, num_beams=4,
                    length_penalty=1.3, early_stopping=True,
                )
            decoded = self._tokenizer.batch_decode(out, skip_special_tokens=True)
            results.extend([d.strip() for d in decoded])
        return results
