"""CuneiScribe CLI: translate, convert, and render with confidence gating."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="cuneiscribe",
        description="Turn any text into cuneiform clay tablets — reliably.",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- cuneiform: transliteration -> cuneiform Unicode ---
    p_cun = sub.add_parser("cuneiform", help="Convert transliteration to cuneiform signs")
    p_cun.add_argument("text", nargs="?", help="Akkadian transliteration")
    p_cun.add_argument("--reverse", action="store_true", help="Cuneiform -> transliteration")

    # --- render: transliteration -> tablet image (no model needed) ---
    p_rend = sub.add_parser("render", help="Render transliteration as a clay tablet image")
    p_rend.add_argument("text", help="Transliteration or cuneiform text")
    p_rend.add_argument("-o", "--output", default="tablet.svg", help="Output file (svg/png)")
    p_rend.add_argument("--title", default=None, help="Title below tablet")
    p_rend.add_argument("--width", type=int, default=600)
    p_rend.add_argument("--font-size", type=int, default=28)

    # --- translate: English <-> Akkadian ---
    p_trans = sub.add_parser("translate", help="Translate between English and Akkadian")
    p_trans.add_argument("text", help="Text to translate")
    p_trans.add_argument("--to", choices=["en", "ak"], default="ak", help="Target language")
    p_trans.add_argument("--model", default=None, help="Model path")

    # --- craft: full pipeline English -> tablet with confidence gating ---
    p_craft = sub.add_parser("craft", help="English -> cuneiform tablet (with confidence gating)")
    p_craft.add_argument("text", help="English text")
    p_craft.add_argument("-o", "--output", default="tablet.svg", help="Output file")
    p_craft.add_argument("--model", default=None, help="Model path")
    p_craft.add_argument("--force", action="store_true", help="Skip validation, render regardless")
    p_craft.add_argument("--json", action="store_true", help="Output full result as JSON")

    # --- classify: check what mode an input would get ---
    p_cls = sub.add_parser("classify", help="Classify input text (name/short/historical/modern/anomalous)")
    p_cls.add_argument("text", help="Text to classify")

    # --- batch: process CSV/JSONL ---
    p_batch = sub.add_parser("batch", help="Batch process CSV/JSONL/TXT file")
    p_batch.add_argument("input", help="Input file (csv/jsonl/txt)")
    p_batch.add_argument("-o", "--output", default="output.jsonl", help="Output file")
    p_batch.add_argument("--model", default=None, help="Model path")
    p_batch.add_argument("--direction", choices=["en2ak", "ak2en"], default="en2ak")

    # --- info: show sign info ---
    p_info = sub.add_parser("info", help="Show cuneiform sign information")
    p_info.add_argument("sign", help="Transliteration value (e.g., 'an', 'LUGAL')")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "cuneiform":
        from cuneiscribe.knowledge.cuneiform import CuneiformConverter
        conv = CuneiformConverter()
        text = args.text or sys.stdin.read().strip()
        if args.reverse:
            print(conv.from_cuneiform(text))
        else:
            print(conv.to_cuneiform(text))

    elif args.command == "render":
        from cuneiscribe.knowledge.cuneiform import CuneiformConverter
        from cuneiscribe.interfaces.renderer import TabletRenderer

        conv = CuneiformConverter()
        has_cuneiform = any(0x12000 <= ord(c) <= 0x1254F for c in args.text)
        cuneiform = args.text if has_cuneiform else conv.to_cuneiform(args.text)

        renderer = TabletRenderer(width=args.width, font_size=args.font_size)
        ext = Path(args.output).suffix.lower()
        if ext == ".png":
            renderer.render_png(cuneiform, title=args.title, output_path=args.output)
        else:
            svg = renderer.render_svg(cuneiform, title=args.title)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(svg)
        print(f"Saved to {args.output}")

    elif args.command == "translate":
        from cuneiscribe.models.translator import AkkadianTranslator
        tr = AkkadianTranslator(args.model)
        if args.to == "en":
            print(tr.to_english(args.text))
        else:
            print(tr.to_akkadian(args.text))

    elif args.command == "craft":
        from cuneiscribe.core import CuneiScribe
        tc = CuneiScribe(model_path=args.model)
        ext = Path(args.output).suffix.lower()
        fmt = "png" if ext == ".png" else "svg"
        result = tc.craft(args.text, format=fmt, force_render=args.force,
                          output_path=args.output)

        if args.json:
            out = {
                "input": result.input_text,
                "type": result.input_type,
                "akkadian": result.akkadian,
                "cuneiform": result.cuneiform,
                "confidence": result.confidence,
                "warnings": result.warnings,
                "mode": result.mode,
                "suggestion": result.suggestion,
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            # 4-panel display
            print(f"Input:          {result.input_text}")
            print(f"Type:           {result.input_type}")
            print(f"Akkadian:       {result.akkadian}")
            print(f"Cuneiform:      {result.cuneiform}")
            print(f"Confidence:     {result.confidence:.2f}")
            print(f"Suggestion:     {result.suggestion}")
            if result.warnings:
                print("Warnings:")
                for w in result.warnings:
                    print(f"  - {w}")
            if result.image:
                print(f"Saved to {args.output}")
            else:
                print("(No tablet rendered — output did not pass validation)")

    elif args.command == "classify":
        from cuneiscribe.pipeline.classifier import classify
        result = classify(args.text)
        print(f"Type:       {result.input_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Mode:       {result.suggested_mode}")
        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  - {w}")

    elif args.command == "batch":
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
        from cuneiscribe.pipeline.batch import process_batch, read_input, write_output

        texts = read_input(args.input)
        print(f"Read {len(texts)} items from {args.input}")

        translator = None
        if args.model:
            from cuneiscribe.models.translator import AkkadianTranslator
            translator = AkkadianTranslator(args.model)

        results = process_batch(texts, translator=translator, direction=args.direction)
        write_output(results, args.output)

        # Print summary
        from collections import Counter
        statuses = Counter(r.get("status", "unknown") for r in results)
        print(f"Done: {dict(statuses)}")

    elif args.command == "info":
        from cuneiscribe.knowledge.cuneiform import CuneiformConverter
        conv = CuneiformConverter()
        info = conv.get_sign_info(args.sign)
        if info:
            print(f"Sign:      {info['transliteration']}")
            print(f"Cuneiform: {info['cuneiform']}")
            print(f"Unicode:   {info['unicode']}")
        else:
            print(f"Sign '{args.sign}' not found in database ({conv.num_signs} signs loaded)")


if __name__ == "__main__":
    main()
