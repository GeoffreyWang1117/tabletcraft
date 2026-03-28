"""Gradio web demo for TabletCraft with confidence gating."""

from typing import Optional


def create_demo(model_path: Optional[str] = None):
    """Create a Gradio demo with 4-panel display and confidence gating."""
    import gradio as gr
    from tabletcraft.knowledge.cuneiform import CuneiformConverter
    from tabletcraft.interfaces.renderer import TabletRenderer
    from tabletcraft.pipeline.classifier import classify
    from tabletcraft.pipeline.validator import validate

    conv = CuneiformConverter()
    renderer = TabletRenderer(width=600, font_size=30, chars_per_line=18)

    translator = None
    if model_path:
        from tabletcraft.models.translator import AkkadianTranslator
        translator = AkkadianTranslator(model_path)

    def transliterate_tab(transliteration: str):
        cuneiform = conv.to_cuneiform(transliteration)
        svg = renderer.render_svg(cuneiform, title=transliteration[:80])
        return cuneiform, svg, "1.00 (direct transliteration)", ""

    def craft_tab(english_text: str):
        if translator is None:
            return "", "", "", "No model loaded", ""

        # Classify
        cls = classify(english_text)
        if cls.input_type.value == "anomalous":
            return "", "", "", f"Input rejected: {'; '.join(cls.warnings)}", ""

        # Translate
        akkadian = translator.to_akkadian(english_text)

        # Validate
        val = validate(akkadian, english_text)

        # Convert + render (or fallback)
        if val.suggestion == "fallback":
            warnings = cls.warnings + val.issues + ["Output did not pass validation"]
            return akkadian, "", "", "\n".join(warnings), f"Confidence: {val.score:.2f} — FALLBACK"

        cuneiform = conv.to_cuneiform(akkadian)
        title = english_text[:80]
        if val.suggestion == "render_with_caveat":
            title = f"[approximate] {title}"
        svg = renderer.render_svg(cuneiform, title=title)

        all_warnings = cls.warnings + val.issues
        all_warnings.append("Machine-generated — approximate rendering, not authentic ancient text")
        warning_text = "\n".join(all_warnings) if all_warnings else ""
        conf_text = f"Confidence: {val.score:.2f} — {val.suggestion.upper()}"

        return akkadian, cuneiform, svg, warning_text, conf_text

    def translate_to_english(akkadian_text: str):
        if translator is None:
            return "No model loaded"
        return translator.to_english(akkadian_text)

    with gr.Blocks(
        title="TabletCraft",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # TabletCraft
            ### Bridging a 4,000-year cultural gap

            Read ancient tablets or write your own messages in cuneiform.
            All outputs are approximate — see warnings for details.
            """
        )

        with gr.Tabs():
            # Tab 1: Transliteration -> Cuneiform (no model needed)
            with gr.TabItem("Transliteration -> Cuneiform"):
                gr.Markdown("Enter Akkadian transliteration to see cuneiform signs. No model needed.")
                with gr.Row():
                    with gr.Column():
                        translit_input = gr.Textbox(
                            label="Akkadian Transliteration",
                            placeholder="e.g., LUGAL dan-nu LUGAL KUR aš-šur",
                            lines=2,
                        )
                        translit_btn = gr.Button("Convert", variant="primary")
                    with gr.Column():
                        cuneiform_out1 = gr.Textbox(label="Cuneiform Unicode")
                        tablet_svg1 = gr.HTML(label="Clay Tablet")
                        conf_out1 = gr.Textbox(label="Confidence")
                        warn_out1 = gr.Textbox(label="Warnings")

                translit_btn.click(
                    transliterate_tab,
                    inputs=[translit_input],
                    outputs=[cuneiform_out1, tablet_svg1, conf_out1, warn_out1],
                )
                gr.Examples(
                    examples=[
                        ["LUGAL dan-nu LUGAL kiš-ša-ti LUGAL KUR aš-šur"],
                        ["a-na {d}aš-šur EN GAL-e EN-ia"],
                        ["um-ma šar-ru-um-ma a-na mar-ia-tim"],
                    ],
                    inputs=[translit_input],
                )

            # Tab 2: English -> Tablet (requires model, with gating)
            if translator is not None:
                with gr.TabItem("English -> Clay Tablet"):
                    gr.Markdown(
                        "Type English text. The system classifies your input, "
                        "translates to Akkadian, validates the output, and renders "
                        "a tablet — or shows a warning if the result is unreliable."
                    )
                    with gr.Row():
                        with gr.Column():
                            eng_input = gr.Textbox(
                                label="English Text",
                                placeholder="e.g., The mighty king rules the land",
                                lines=2,
                            )
                            eng_btn = gr.Button("Craft Tablet", variant="primary")
                        with gr.Column():
                            ak_out = gr.Textbox(label="Akkadian Transliteration")
                            cun_out = gr.Textbox(label="Cuneiform Unicode")
                            tablet_svg2 = gr.HTML(label="Clay Tablet")
                            warn_out2 = gr.Textbox(label="Warnings / Caveats")
                            conf_out2 = gr.Textbox(label="Confidence")

                    eng_btn.click(
                        craft_tab,
                        inputs=[eng_input],
                        outputs=[ak_out, cun_out, tablet_svg2, warn_out2, conf_out2],
                    )

                # Tab 3: Akkadian -> English
                with gr.TabItem("Akkadian -> English"):
                    gr.Markdown("Translate Akkadian transliteration to English.")
                    with gr.Row():
                        ak_input = gr.Textbox(label="Akkadian Transliteration", lines=2)
                        ak_btn = gr.Button("Translate", variant="primary")
                    en_output = gr.Textbox(label="English Translation")
                    ak_btn.click(translate_to_english, inputs=[ak_input], outputs=[en_output])

        gr.Markdown(
            """
            ---
            **TabletCraft** | All outputs are machine-generated approximations.
            Consult Assyriological expertise for research or public-facing use.
            """
        )

    return demo


def launch(model_path: Optional[str] = None, share: bool = False, port: int = 7860):
    demo = create_demo(model_path)
    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(args.model, args.share, args.port)
