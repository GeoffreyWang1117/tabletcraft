"""HuggingFace Spaces entry point.

Deploy: create a HF Space, upload this file + cuneiscribe/ + knowledge/
Set Space SDK to "gradio".
"""

import os
from cuneiscribe.interfaces.demo import create_demo

# On HF Spaces, model path from env var or default
model_path = os.environ.get("CUNEISCRIBE_MODEL", None)

demo = create_demo(model_path=model_path)
demo.launch()
