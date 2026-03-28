"""HuggingFace Spaces entry point.

Deploy: create a HF Space, upload this file + tabletcraft/ + knowledge/
Set Space SDK to "gradio".
"""

import os
from tabletcraft.interfaces.demo import create_demo

# On HF Spaces, model path from env var or default
model_path = os.environ.get("TABLETCRAFT_MODEL", None)

demo = create_demo(model_path=model_path)
demo.launch()
