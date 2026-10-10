from pathlib import Path
import gradio as gr
import time
from core.config import SUBS_OUTPUT_DIR, LANGUAGES
from core.transcriber import transcribe
from core.subtitle import to_srt
from core.burner import burn_subtitle
from core.cache import srt_path_for,file_hash