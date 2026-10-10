from imports import *
from functions.srt_to_text import srt_to_text
def reload_preview(video_path, srt_file):
    if not video_path or not srt_file:
        raise gr.Error("Cần có video và file phụ đề.")
    srt_path = Path(srt_file)
    text = srt_to_text(srt_path.read_text(encoding="utf-8"))
    return gr.Video(value=video_path, subtitles=str(srt_path)), text
