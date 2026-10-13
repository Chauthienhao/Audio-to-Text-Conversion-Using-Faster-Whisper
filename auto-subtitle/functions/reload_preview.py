from imports import *

def update_subtitle(video_path, srt_text, srt_file):
    if not video_path:
        raise gr.Error("Cần có video.")
    if not srt_file:
        raise gr.Error("Chưa có phụ đề. Hãy bấm 'Tạo phụ đề' trước.")
    if not srt_text or not srt_text.strip():
        raise gr.Error("Nội dung phụ đề đang trống.")

    srt_path = Path(srt_file)
    srt_path.write_text(srt_text, encoding="utf-8")
    return (
        gr.Video(value=video_path, subtitles=str(srt_path)),
        gr.update(value=str(srt_path), visible=True),
    )