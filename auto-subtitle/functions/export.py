from imports import *
def export_video(video_path, srt_file, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Vui lòng chọn video trước.")
    if not srt_file:
        raise gr.Error("Chưa có file phụ đề. Hãy bấm 'Tạo phụ đề' trước.")
    progress(0.1, desc="Đang ghi phụ đề vào video...")
    out = SUBS_OUTPUT_DIR / f"{file_hash(video_path)}_{file_hash(srt_file)}.mp4"
    if out.exists():
        return str(out)
    try:
        return burn_subtitle(video_path, srt_file, out)
    except RuntimeError as e:
        raise gr.Error(f"ffmpeg lỗi: {e}")