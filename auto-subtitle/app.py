from pathlib import Path
import gradio as gr
import time
from core.config import SUBS_OUTPUT_DIR, LANGUAGES
from core.transcriber import transcribe
from core.subtitle import to_srt
from core.burner import burn_subtitle
from core.cache import srt_path_for,file_hash

#region softsub cho video xem trên web
def generate_subtitle(video_path, lang_name, progress=gr.Progress()):
    start = time.time()
    if not video_path:
        raise gr.Error("Vui lòng chọn video trước.")
    language = LANGUAGES[lang_name]
    srt_path = srt_path_for(video_path,language,SUBS_OUTPUT_DIR)
    #nếu tồn tại phụ đề có sẵn thì sẽ dùng lại để sub cho video.
    if srt_path.exists():
        progress(1,desc="Đã có phụ đề, tái sử dụng lại.")
        return gr.Video(value=video_path, subtitles=str(srt_path)), str(srt_path)
    #Nếu chưa có phụ dề sẵn
    progress(0,desc="nạp model...")
    segments, detected = transcribe(
        video_path,
        language=language,
        on_progress=lambda p: progress(p,desc="Đang nhận dạng...")
    )
    if not segments:
        raise gr.Error("Không nhận dạng được lời thoại nào trong video.")

    #ghi file và trả kết quả
    srt_path.write_text(to_srt(segments),encoding="utf-8")
    print(f"Softsub time: {time.time() - start:.02f}")
    return gr.Video(value=video_path, subtitles=str(srt_path)), str(srt_path)
#endregion
#region export video
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
#endregion
with gr.Blocks(title="Auto Subtitle") as demo:
    gr.Markdown("# Tạo phụ đề tự động với Faster-Whisper")
    with gr.Row():
        with gr.Column():
            video_in = gr.File(label="Video đầu vào")
            lang = gr.Dropdown(list(LANGUAGES), value="Tự động", label="Ngôn ngữ")
            btn = gr.Button("Tạo phụ đề", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Xem thử có phụ đề")
            file_out = gr.File(label="Tải file .srt")
            export_btn = gr.Button("Xuất video có phụ đề cứng")
            video_final = gr.File(label="Tải video có phụ đề")
    btn.click(generate_subtitle, [video_in, lang], [video_out, file_out])
    export_btn.click(export_video, [video_in, file_out], video_final)
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch()