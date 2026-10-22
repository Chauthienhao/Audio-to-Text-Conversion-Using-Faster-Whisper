from imports import *

def generate_subtitle(video_path, lang_name, progress=gr.Progress()):
    start = time.time()
    if not video_path:
        raise gr.Error("Vui lòng chọn video trước.")
    language = LANGUAGES[lang_name]
    srt_path = srt_path_for(video_path,language,SUBS_OUTPUT_DIR)
    #nếu tồn tại phụ đề có sẵn thì sẽ dùng lại để sub cho video.
    if srt_path.exists():
        progress(1,desc="Đã có phụ đề, tái sử dụng lại.")
        full_text = srt_path.read_text(encoding='utf-8')
        return gr.Video(value=video_path, subtitles=str(srt_path)), gr.update(value=str(srt_path), label="Tải file .srt", visible=True), full_text
    #Nếu chưa có phụ đề sẵn
    progress(0,desc="nạp model...")
    model_time = time.perf_counter()
    segments, detected = transcribe(
        video_path,
        language=language,
        on_progress=lambda p: progress(p,desc="Đang nhận dạng...")
    )
    model_end = time.perf_counter()
    if not segments:
        raise gr.Error("Không nhận dạng được lời thoại nào trong video.")

    #ghi file và trả kết quả
    srt_path.write_text(to_srt(segments),encoding="utf-8")
    full_text = srt_path.read_text(encoding='utf-8')
    full_time = time.time() - start
    model_time = model_end - model_time
    print(f"Full time: {full_time:.02f}")
    print(f"Softsub time: {full_time - model_time:.02f}")
    return gr.Video(value=video_path, subtitles=str(srt_path)), gr.update(value=str(srt_path), label="Tải file .srt", visible=True), full_text

