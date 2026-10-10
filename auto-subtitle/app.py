#tách cách hàm thành các file khác nhau thay vì gộp thành 1 file app.py
from imports import *
from functions import *

with gr.Blocks(title="Auto Subtitle") as demo:
    gr.Markdown("# Tạo phụ đề tự động với Faster-Whisper")
    with gr.Row():
        with gr.Column():
            video_in = gr.File(label="Video đầu vào")
            lang = gr.Dropdown(list(LANGUAGES), value="Tự động", label="Ngôn ngữ")
            btn_cr_sub = gr.Button("Tạo phụ đề", variant="primary")
            preview_script = gr.Textbox(label="Hiện phụ đề",lines=8,interactive=True)
            btn_update = gr.Button("Cập nhật phụ đề", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Xem thử có phụ đề",interactive=False)
            file_out = gr.File(visible=False)
            export_btn = gr.Button("Xuất video có phụ đề cứng")
            video_final = gr.File(label="Tải video có phụ đề")
    btn_cr_sub.click(fn=generate_subtitle, 
                     inputs=[video_in, lang], 
                     outputs=[video_out, file_out, preview_script])
    # file_out.upload(lambda: None, None, video_out).then(
    # reload_preview, [video_in, file_out], [video_out, preview_script]
    # )
    file_out.clear(lambda: gr.update(visible=False),None,file_out)
    export_btn.click(export_video, [video_in, file_out], video_final)
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch()
