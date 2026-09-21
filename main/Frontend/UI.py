import gradio as gr
import os
import subprocess
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # Trỏ tới thư mục Frontend
root_dir = os.path.dirname(current_dir) # Lùi lại 1 cấp ra thư mục cha (main)

# 2. Đưa thư mục gốc vào danh sách tìm kiếm module của Python
sys.path.append(root_dir)
from backend.app.process import transcript_audio, save

# Hàm quy đổi giây sang chuẩn SRT bắt buộc để FFmpeg có thể đọc và ép chữ vào video
def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msec = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msec:03d}"

def process_media(file_path):
    # Phân loại Audio hay Video
    ext = os.path.splitext(file_path)[-1].lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
    is_video = ext in video_extensions
    
    srt_filename ="phude_ketqua.srt"
    output_video_filename ="video_cophude.mp4"

    # 1. Gọi hàm transcript_audio từ file process.py
    segments = transcript_audio(input=file_path)
    save(segments, srt_filename)
    # 3. Rẽ nhánh hiển thị
    if is_video:
        # Nhúng cứng phụ đề vào video bằng FFmpeg
        escaped_srt = srt_filename.replace("\\", "/").replace(":", "\\:")
        ffmpeg_cmd = [
            "ffmpeg", "-y", 
            "-i", file_path, 
            "-vf", f"subtitles={escaped_srt}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-preset", "fast",
            output_video_filename
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True,capture_output=True,text=True)
            return output_video_filename, [output_video_filename, srt_filename]
        except subprocess.CalledProcessError as e:   
            print(f"Error occurred while embedding subtitles: {e}")
            print(e.stderr)
            return None, [srt_filename]          
    else:
        # Nếu là Audio: Ẩn video player, chỉ cho tải file SRT
        return None, [srt_filename]

# Thiết kế giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎙️ Giao Diện Nhận Dạng Giọng Nói (Faster-Whisper)")
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Tải lên Video hoặc Audio", type="filepath")
            submit_btn = gr.Button("Bắt đầu xử lý", variant="primary")
            
        with gr.Column():
            output_video = gr.Video(label="Trình phát Video (Có phụ đề)")
            output_files = gr.File(label="Tải kết quả về máy")

    submit_btn.click(
        fn=process_media, 
        inputs=input_file, 
        outputs=[output_video, output_files]
    )

if __name__ == "__main__":
    demo.launch()