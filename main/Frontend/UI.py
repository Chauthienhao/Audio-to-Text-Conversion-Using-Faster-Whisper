#region 1. Import Thư Viện, Nạp DLLs & Định Tuyến Module
import os
import sys
import subprocess
import gradio as gr

# Nạp DLLs CUDA cho tiến trình
nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
if os.path.exists(nvidia_base):
    for sub in ["cublas", "cudnn"]:
        bin_dir = os.path.join(nvidia_base, sub, "bin")
        if os.path.exists(bin_dir):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

# Thêm thư mục gốc vào path để import module backend
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from backend.app.process import transcript_audio, save
#endregion

#region 2. Điều Phối Xử Lý & Hậu Xử Lý Gán Cứng Phụ Đề (FFmpeg)
def process_media(file_path):
    if not file_path:
        return None, []
        
    ext = os.path.splitext(file_path)[-1].lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
    is_video = ext in video_extensions
    
    srt_filename = os.path.join(current_dir, "phude_ketqua.srt")
    output_video_filename = os.path.join(current_dir, "video_cophude.mp4")

    # Nhận diện âm thanh trực tiếp từ video/audio
    segments = transcript_audio(input=file_path)
    save(segments, srt_filename)

    # Rẽ nhánh hậu xử lý
    if is_video:
        # Hậu xử lý bằng FFmpeg: Ép cứng phụ đề vào khung hình video
        escaped_srt = srt_filename.replace("\\", "/").replace(":", "\\:")
        
        ffmpeg_cmd = [
            "ffmpeg", "-y", 
            "-i", file_path, 
            "-vf", f"subtitles='{escaped_srt}'",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-preset", "fast",
            output_video_filename
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            return output_video_filename, [output_video_filename, srt_filename]
        except subprocess.CalledProcessError as e:   
            print(f"Lỗi khi gắn phụ đề bằng FFmpeg: {e}")
            print(e.stderr)
            return None, [srt_filename]          
    else:
        # File âm thanh thuần: chỉ trả về tệp phụ đề .srt
        return None, [srt_filename]
#endregion

#region 3. Giao Diện Người Dùng Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎙 Giao Diện Nhận Dạng Giọng Nói & Tự Động Gắn Phụ Đề (Faster-Whisper)")
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Tải lên Video hoặc Audio", type="filepath")
            submit_btn = gr.Button("Bắt đầu xử lý", variant="primary")
            
        with gr.Column():
            output_video = gr.Video(label="Trình phát Video (Đã gắn phụ đề)")
            output_files = gr.File(label="Tải kết quả về máy (Video / File SRT)")

    submit_btn.click(
        fn=process_media, 
        inputs=input_file, 
        outputs=[output_video, output_files]
    )

if __name__ == "__main__":
    demo.launch()
#endregion