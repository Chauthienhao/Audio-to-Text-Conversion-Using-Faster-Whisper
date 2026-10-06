#region 1. Import Thư Viện & Cấu Hình DLLs CUDA
import os
import sys
import re
import subprocess
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Tự động nạp thư viện CUDA DLLs (cublas, cudnn) để chạy được trên GPU
nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
if os.path.exists(nvidia_base):
    for sub in ["cublas", "cudnn"]:
        bin_dir = os.path.join(nvidia_base, sub, "bin")
        if os.path.exists(bin_dir):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
#endregion

#region 2. Tiền Xử Lý: Bóc Tách & Chuẩn Hóa Âm Thanh Bằng FFmpeg
def extract_and_preprocess_audio(input_media_path: str, output_wav_path: str = "temp_preprocessed.wav") -> str:
    """
    Tiền xử lý đầu vào: Sử dụng FFmpeg để trích xuất âm thanh từ video/audio bất kỳ,
    chuẩn hóa về định dạng tối ưu nhất cho mô hình Whisper:
    - 1 kênh (Mono: -ac 1)
    - Tần số lấy mẫu chuẩn (16000Hz: -ar 16000)
    - Định dạng PCM 16-bit không nén (-c:a pcm_s16le)
    """
    if not os.path.exists(input_media_path):
        raise FileNotFoundError(f"Tệp tin {input_media_path} không tồn tại.")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_media_path,
        "-vn",                      # Bỏ luồng hình ảnh, chỉ lấy âm thanh
        "-ac", "1",                 # Ép về 1 kênh Mono để giảm tải tính toán
        "-ar", "16000",             # Ép chuẩn tần số 16kHz chuẩn cho Whisper
        "-c:a", "pcm_s16le",        # Xuất định dạng âm thanh WAV chuẩn
        output_wav_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        return output_wav_path
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi tiền xử lý âm thanh bằng FFmpeg: {e}")
        # Nếu có lỗi phát sinh, fallback sử dụng lại file gốc
        return input_media_path
#endregion

#region 3. Hậu Xử Lý Văn Bản & Ngắt Dòng Phụ Đề
def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)

def chunking_segments(segments: list):
    processed_segments = []
    max_words_per_line = 20
    split_punctuations = {'.', '!', '?', ';', ':'}
    
    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            current_chunk = []
            chunk_start = None
            
            for word in segment.words:
                if chunk_start is None:
                    chunk_start = word.start
                
                clean_word = word.word.strip()
                current_chunk.append(clean_word)
                
                has_punctuation = any(p in clean_word for p in split_punctuations)
                
                if has_punctuation or len(current_chunk) >= max_words_per_line:
                    processed_segments.append({
                        'start': chunk_start,
                        'end': word.end,
                        'text': clean_space(" ".join(current_chunk)).strip()
                    })
                    current_chunk = []
                    chunk_start = None
            
            if current_chunk:
                processed_segments.append({
                    'start': chunk_start,
                    'end': segment.words[-1].end,
                    'text': clean_space(" ".join(current_chunk)).strip()
                })
        else:
            processed_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': clean_space(segment.text).strip()
            })
    return processed_segments
#endregion

#region 4. Khởi Tạo Mô Hình & Suy Luận Nhận Dạng (ASR Core)
def transcript_audio(
        input: str = "video.mp4",
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 3,
        vad_filter: bool = True):
    if not os.path.exists(input):
        raise FileNotFoundError(f"Tệp tin {input} không tồn tại.")
        
    # BƯỚC TIỀN XỬ LÝ: Dùng FFmpeg bóc tách âm thanh sang chuẩn 16kHz Mono trước khi đưa vào AI
    temp_audio = "temp_preprocessed.wav"
    audio_for_model = extract_and_preprocess_audio(input, temp_audio)

    # Khởi tạo mô hình Faster-Whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    transcript_kwargs = {
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "word_timestamps": True
    }

    # Đưa file âm thanh đã tiền xử lý vào pipeline gom mẻ
    batched_pipeline = BatchedInferencePipeline(model=model)
    segments, _ = batched_pipeline.transcribe(audio_for_model, **transcript_kwargs, batch_size=8)
    
    segments = list(segments)
    processed_segments = chunking_segments(segments)

    # Dọn dẹp tệp âm thanh tạm sau khi nhận dạng xong
    if os.path.exists(temp_audio):
        try:
            os.remove(temp_audio)
        except Exception:
            pass

    return processed_segments
#endregion

#region 5. Định Dạng Thời Gian & Lưu Tệp SRT
def format_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msec = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msec:03d}"

def save(segments: list, output: str): 
    with open(output, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")
#endregion