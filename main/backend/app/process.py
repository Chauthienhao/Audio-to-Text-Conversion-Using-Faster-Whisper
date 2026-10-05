#region 1. Import Thư Viện & Cấu Hình DLLs CUDA
import os
import sys
import re
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

#region 2. Tiền Xử Lý Văn Bản & Ngắt Dòng Phụ Đề
def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)

def preprocess_transcript(segments: list):
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
                
                # Ngắt dòng khi gặp dấu câu hoặc đạt số từ tối đa
                if has_punctuation or len(current_chunk) >= max_words_per_line:
                    processed_segments.append({
                        'start': chunk_start,
                        'end': word.end,
                        'text': clean_space(" ".join(current_chunk)).strip()
                    })
                    current_chunk = []
                    chunk_start = None
            
            # Lưu những từ còn dư ở cuối đoạn
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

#region 3. Khởi Tạo Mô Hình & Suy Luận Nhận Dạng (ASR Core)
def transcript_audio(
        input: str = "video.mp4",
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "int8",
        beam_size: int = 3,
        vad_filter: bool = True):
    if not os.path.exists(input):
        raise FileNotFoundError(f"Tệp tin {input} không tồn tại.")
        
    # Khởi tạo mô hình Faster-Whisper
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Cấu hình tham số nhận dạng
    transcript_kwargs = {
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "word_timestamps": True
    }

    # Suy luận gom mẻ (Batched Inference)
    batched_pipeline = BatchedInferencePipeline(model=model)
    segments, _ = batched_pipeline.transcribe(input, **transcript_kwargs, batch_size=8)
    
    segments = list(segments)
    processed_segments = preprocess_transcript(segments)
    return processed_segments
#endregion

#region 4. Định Dạng Thời Gian & Lưu Tệp SRT
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