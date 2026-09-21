import os 
from faster_whisper import WhisperModel, BatchedInferencePipeline
import re

def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

def preprocess_transcript(segments : list):
    processed_segments = []
    max_words_per_line = 20
    split_punctuations = {'.','!','?',';',':'}  # Regular expression to split on punctuation
    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            current_chunk = []
            chunk_start = None
            
            for word in segment.words:
                if chunk_start is None:
                    chunk_start = word.start # Lấy thời gian bắt đầu của từ đầu tiên
                
                clean_word = word.word.strip()
                current_chunk.append(clean_word)
                
                # Kiểm tra từ có chứa dấu câu ở cuối hay không (vd: "nhé.", "đi,")
                has_punctuation = any(p in clean_word for p in split_punctuations)
                
                # Ngắt dòng nếu: Gặp dấu câu HOẶC độ dài đã đạt mức tối đa
                if has_punctuation or len(current_chunk) >= max_words_per_line:
                    processed_segments.append({
                        'start': chunk_start,
                        'end': word.end,
                        'text': clean_space(" ".join(current_chunk)).strip()
                    })
                    # Reset lại để chuẩn bị cho dòng tiếp theo
                    current_chunk = []
                    chunk_start = None
            
            # Lưu lại những từ còn dư ở cuối đoạn chưa được đóng thành dòng
            if current_chunk:
                processed_segments.append({
                    'start': chunk_start,
                    'end': segment.words[-1].end,
                    'text': clean_space(" ".join(current_chunk)).strip()
                })
        else:
            # Chạy dự phòng (fallback) nếu không bật word_timestamps
            processed_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': clean_space(segment.text).strip()
            })
    return processed_segments

def transcript_audio(
        input: str = "video.mp4",
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int =3,
        vad_filter: bool = True):
    if not os.path.exists(input):           # Kiểm tra xem file có tồn tại không
        raise FileNotFoundError(f"File {input} does not exist.")
    model = WhisperModel(model_size, device=device, compute_type=compute_type) #khởi tạo model

    # Cấu hình cho tham số
    transcript_kwargs = {"beam_size": beam_size, "vad_filter": vad_filter, "word_timestamps": True}

    # Chạy transcription
    batched_pipeline = BatchedInferencePipeline(model=model)
    segments, _ = batched_pipeline.transcribe(input, **transcript_kwargs, batch_size=8)
    segments = list(segments)
    processed_segments = preprocess_transcript(segments)
    return processed_segments

# Hàm quy đổi giây sang chuẩn SRT bắt buộc để FFmpeg có thể đọc và ép chữ vào video
def format_time(seconds):
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