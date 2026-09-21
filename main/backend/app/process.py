import os 
from faster_whisper import WhisperModel, BatchedInferencePipeline
import re

def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

def preprocess_transcript(segments : list):
    processed_segments = []
    max_words_per_line = 12
    for segment in segments:
        # processed_segments.append({
        #     'start': segment.start,         # Start time of the segment
        #     'end': segment.end,         # End time of the segment
        #     'text': clean_space(segment.text).strip()         # Cleaned text of the segment
        # })
        # Kiểm tra xem AI có trả về mốc thời gian cho từng từ không
        if hasattr(segment, 'words') and segment.words:
            current_chunk = []
            chunk_start = None
            
            for word in segment.words:
                if chunk_start is None:
                    chunk_start = word.start # Lấy thời gian bắt đầu của từ đầu tiên trong dòng
                
                current_chunk.append(word.word.strip())
                
                # Cứ đủ 12 chữ là đóng thành một dòng phụ đề
                if len(current_chunk) >= max_words_per_line:
                    processed_segments.append({
                        'start': chunk_start,
                        'end': word.end, # Lấy thời gian kết thúc của từ cuối cùng
                        'text': " ".join(current_chunk)
                    })
                    current_chunk = []
                    chunk_start = None
            
            # Lưu lại những từ còn dư ở cuối câu chưa đủ 12 chữ
            if current_chunk:
                processed_segments.append({
                    'start': chunk_start,
                    'end': segment.words[-1].end,
                    'text': " ".join(current_chunk)
                })
        else:
            # Chạy dự phòng (fallback) nếu quên bật word_timestamps
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
        beam_size: int =5,
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