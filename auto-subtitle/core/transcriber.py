from faster_whisper import WhisperModel, BatchedInferencePipeline
from .config import MODEL_SIZE,DEVICE,COMPUTE_TYPE
import time
import re
_model = None

def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE,device=DEVICE,compute_type=COMPUTE_TYPE)
    return _model

def transcribe(video_path,language=None,on_progress=None):
        t0 = time.perf_counter()
        model = get_model()
        batched_model = BatchedInferencePipeline(model=model)
        t_load = time.perf_counter()
        segments,info = batched_model.transcribe(video_path,
                                         language=language,
                                         beam_size=5,
                                         vad_filter=True,
                                         vad_parameters=dict(
                                            threshold=0.6,                 # mặc định 0.5, tăng để bớt nhận nhầm nhạc là tiếng nói
                                            min_silence_duration_ms=2000,
                                            speech_pad_ms=300,
                                        ),
                                         condition_on_previous_text=False,
                                         word_timestamps=True
                                         )
        #Chạy qua vòng lặp để lấy tiến độ % cho giao diện
        t_prep = time.perf_counter()
        raw_segments = []
        for seg in segments:
            raw_segments.append(seg)
            if on_progress and info.duration>0:
                on_progress(min(seg.end/info.duration,1.0))
        t_done = time.perf_counter()

        #Chunking lại dữ liệu thô đã lặp ở trên
        processed_segments = chunking_segments(raw_segments)
        infer_time = t_done - t_prep
        print(f"Load model:      {t_load - t0:.2f}s")
        print(f"prep audio: {t_prep - t_load:.2f}s")
        print(f"reg:      {infer_time:.2f}s")
        if info.duration > 0:
            print(f"speed: {info.duration / infer_time:.1f}x realtime (length: {info.duration:.0f}s)")
        return processed_segments, info.language

#Gộp mọi khoảng trắng liên tiếp (dấu cách, tab, xuống dòng) thành 1 dấu cách
def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)


#Function dùng để hậu xử lý segments
#mốc thời gian lấy theo từng từ (cần word_timestamps=True khi transcribe)
def chunking_segments(segments: list):
    processed_segments = []     # kết quả: list các dòng {'start', 'end', 'text'}
    max_words_per_line = 20
    max_char = 84       #~2 dòng x 42 ký tự
    max_dur = 6.0       #thời lượng tối đa
    split_punctuations = ('.', '!', '?', ';')

    # Chốt một dòng phụ đề và thêm vào kết quả
    def flush(chunk, start, end):
        processed_segments.append({
            'start': start,
            'end': end,
            'text': clean_space(" ".join(chunk)).strip()
        })
    
    for segment in segments:
        # Trường hợp chính: segment có danh sách từ kèm mốc thời gian
        if hasattr(segment, 'words') and segment.words:
            current_chunk = []  # các từ của dòng đang gom
            chunk_start = None  # giờ bắt đầu dòng = word.start của từ đầu tiên
            prev_end = None     # giờ kết thúc của từ cuối cùng đã vào dòng
            
            for word in segment.words:
                clean_word = word.word.strip()
                # Bỏ từ rỗng để giờ của nó không làm lệch start/end của dòng
                if not clean_word:
                    continue

                # Token chỉ có dấu câu (vd "...") đứng đầu dòng mới:
                # nối chữ vào dòng trước, KHÔNG đổi giờ của dòng trước
                if not any(ch.isalnum() for ch in clean_word) and not current_chunk and processed_segments:
                    processed_segments[-1]['text'] += clean_word
                    continue

                # check số ký tự và thời lượng, nếu vượt thì chốt dòng hiện tại trước
                if current_chunk and (len(" ".join(current_chunk + [clean_word])) > max_char or word.end - chunk_start > max_dur):
                    flush(current_chunk, chunk_start, prev_end)
                    current_chunk = []
                    chunk_start = None

                # nếu chưa có giờ bắt đầu dòng thì lấy giờ của từ đầu tiên
                if chunk_start is None:
                    chunk_start = word.start
                current_chunk.append(clean_word)
                prev_end = word.end

                # Ngắt khi dấu câu nằm ở CUỐI từ (không cắt "3.5", "TP.HCM", "8:30")
                # hoặc khi đã đủ 20 từ
                if clean_word.endswith(split_punctuations) or len(current_chunk) >= max_words_per_line:
                    flush(current_chunk, chunk_start, word.end)
                    current_chunk = []
                    chunk_start = None

            # Chốt thành dòng cuối nếu hêt segment mà còn từ chưa chốt
            if current_chunk:
                flush(current_chunk, chunk_start, prev_end)

        # Dự phòng: segment không có words (vd quên bật word_timestamps)
        # giữ nguyên cả segment làm một dòng
        else:
            processed_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': clean_space(segment.text).strip()
            })
    return processed_segments