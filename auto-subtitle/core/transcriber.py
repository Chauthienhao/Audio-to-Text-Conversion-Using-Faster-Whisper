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
        t_prep = time.perf_counter()
        results = []
        processed_segments = chunking_segments(segments)
        # for seg in segments:
        #     results.append({"start": seg.start, "end": seg.end, "text": seg.text})
        #     if on_progress and info.duration>0:
        #         on_progress(min(seg.end/info.duration,1.0))
        t_done = time.perf_counter()
        infer_time = t_done - t_prep
        print(f"Load model:      {t_load - t0:.2f}s")
        print(f"prep audio: {t_prep - t_load:.2f}s")
        print(f"reg:      {infer_time:.2f}s")
        if info.duration > 0:
            print(f"speed: {info.duration / infer_time:.1f}x realtime (length: {info.duration:.0f}s)")
        return processed_segments, info.language

#Function cho việc dọn khoảng trắng
def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)


#Function dùng để hậu xử lý segments
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