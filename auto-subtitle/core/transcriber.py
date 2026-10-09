from faster_whisper import WhisperModel
from .config import MODEL_SIZE,DEVICE,COMPUTE_TYPE
import time
_model = None

def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE,device=DEVICE,compute_type=COMPUTE_TYPE)
    return _model

def transcribe(video_path,language=None,on_progress=None):
        t0 = time.perf_counter()
        model = get_model()
        t_load = time.perf_counter()
        segments,info = model.transcribe(video_path,
                                         language=language,
                                         beam_size=5,
                                         vad_filter=True,
                                         condition_on_previous_text=False,
                                         word_timestamps=True
                                         )
        t_prep = time.perf_counter()
        results = []
        for seg in segments:
            results.append({"start": seg.start, "end": seg.end, "text": seg.text})
            if on_progress and info.duration>0:
                on_progress(min(seg.end/info.duration,1.0))
        t_done = time.perf_counter()
        infer_time = t_done - t_prep
        print(f"Load model:      {t_load - t0:.2f}s")
        print(f"prep audio: {t_prep - t_load:.2f}s")
        print(f"reg:      {infer_time:.2f}s")
        if info.duration > 0:
            print(f"speed: {info.duration / infer_time:.1f}x realtime (length: {info.duration:.0f}s)")
        return results, info.language