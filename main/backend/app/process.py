import os 
from faster_whisper import WhisperModel, BatchedInferencePipeline

def preprocess_transcript(segments : list):
    return None

def transcript_audio(
        input: str = "video.mp4",
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int =5,
        vad_filter: bool = False):
    if not os.path.exists(input):
        raise FileNotFoundError(f"File {input} does not exist.")
    model = WhisperModel(model_size, device=device, compute_type=compute_type) #khởi tạo model

    # Cấu hình cho tham số
    transcript_kwargs = {"beam_size": beam_size, "vad_filter": vad_filter}

    # Chạy transcription
    batched_pipeline = BatchedInferencePipeline(model=model)
    segments, _ = batched_pipeline.transcribe(input, **transcript_kwargs, batch_size=16)
    segments = preprocess_transcript(segments)
    processed_segments = preprocess_transcript(segments)
    return processed_segments