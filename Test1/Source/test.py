from faster_whisper import BatchedInferencePipeline, WhisperModel
import os
import re

model = WhisperModel("base", device="cpu", compute_type="int8")
batched_model = BatchedInferencePipeline(model=model)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Samples"))

TEST_CASES = [
    {"group": "Noise", "file": "bucky_and_steve_at_the_stark_expo__captain_america_the_first_avenger.mp4_v1.mp4", "lang": "en"},
]
for item in TEST_CASES:
        group = item["group"]
        filename = item["file"]
        lang = item["lang"]
        filepath = os.path.join(SAMPLES_DIR, group, filename)
segments, info = batched_model.transcribe(filepath, word_timestamps=True,vad_filter=True)
segments = list(segments)

print("Segmentation mặc định của Faster Whisper:")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

def clean_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)

def chunk_segments(segments: list):
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

# Chunk the segments
processed_segments = chunk_segments(segments)
print("\nSau khi xử lý và ngắt dòng phụ đề:")
# Print the processed segments
for segment in processed_segments:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")