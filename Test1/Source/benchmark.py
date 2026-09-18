import os
import time
import tracemalloc
import whisper
from faster_whisper import WhisperModel
from pywhispercpp.model import Model as WhisperCppModel

MODEL_SIZE = "base"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Samples"))

TEST_CASES = [
    {"group": "Clean", "file": "VIVOSSPK01_R004.wav", "lang": "vi"},
    {"group": "Clean", "file": "VIVOSSPK01_R005.wav", "lang": "vi"},
    {"group": "Clean", "file": "VIVOSSPK02_R002.wav", "lang": "vi"},
    {"group": "Clean", "file": "VIVOSSPK08_R001.wav", "lang": "vi"},
    {"group": "Noise", "file": "7-days-exploring-an-underground-city.wav", "lang": None},
    {"group": "Noise", "file": "escape-100-cops-win-500000.wav", "lang": None},
]

def run_benchmark():
    print("Dang nap mo hinh")
    model_orig = whisper.load_model(MODEL_SIZE, device="cpu")
    model_fast = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    model_cpp = WhisperCppModel(MODEL_SIZE, n_threads=4)

    print("\n" + "=" * 90)
    print(f"{'Nhom':<7} | {'Ten file test':<40} | {'Mo hinh':<15} | {'Thoi gian':<10} | {'Dinh RAM':<10}")
    print("=" * 90)

    for item in TEST_CASES:
        group = item["group"]
        filename = item["file"]
        lang = item["lang"]
        filepath = os.path.join(SAMPLES_DIR, group, filename)

        if not os.path.exists(filepath):
            print(f"Bo qua (khong tim thay): {filepath}")
            continue

        tracemalloc.start()
        t0 = time.time()
        res_orig = model_orig.transcribe(filepath, language=lang)
        t_orig = time.time() - t0
        _, mem_orig = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        text_orig = res_orig["text"].strip().replace("\n", " ")

        tracemalloc.start()
        t1 = time.time()
        segs, _ = model_fast.transcribe(filepath, language=lang, vad_filter=False) #Có thể để True nếu muốn lọc tiếng ồn
        text_fast = " ".join([s.text for s in segs]).strip().replace("\n", " ")
        t_fast = time.time() - t1
        _, mem_fast = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        t2 = time.time()
        cpp_lang = lang if lang else "auto"
        audio_16k = whisper.load_audio(filepath)
        segs_cpp = model_cpp.transcribe(audio_16k, language=cpp_lang)
        text_cpp = " ".join([seg.text for seg in segs_cpp]).strip().replace("\n", " ")
        t_cpp = time.time() - t2
        _, mem_cpp = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{group:<7} | {filename:<40} | {'OpenAI':<15} | {t_orig:<8.2f}s | {mem_orig/(1024*1024):<8.1f}MB")
        print(f"{'':<7} | {'':<40} | {'Faster-Whisper':<15} | {t_fast:<8.2f}s | {mem_fast/(1024*1024):<8.1f}MB")
        print(f"{'':<7} | {'':<40} | {'Whisper.cpp':<15} | {t_cpp:<8.2f}s | {mem_cpp/(1024*1024):<8.1f}MB")
        print(f"   [OpenAI Text]        : {text_orig}")
        print(f"   [Faster-Whisper Text]: {text_fast}")
        print(f"   [Whisper.cpp Text]   : {text_cpp}")
        print("-" * 90)

if __name__ == "__main__":
    run_benchmark()