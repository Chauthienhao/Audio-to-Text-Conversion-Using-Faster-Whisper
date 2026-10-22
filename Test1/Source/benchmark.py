"""
Benchmark 3 bien the Whisper tren tap VIVOS (test set):
  1. Whisper goc cua OpenAI - chay LOCAL qua PyTorch (khong goi API cloud)
  2. Faster-Whisper (CTranslate2)
  3. Whisper.cpp (qua pywhispercpp)

So voi ban cu, file nay sua 3 loi phuong phap da bi thay nhan xet:

  (A) "OpenAI Whisper" trong bang 1.3 KHONG phai API cloud - no la
      `whisper.load_model()` chay local bang PyTorch, giong dinh nghia
      "Whisper goc" o muc 1.1. Script nay KHONG goi API thuc (khong can
      API key, khong can mang) - chi can doi lai nhan trong bao cao
      thanh "Whisper goc (OpenAI, chay local qua PyTorch)".

  (B) tracemalloc CHI thay bo nho cap phat o tang Python, khong thay duoc
      bo nho ma CTranslate2 / whisper.cpp / PyTorch tu cap phat ben duoi
      (C/C++/CUDA) -> giai thich vi sao so lieu cu chi 3-31MB (vo ly).
      Script nay doi sang do RSS (Resident Set Size) cua toan tien trinh
      bang psutil, lay delta NGAY TRUOC/SAU KHI NAP MODEL (khong do luc
      transcribe) - day moi la "bo nho model su dung" dung nghia.

  (C) Khong co WER/CER thuc, chi noi "dao dong". Script nay doc ground
      truth tu prompts.txt cua VIVOS, chuan hoa van ban (NFC, lower,
      bo dau cau) va tinh WER/CER bang jiwer, tren CA TAP MAU VIVOS
      (khong chi 4 file cu dinh) - tinh WER/CER o muc corpus (ghep toan
      bo tham chieu/giai ma lai) de dung ve mat thong ke, khong lay
      trung binh don gian cua WER tung file.

  (D) Ghi ro cau hinh (model_size / device / compute_type) ngay trong
      tung dong ket qua, khong de nguoi doc phai doan.

Cai dat truoc khi chay:
    pip install psutil jiwer openai-whisper faster-whisper pywhispercpp

Cau truc du lieu VIVOS mong doi (tap test chuan tu Kaggle/VIVOS):
    <VIVOS_ROOT>/waves/<speaker_id>/<file_id>.wav
    <VIVOS_ROOT>/prompts.txt   (moi dong: "<file_id> <CAU THAM CHIEU>")

Neu ban da tach rieng file .wav ra 1 thu muc "Samples/Clean/..." (khong
con cau truc <speaker_id>/ long nhau) thi van chay duoc - script tu do
glob("**/*.wav", recursive=True) va chi can ten file (khong ke duong dan)
khop voi cot dau cua prompts.txt.
"""

import os
import re
import csv
import glob
import time
import random
import unicodedata

import psutil

try:
    import jiwer
except ImportError:
    raise SystemExit("Thieu thu vien jiwer. Chay: pip install jiwer")

import whisper
from faster_whisper import WhisperModel
from pywhispercpp.model import Model as WhisperCppModel


# ============================= CAU HINH =====================================

MODEL_SIZE = "base"

# --- Duong dan VIVOS (SUA LAI CHO DUNG MAY BAN) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIVOS_WAVES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Samples", "vivos", "test", "waves"))
VIVOS_PROMPTS_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Samples", "vivos", "test", "prompts.txt"))

# Neu khong tim thay cau truc chuan tren, thu fallback ve thu muc Samples/Clean
# (cau truc phang ban da dung truoc do).
FALLBACK_SAMPLES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Samples", "Clean"))

# So file VIVOS se test. De None = chay TOAN BO tap test (co the rat lau tren CPU).
# Nen bat dau voi so nho (30-50) de kiem tra script chay dung, roi tang dan cho
# so lieu bao cao cuoi cung. Dat seed co dinh de tap mau co the lap lai duoc.
MAX_VIVOS_FILES = 50
RANDOM_SEED = 42

VAD_FILTER = False  # giu nguyen lua chon cua ban: khong loc VAD khi benchmark

# Cac file "Noise" khong co ground truth (giu lai de xem chat luong dinh tinh,
# KHONG tinh duoc WER/CER vi khong co transcript tham chieu).
NOISE_SAMPLES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "Samples", "Noise"))
NOISE_TEST_CASES = [
    {"file": "7-days-exploring-an-underground-city.wav", "lang": None},
    {"file": "escape-100-cops-win-500000.wav", "lang": None},
]

OUT_DIR = CURRENT_DIR
DETAIL_CSV = os.path.join(OUT_DIR, "vivos_detailed_results.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "vivos_summary.csv")


# ============================ TIEN ICH ======================================

def normalize_vi(text: str) -> str:
    """Chuan hoa van ban tieng Viet truoc khi tinh WER/CER:
    - NFC unicode (tranh dau thanh bi tach to hop khac voi ban tham chieu)
    - ha chu thuong
    - bo dau cau, giu lai chu/so/khoang trang
    - gon khoang trang thua
    Luu y (han che): khong chuan hoa so viet bang chu vs. chu so
    (vd model doc "5" nhung prompts.txt ghi "nam") - day la nguon WER
    con lai ngoai tam kiem soat cua script, can ghi chu trong bao cao.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def measure_rss_mb() -> float:
    """RSS (Resident Set Size) hien tai cua tien trinh, don vi MB.
    Day la bo nho THAT ma OS da cap cho tien trinh nay, bao gom ca
    phan cap phat trong cac thu vien C/C++/CUDA (CTranslate2, whisper.cpp,
    PyTorch) - khac voi tracemalloc chi thay tang Python."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_vivos_prompts(prompts_path: str) -> dict:
    """Doc prompts.txt cua VIVOS: moi dong 'file_id CAU THAM CHIEU...'."""
    prompts = {}
    if not os.path.exists(prompts_path):
        return prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            file_id, ref_text = parts
            prompts[file_id] = ref_text
    return prompts


def find_vivos_items(waves_dir: str, fallback_dir: str, prompts: dict, max_files, seed):
    """Tim tat ca file .wav co trong prompts.txt, tra ve list (file_id, filepath, ref_text)."""
    search_dirs = [d for d in [waves_dir, fallback_dir] if os.path.isdir(d)]
    items = []
    seen_ids = set()
    for base_dir in search_dirs:
        for filepath in glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True):
            file_id = os.path.splitext(os.path.basename(filepath))[0]
            if file_id in seen_ids:
                continue
            ref_text = prompts.get(file_id)
            if ref_text is None:
                continue  # khong co ground truth -> khong dung de tinh WER/CER
            items.append((file_id, filepath, ref_text))
            seen_ids.add(file_id)

    items.sort(key=lambda x: x[0])  # thu tu on dinh truoc khi sample

    if max_files is not None and len(items) > max_files:
        rng = random.Random(seed)
        items = rng.sample(items, max_files)
        items.sort(key=lambda x: x[0])

    return items


def load_model_with_mem(label: str, load_fn):
    """Nap 1 model, do RSS truoc/sau de biet model do chiem them bao nhieu MB."""
    mem_before = measure_rss_mb()
    t0 = time.time()
    model = load_fn()
    load_time = time.time() - t0
    mem_after = measure_rss_mb()
    delta_mb = mem_after - mem_before
    print(f"  [{label}] nap xong trong {load_time:.2f}s | RAM tang them: {delta_mb:.1f} MB "
          f"(RSS {mem_before:.0f} -> {mem_after:.0f} MB)")
    return model, delta_mb


# ============================== MAIN =========================================

def run_benchmark():
    print(f"Cau hinh dung chung: model_size={MODEL_SIZE}, device=cpu, vad_filter={VAD_FILTER}")
    print("Dang nap 3 model (do RAM tang them cho tung model ngay sau khi nap)...")

    model_orig, mem_model_orig = load_model_with_mem(
        "Whisper goc (PyTorch, local)",
        lambda: whisper.load_model(MODEL_SIZE, device="cpu"),
    )
    model_fast, mem_model_fast = load_model_with_mem(
        "Faster-Whisper (CTranslate2, int8)",
        lambda: WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8"),
    )
    model_cpp, mem_model_cpp = load_model_with_mem(
        "Whisper.cpp (pywhispercpp)",
        lambda: WhisperCppModel(MODEL_SIZE, n_threads=4),
    )

    engine_configs = {
        "Whisper goc (PyTorch, local)": f"{MODEL_SIZE}/cpu/fp32",
        "Faster-Whisper": f"{MODEL_SIZE}/cpu/int8",
        "Whisper.cpp": f"{MODEL_SIZE}/cpu/ggml",
    }
    engine_model_mem = {
        "Whisper goc (PyTorch, local)": mem_model_orig,
        "Faster-Whisper": mem_model_fast,
        "Whisper.cpp": mem_model_cpp,
    }

    # ---------------- Tap VIVOS co ground truth: tinh WER/CER thuc ----------------
    prompts = load_vivos_prompts(VIVOS_PROMPTS_PATH)
    items = find_vivos_items(VIVOS_WAVES_DIR, FALLBACK_SAMPLES_DIR, prompts, MAX_VIVOS_FILES, RANDOM_SEED)

    if not items:
        print("\n[CANH BAO] Khong tim thay file VIVOS nao khop voi prompts.txt.")
        print(f"  Da tim trong: {VIVOS_WAVES_DIR}")
        print(f"  va fallback : {FALLBACK_SAMPLES_DIR}")
        print(f"  prompts.txt : {VIVOS_PROMPTS_PATH} (ton tai: {os.path.exists(VIVOS_PROMPTS_PATH)})")
        print("  -> Sua lai VIVOS_WAVES_DIR / VIVOS_PROMPTS_PATH o dau file roi chay lai.")
    else:
        print(f"\nSe benchmark tren {len(items)} file VIVOS co ground truth "
              f"(seed={RANDOM_SEED}, max_files={MAX_VIVOS_FILES}).")

    detail_rows = []
    refs_norm, hyps_orig_norm, hyps_fast_norm, hyps_cpp_norm = [], [], [], []
    times_orig, times_fast, times_cpp = [], [], []

    print("\n" + "=" * 100)
    print(f"{'File':<22} | {'Engine':<30} | {'Config':<14} | {'Thoi gian':<10}")
    print("=" * 100)

    for idx, (file_id, filepath, ref_text) in enumerate(items, 1):
        ref_norm = normalize_vi(ref_text)

        t0 = time.time()
        res_orig = model_orig.transcribe(filepath, language="vi")
        t_orig = time.time() - t0
        text_orig = res_orig["text"].strip().replace("\n", " ")

        t1 = time.time()
        segs, _ = model_fast.transcribe(filepath, language="vi", vad_filter=VAD_FILTER)
        text_fast = " ".join(s.text for s in segs).strip().replace("\n", " ")
        t_fast = time.time() - t1

        t2 = time.time()
        audio_16k = whisper.load_audio(filepath)
        segs_cpp = model_cpp.transcribe(audio_16k, language="vi")
        text_cpp = " ".join(seg.text for seg in segs_cpp).strip().replace("\n", " ")
        t_cpp = time.time() - t2

        hyp_orig_norm = normalize_vi(text_orig)
        hyp_fast_norm = normalize_vi(text_fast)
        hyp_cpp_norm = normalize_vi(text_cpp)

        refs_norm.append(ref_norm)
        hyps_orig_norm.append(hyp_orig_norm)
        hyps_fast_norm.append(hyp_fast_norm)
        hyps_cpp_norm.append(hyp_cpp_norm)
        times_orig.append(t_orig)
        times_fast.append(t_fast)
        times_cpp.append(t_cpp)

        print(f"{file_id:<22} | {'Whisper goc (PyTorch)':<30} | {engine_configs['Whisper goc (PyTorch, local)']:<14} | {t_orig:<8.2f}s")
        print(f"{'':<22} | {'Faster-Whisper':<30} | {engine_configs['Faster-Whisper']:<14} | {t_fast:<8.2f}s")
        print(f"{'':<22} | {'Whisper.cpp':<30} | {engine_configs['Whisper.cpp']:<14} | {t_cpp:<8.2f}s")
        print(f"   [Tham chieu]        : {ref_text}")
        print(f"   [Whisper goc]       : {text_orig}")
        print(f"   [Faster-Whisper]    : {text_fast}")
        print(f"   [Whisper.cpp]       : {text_cpp}")
        print("-" * 100)

        detail_rows.append({
            "file_id": file_id, "reference": ref_text,
            "engine": "Whisper goc (PyTorch, local)", "config": engine_configs["Whisper goc (PyTorch, local)"],
            "hypothesis": text_orig, "time_s": round(t_orig, 3),
            "wer": round(jiwer.wer(ref_norm, hyp_orig_norm), 4) if ref_norm else None,
            "cer": round(jiwer.cer(ref_norm, hyp_orig_norm), 4) if ref_norm else None,
        })
        detail_rows.append({
            "file_id": file_id, "reference": ref_text,
            "engine": "Faster-Whisper", "config": engine_configs["Faster-Whisper"],
            "hypothesis": text_fast, "time_s": round(t_fast, 3),
            "wer": round(jiwer.wer(ref_norm, hyp_fast_norm), 4) if ref_norm else None,
            "cer": round(jiwer.cer(ref_norm, hyp_fast_norm), 4) if ref_norm else None,
        })
        detail_rows.append({
            "file_id": file_id, "reference": ref_text,
            "engine": "Whisper.cpp", "config": engine_configs["Whisper.cpp"],
            "hypothesis": text_cpp, "time_s": round(t_cpp, 3),
            "wer": round(jiwer.wer(ref_norm, hyp_cpp_norm), 4) if ref_norm else None,
            "cer": round(jiwer.cer(ref_norm, hyp_cpp_norm), 4) if ref_norm else None,
        })

        if idx % 10 == 0:
            print(f"... da xong {idx}/{len(items)} file ...")

    # ---------------- WER/CER o muc CORPUS (dung ve thong ke, khong phai trung binh) ----------------
    summary_rows = []
    if refs_norm:
        for engine, hyps, times in [
            ("Whisper goc (PyTorch, local)", hyps_orig_norm, times_orig),
            ("Faster-Whisper", hyps_fast_norm, times_fast),
            ("Whisper.cpp", hyps_cpp_norm, times_cpp),
        ]:
            corpus_wer = jiwer.wer(refs_norm, hyps)
            corpus_cer = jiwer.cer(refs_norm, hyps)
            avg_time = sum(times) / len(times)
            summary_rows.append({
                "engine": engine,
                "config": engine_configs[engine],
                "n_files": len(items),
                "avg_time_s": round(avg_time, 3),
                "corpus_wer_percent": round(corpus_wer * 100, 2),
                "corpus_cer_percent": round(corpus_cer * 100, 2),
                "model_load_mem_mb": round(engine_model_mem[engine], 1),
            })

        print("\n" + "=" * 100)
        print("TONG KET (tren tap mau VIVOS co ground truth) - dung de dien bang 1.3")
        print("=" * 100)
        print(f"{'Engine':<30} | {'Config':<14} | {'N':<4} | {'TB thoi gian':<12} | {'WER':<8} | {'CER':<8} | {'RAM model':<10}")
        for r in summary_rows:
            print(f"{r['engine']:<30} | {r['config']:<14} | {r['n_files']:<4} | "
                  f"{r['avg_time_s']:<10.2f}s | {r['corpus_wer_percent']:<6.2f}% | "
                  f"{r['corpus_cer_percent']:<6.2f}% | {r['model_load_mem_mb']:<8.1f}MB")

    # ---------------- Tap Noise (khong co ground truth -> chi dinh tinh) ----------------
    print("\n" + "=" * 100)
    print("Tap 'Noise' (khong co transcript tham chieu - chi xem dinh tinh, KHONG tinh WER/CER)")
    print("=" * 100)
    for item in NOISE_TEST_CASES:
        filepath = os.path.join(NOISE_SAMPLES_DIR, item["file"])
        if not os.path.exists(filepath):
            print(f"Bo qua (khong tim thay): {filepath}")
            continue
        lang = item["lang"]

        t0 = time.time()
        res_orig = model_orig.transcribe(filepath, language=lang)
        t_orig = time.time() - t0
        text_orig = res_orig["text"].strip().replace("\n", " ")

        t1 = time.time()
        segs, _ = model_fast.transcribe(filepath, language=lang, vad_filter=VAD_FILTER)
        text_fast = " ".join(s.text for s in segs).strip().replace("\n", " ")
        t_fast = time.time() - t1

        t2 = time.time()
        cpp_lang = lang if lang else "auto"
        audio_16k = whisper.load_audio(filepath)
        segs_cpp = model_cpp.transcribe(audio_16k, language=cpp_lang)
        text_cpp = " ".join(seg.text for seg in segs_cpp).strip().replace("\n", " ")
        t_cpp = time.time() - t2

        print(f"{item['file']}:")
        print(f"  Whisper goc    : {t_orig:.2f}s | {text_orig}")
        print(f"  Faster-Whisper : {t_fast:.2f}s | {text_fast}")
        print(f"  Whisper.cpp    : {t_cpp:.2f}s | {text_cpp}")
        print("-" * 100)

    # ---------------- Xuat CSV ----------------
    if detail_rows:
        with open(DETAIL_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"\nDa luu chi tiet tung file: {DETAIL_CSV}")

    if summary_rows:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Da luu bang tong ket: {SUMMARY_CSV}")


if __name__ == "__main__":
    run_benchmark()