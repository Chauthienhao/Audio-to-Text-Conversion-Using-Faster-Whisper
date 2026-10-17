"""Worker: chạy MỘT cấu hình (model, device, compute_type) trong một tiến trình riêng.
Dùng: python bench_worker.py payload.json   (do bench_common.run_config gọi)"""
import gc, json, os, re, statistics, sys, threading, time, traceback, unicodedata
from pathlib import Path
import psutil
from bench_metrics import derive_metrics

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
OUT = Path(cfg["output_json"])
T0 = time.time()
phase = {"name": "startup"}
samples = []
halt = threading.Event()
nvml = handle = None
gpu_error = None


def init_nvml():
    global nvml, handle, gpu_error
    if cfg["device"] != "cuda":
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(cfg.get("gpu_index", 0)))
        nvml = pynvml
    except Exception as e:
        gpu_error = repr(e)


def proc_gpu_mb(pid):
    """VRAM của riêng tiến trình qua NVML. Trên Windows (WDDM) thường trả None => dùng VRAM thiết bị trừ nền."""
    for fn in (nvml.nvmlDeviceGetComputeRunningProcesses, nvml.nvmlDeviceGetGraphicsRunningProcesses):
        try:
            for p in fn(handle):
                if p.pid == pid and p.usedGpuMemory is not None and p.usedGpuMemory < 2**40:
                    return p.usedGpuMemory / 2**20
        except Exception:
            pass
    return None


def sample_loop():
    pid = os.getpid()
    proc = psutil.Process(pid)
    proc.cpu_percent(None)  # gọi mồi: lần đầu luôn trả 0
    psutil.cpu_percent(None)
    while not halt.is_set():
        gpu = util = gpu_proc = None
        if nvml is not None:
            try:
                gpu = nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
                util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                pass
            gpu_proc = proc_gpu_mb(pid)
        samples.append({"t": time.time() - T0, "phase": phase["name"],
                        "cpu_pct": proc.cpu_percent(None),   # theo tiến trình, có thể > 100
                        "ram_mb": proc.memory_info().rss / 2**20,
                        "sys_cpu_pct": psutil.cpu_percent(None),            # toàn hệ thống, 0-100
                        "sys_ram_mb": psutil.virtual_memory().used / 2**20,
                        "gpu_mb": gpu, "gpu_util": util, "gpu_proc_mb": gpu_proc})
        halt.wait(float(cfg.get("sample_interval_sec", 0.2)))


def normalize_text(t):
    t = unicodedata.normalize("NFC", str(t or "")).lower()
    t = re.sub(r"[^\w\s]", " ", t)       # bỏ dấu câu, GIỮ dấu thanh
    t = t.replace("_", " ")
    return " ".join(t.split())


def transcribe_one(model, path):
    t = time.perf_counter()
    segments, info = model.transcribe(str(path), **cfg["transcribe_kwargs"])
    text = " ".join(s.text.strip() for s in segments)   # duyệt hết generator mới tính xong
    return text.strip(), time.perf_counter() - t, getattr(info, "duration", None)


def stats(s):
    idle = [x for x in s if x["phase"] == "idle"]
    run = [x for x in s if x["phase"] in ("model_load", "warmup", "evaluation", "long_audio")]
    ev = [x for x in s if x["phase"] == "evaluation"]
    out = {"sample_count": len(s)}
    if run:
        out["cpu_peak_pct"] = max(x["cpu_pct"] for x in run)
        out["ram_peak_mb"] = max(x["ram_mb"] for x in run)
        if idle:
            out["ram_idle_mb"] = statistics.mean(x["ram_mb"] for x in idle)
            out["ram_delta_peak_mb"] = out["ram_peak_mb"] - out["ram_idle_mb"]
    if ev:
        out["cpu_avg_pct_eval"] = statistics.mean(x["cpu_pct"] for x in ev)
        gu_ev = [x["gpu_util"] for x in ev if x["gpu_util"] is not None]
        if gu_ev:
            out["gpu_util_avg_eval_pct"] = statistics.mean(gu_ev)
            out["gpu_util_eval_n"] = len(gu_ev)
    g_run = [x["gpu_mb"] for x in run if x["gpu_mb"] is not None]
    g_idle = [x["gpu_mb"] for x in idle if x["gpu_mb"] is not None]
    if g_run and g_idle:
        out["vram_baseline_mb"] = statistics.median(g_idle)          # trung vị: bền với nhiễu từ ứng dụng khác
        out["vram_baseline_spread_mb"] = max(g_idle) - min(g_idle)    # nền dao động bao nhiêu trong lúc nghỉ
        out["vram_peak_mb"] = max(g_run)
        out["vram_delta_peak_mb"] = max(0.0, max(g_run) - out["vram_baseline_mb"])
        gu = [x["gpu_util"] for x in run if x["gpu_util"] is not None]
        out["gpu_util_peak_pct"] = max(gu) if gu else None
    out.update(derive_metrics(s, psutil.cpu_count()))
    return out


def run(result):
    phase["name"] = "startup"
    from faster_whisper import WhisperModel
    import faster_whisper, ctranslate2
    result["faster_whisper_version"] = getattr(faster_whisper, "__version__", "?")
    result["ctranslate2_version"] = getattr(ctranslate2, "__version__", "?")

    phase["name"] = "idle"
    time.sleep(float(cfg.get("idle_s", 3)))          # đo mức nền trước khi nạp model

    phase["name"] = "model_load"
    print("  đang nạp model ...", flush=True)
    kw = {"device": cfg["device"], "compute_type": cfg["compute_type"]}
    if cfg["device"] == "cuda":
        kw["device_index"] = int(cfg.get("gpu_index", 0))
    t = time.perf_counter()
    model = WhisperModel(cfg["model"], **kw)        # cố ý KHÔNG truyền cpu_threads (giữ mặc định)
    result["model_load_sec"] = time.perf_counter() - t
    try:
        result["effective_compute_type"] = str(model.model.compute_type)
    except Exception:
        result["effective_compute_type"] = None

    items = cfg["eval_items"]
    phase["name"] = "warmup"
    print(f"  nạp xong ({result['model_load_sec']:.1f}s), đang khởi động ...", flush=True)
    t = time.perf_counter()
    transcribe_one(model, cfg["warmup_audio"])
    result["warmup_sec"] = time.perf_counter() - t

    phase["name"] = "evaluation"
    refs, hyps, tot_audio = [], [], 0.0
    t_eval = time.perf_counter()
    for i, it in enumerate(items, 1):
        hyp, dt, dur = transcribe_one(model, it["audio"])
        tot_audio += dur or 0.0
        refs.append(normalize_text(it["reference"]))
        hyps.append(normalize_text(hyp))
        result["hypotheses"].append({"audio": it["audio"], "reference": it["reference"],
                                     "hypothesis": hyp, "time_sec": dt, "audio_sec": dur})
        if i % 10 == 0 or i == len(items):
            print(f"  đang đánh giá: {i}/{len(items)} file ({100 * i / len(items):.0f}%)", flush=True)
    result["eval_inference_sec"] = time.perf_counter() - t_eval
    result["eval_audio_sec"] = tot_audio
    result["rtf"] = result["eval_inference_sec"] / tot_audio if tot_audio > 0 else None
    result["n_eval"] = len(items)
    from jiwer import wer, cer
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]      # bỏ tham chiếu rỗng sau chuẩn hóa
    result["wer"] = float(wer([p[0] for p in pairs], [p[1] for p in pairs])) if pairs else None
    result["cer"] = float(cer([p[0] for p in pairs], [p[1] for p in pairs])) if pairs else None

    lp = cfg.get("long_audio_path")
    if lp:
        phase["name"] = "long_audio"
        _, dt, dur = transcribe_one(model, lp)
        result["long_sec"], result["long_audio_sec"] = dt, dur
        result["long_rtf"] = dt / dur if dur else None

    del model                                            # giải phóng model rồi đo nền lần nữa (ước lượng trôi nền)
    gc.collect()
    phase["name"] = "post_idle"
    time.sleep(float(cfg.get("post_idle_s", 3)))


result = {"model": cfg["model"], "device": cfg["device"], "compute_type": cfg["compute_type"],
          "repeat": cfg["repeat"], "status": "error", "error": None, "hypotheses": [],
          "transcribe_kwargs": cfg["transcribe_kwargs"]}
try:
    init_nvml()
    threading.Thread(target=sample_loop, daemon=True).start()
    import ctranslate2
    supported = sorted(ctranslate2.get_supported_compute_types(cfg["device"]))
    result["supported_compute_types"] = supported
    if cfg["compute_type"] not in supported:      # kiểm tra TRƯỚC khi nạp model, không so khớp chuỗi lỗi
        result["status"] = "unsupported"
        result["error"] = (f"compute_type={cfg['compute_type']} không nằm trong các kiểu "
                           f"CTranslate2 hỗ trợ trên {cfg['device']}: {supported}")
    else:
        run(result)
        result["status"] = "ok"
except Exception as e:
    result["error"] = repr(e)
    result["traceback"] = traceback.format_exc()[-4000:]
finally:
    phase["name"] = "shutdown"
    halt.set()
    time.sleep(0.3)
    result["gpu_error"] = gpu_error
    result.update(stats(list(samples)))
    result["resource_samples"] = samples
    result["cpu_count"] = psutil.cpu_count()
    result["worker_total_sec"] = time.time() - T0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print("  worker xong:", result["status"], flush=True)
