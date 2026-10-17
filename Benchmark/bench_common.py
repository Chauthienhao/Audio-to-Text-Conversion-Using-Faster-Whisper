"""Phần dùng chung cho mọi notebook benchmark: chạy một cấu hình + phân tích kết quả."""
import json, os, statistics, subprocess, sys, threading, time
from pathlib import Path
import pandas as pd
from bench_metrics import derive_metrics

PROJECT_DIR = Path.cwd()                       # thư mục chứa notebook và data/
RESULTS_DIR = PROJECT_DIR / "benchmark_results"
RAW_DIR = RESULTS_DIR / "raw"
PAYLOAD_DIR = RESULTS_DIR / "payloads"
for _d in (RAW_DIR, PAYLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = PROJECT_DIR / "data" / "manifest.csv"   # 200 câu VIVOS test, seed 42; cột: audio, reference
if os.environ.get("BENCH_MANIFEST"):                       # Colab: chọn manifest khác mà không sửa code
    MANIFEST_PATH = Path(os.environ["BENCH_MANIFEST"])
LONG_AUDIO_PATH = PROJECT_DIR / "data" / "long_audio.wav"
WORKER_PATH = Path(__file__).with_name("bench_worker.py")

# Dùng CHUNG cho mọi cấu hình để so sánh công bằng. temperature=0.0 khác mặc định của thư viện.
TRANSCRIBE_KWARGS = {"language": "vi", "task": "transcribe", "beam_size": 5,
                     "temperature": 0.0, "condition_on_previous_text": True,
                     "vad_filter": False, "word_timestamps": False}

GPU_INDEX = 0


# ---------------------------------------------------------------- dữ liệu
def load_manifest(max_files=None):
    df = pd.read_csv(MANIFEST_PATH)
    assert {"audio", "reference"} <= set(df.columns), "CSV cần cột audio, reference"
    df["audio"] = df["audio"].astype(str).map(
        lambda p: p if Path(p).is_absolute() else str((PROJECT_DIR / p).resolve()))
    df["reference"] = df["reference"].fillna("").astype(str).str.strip()
    df = df[df["reference"] != ""]
    missing = [p for p in df["audio"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} file audio, ví dụ: {missing[:3]}")
    if max_files:
        df = df.head(max_files)
    assert len(df) > 0, "Manifest rỗng"
    return df.to_dict("records")


def ensure_model_downloaded(model):
    """Tải/kiểm tra cache trước khi đo, để thời gian nạp không gồm thời gian tải."""
    from faster_whisper.utils import download_model
    print(f"Kiểm tra/tải model '{model}' ...")
    download_model(model)


# ---------------------------------------------------------------- chạy một cấu hình
def _run_worker(payload_path, timeout_s):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.Popen([sys.executable, str(WORKER_PATH), str(payload_path)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    timer = threading.Timer(timeout_s, p.kill)
    timer.start()
    tail, t0 = [], time.time()
    try:
        for line in p.stdout:
            print(line.rstrip())
            tail.append(line)
    finally:
        timer.cancel()
    p.wait()
    return p.returncode, "".join(tail)[-4000:], (time.time() - t0) >= timeout_s


def run_config(cfg, items, warmup, long_path=None, timeout_s=3 * 3600):
    key = f"{cfg['model']}__{cfg['device']}__{cfg['compute_type']}__r{cfg['repeat']}"
    out = RAW_DIR / f"{key}.json"
    if out.exists():
        try:
            if json.loads(out.read_text(encoding="utf-8")).get("status") == "ok":
                print(f"  Đã có kết quả ok, bỏ qua: {out.name}")
                return json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            pass
        out.unlink(missing_ok=True)               # kết quả lỗi cũ không phải cache hợp lệ
    payload = {**cfg, "output_json": str(out), "eval_items": items, "warmup_audio": warmup,
               "transcribe_kwargs": TRANSCRIBE_KWARGS, "gpu_index": GPU_INDEX,
               "sample_interval_sec": 0.2, "idle_s": 5, "post_idle_s": 3,
               "long_audio_path": str(long_path) if long_path else None}
    pp = PAYLOAD_DIR / f"{key}.json"
    pp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    code, tail, timed_out = _run_worker(pp, timeout_s)
    if not out.exists():                           # worker sập hoặc bị kill: tự ghi JSON lỗi
        out.write_text(json.dumps({**cfg, "status": "error", "hypotheses": [],
                                   "error": "timeout" if timed_out else f"worker thoát mã {code}",
                                   "traceback": tail}, ensure_ascii=False), encoding="utf-8")
    return json.loads(out.read_text(encoding="utf-8"))


def run_benchmark(model, device, compute_type, repeats=1, max_files=5, run_long=False):
    items = load_manifest(max_files)
    print(f"Cấu hình: {model} | {device} | {compute_type} | lặp {repeats} | {len(items)} file")
    ensure_model_downloaded(model)
    long_path = LONG_AUDIO_PATH if (run_long and LONG_AUDIO_PATH.is_file()) else None
    if run_long and long_path is None:
        print("  Không thấy file dài:", LONG_AUDIO_PATH)
    for r in range(1, repeats + 1):
        print(f"--- lần {r}/{repeats}")
        res = run_config({"model": model, "device": device, "compute_type": compute_type, "repeat": r},
                         items, items[0]["audio"], long_path)
        print(f"  => {res.get('status')} | {res.get('error') or ''}")
    show_results(model, device, compute_type)


def run_all(configs, repeats=1, max_files=100, run_long=False, cooldown_s=45, timeout_s=2700,
            on_config_done=None):
    """Chạy tuần tự nhiều cấu hình [(model, device, compute_type), ...] trong MỘT lần.
    - Mỗi lần lặp = một tiến trình worker mới => RAM/VRAM của lần trước được hệ điều hành thu hồi.
    - Mỗi worker tự đo mức nền (CPU/RAM hệ thống, VRAM) ở pha idle và trừ ra khi tính chỉ số ròng.
    - Giữa các lần nghỉ cooldown_s giây cho máy nguội."""
    items = load_manifest(max_files)
    long_path = LONG_AUDIO_PATH if (run_long and LONG_AUDIO_PATH.is_file()) else None
    print(f"{len(configs)} cấu hình x {repeats} lần | {len(items)} file")
    for m in sorted({c[0] for c in configs}):
        ensure_model_downloaded(m)
    total = len(configs) * repeats
    done, t_start = 0, time.time()
    for i, (model, device, ct) in enumerate(configs, 1):
        for r in range(1, repeats + 1):
            pct = 100 * done / total
            el = time.time() - t_start
            eta = f" | còn khoảng {el / done * (total - done) / 60:.0f} phút" if done else ""
            print(f"\n=== [{i}/{len(configs)}] {model} | {device} | {ct} | lần {r}/{repeats} "
                  f"| tiến độ {pct:.0f}% ({done}/{total}) | đã chạy {el / 60:.1f} phút{eta}", flush=True)
            res = run_config({"model": model, "device": device, "compute_type": ct, "repeat": r},
                             items, items[0]["audio"], long_path, timeout_s)
            done += 1
            if on_config_done:                                   # ví dụ: xuất summary/predictions lên Drive
                try:
                    on_config_done(load_results())
                except Exception as e:
                    print("  on_config_done lỗi:", repr(e))
            if res.get("status") == "ok":
                print(f"  XONG {model}|{device}|{ct}: WER {res['wer']:.3f} | CER {res['cer']:.3f} | "
                      f"RTF {res['rtf']:.3f} | nạp model {res['model_load_sec']:.1f}s", flush=True)
            else:
                print(f"  {res.get('status')} {model}|{device}|{ct}: {str(res.get('error'))[:150]}", flush=True)
            print(f"  >>> Tổng tiến độ: {100 * done / total:.0f}% ({done}/{total})", flush=True)
            if done < total:
                print(f"  nghỉ {cooldown_s}s cho máy nguội ...", flush=True)
                time.sleep(cooldown_s)
    return load_results()


NET_METRICS = ["sys_cpu_baseline_pct", "sys_cpu_baseline_std_pct", "sys_ram_baseline_mb",
               "vram_baseline_std_mb", "bg_cpu_pct_eval", "cpu_net_pct_eval", "ram_net_peak_mb",
               "vram_net_peak_mb", "bg_drift_vram_mb"]
COLS = ["repeat", "status", "effective_compute_type", "model_load_sec", "warmup_sec",
        "eval_inference_sec", "eval_audio_sec", "rtf", "wer", "cer", "cpu_avg_pct_eval",
        "ram_peak_mb", "vram_baseline_mb", "vram_delta_peak_mb", *NET_METRICS, "error"]


def show_results(model, device, compute_type):
    df = load_results()
    if df.empty:
        print("Chưa có kết quả.")
        return None
    m = df[(df.model == model) & (df.device == device) & (df.compute_type == compute_type)]
    return m[[c for c in COLS if c in m.columns]].round(3)


# ---------------------------------------------------------------- phân tích
def load_results():
    rows = []
    for p in sorted(RAW_DIR.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print("Không đọc được", p.name, e)
    df = pd.DataFrame(rows)
    if "cpu_avg_pct_eval" in df and "cpu_count" in df:       # chuẩn hóa về 0-100% của toàn bộ nhân
        df["cpu_avg_norm_eval"] = df["cpu_avg_pct_eval"] / df["cpu_count"]
    if "resource_samples" in df:                              # tính lại từ mẫu cho JSON cũ chưa có số này
        def _gpu_eval(samples):
            v = [x["gpu_util"] for x in (samples if isinstance(samples, list) else [])
                 if x.get("phase") == "evaluation" and x.get("gpu_util") is not None]
            allv = [x["gpu_util"] for x in (samples if isinstance(samples, list) else [])
                    if x.get("phase") in ("model_load", "warmup", "evaluation", "long_audio")
                    and x.get("gpu_util") is not None]
            return pd.Series({"gpu_util_avg_eval_pct": statistics.mean(v) if v else None,
                              "gpu_util_eval_n": len(v) if v else None,
                              "gpu_util_peak_pct": max(allv) if allv else None})
        calc = df["resource_samples"].apply(_gpu_eval)
        for col in calc.columns:
            df[col] = df[col].combine_first(calc[col]) if col in df else calc[col]
        def _derive(samples, ncpu):
            if not isinstance(samples, list) or not samples or not isinstance(ncpu, (int, float)):
                return pd.Series(dtype=object)
            return pd.Series(derive_metrics(samples, ncpu))
        calc = pd.DataFrame([_derive(s, n) for s, n in zip(df["resource_samples"], df.get("cpu_count", [None] * len(df)))],
                            index=df.index)
        for col in calc.columns:                              # chỉ điền chỗ JSON chưa có, không ghi đè số worker đã lưu
            df[col] = df[col].combine_first(calc[col]) if col in df else calc[col]
    return df


def show_status(df):
    if df.empty:
        print("Chưa có kết quả.")
        return
    print(df.groupby(["model", "device", "compute_type", "status"]).size().unstack(fill_value=0))
    bad = df[df.status != "ok"]
    for _, r in bad.iterrows():
        print(f"- {r['model']}|{r['device']}|{r['compute_type']}|r{r['repeat']}: {r['status']}: "
              f"{str(r.get('error'))[:200]}")


METRICS = ["model_load_sec", "warmup_sec", "eval_inference_sec", "rtf", "long_rtf", "wer", "cer",
           "cpu_avg_pct_eval", "cpu_avg_norm_eval", "cpu_peak_pct", "ram_peak_mb", "ram_delta_peak_mb",
           "vram_baseline_mb", "vram_peak_mb", "vram_delta_peak_mb",
           "gpu_util_avg_eval_pct", "gpu_util_peak_pct", *NET_METRICS]


def summarize(df):
    ok = df[df.status == "ok"]
    if ok.empty:
        return pd.DataFrame()
    mets = [m for m in METRICS if m in ok.columns]
    g = ok.groupby(["model", "device", "compute_type"])[mets]
    out = pd.concat([g.size().rename("n_ok"), g.mean().add_suffix("_mean"),
                     g.std().add_suffix("_std")], axis=1).reset_index()
    order = {"tiny": 0, "base": 1, "small": 2, "medium": 3, "large-v3": 4}
    return out.sort_values(["device", "compute_type", "model"],
                           key=lambda s: s.map(order) if s.name == "model" else s).reset_index(drop=True)


def _label(r):
    return f"{r['model']}\n{r['device']}/{r['compute_type']}"


def plot_wer_vs_rtf(s):
    import matplotlib.pyplot as plt
    d = s.dropna(subset=["wer_mean", "rtf_mean"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for dev, mk in (("cpu", "o"), ("cuda", "^")):
        x = d[d.device == dev]
        ax.scatter(x.rtf_mean, x.wer_mean * 100, marker=mk, s=60, label=dev)
        for _, r in x.iterrows():
            ax.annotate(f"{r['model']}/{r['compute_type']}", (r.rtf_mean, r.wer_mean * 100),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("RTF (thấp = nhanh)"); ax.set_ylabel("WER (%)"); ax.set_title("WER so với RTF")
    ax.set_xscale("log"); ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.show()


def plot_bar(s, metric, ylabel, title):
    import matplotlib.pyplot as plt
    col = metric + "_mean"
    d = s.dropna(subset=[col]) if col in s else pd.DataFrame()
    if d.empty:
        print("Chưa có dữ liệu:", metric); return
    err = d[metric + "_std"].fillna(0) if metric + "_std" in d else None
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(d)), 4.5))
    ax.bar([_label(r) for _, r in d.iterrows()], d[col], yerr=err, capsize=3)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.show()


def plot_timeseries(df, model, device, compute_type, repeat=1):
    import matplotlib.pyplot as plt
    m = df[(df.model == model) & (df.device == device) & (df.compute_type == compute_type)
           & (df.repeat == repeat) & (df.status == "ok")]
    if m.empty:
        print("Không có kết quả ok cho cấu hình này."); return
    r = m.iloc[0]
    s = pd.DataFrame(r["resource_samples"])
    base = r.get("vram_baseline_mb")
    ncpu = r.get("cpu_count") or 1
    s["worker_cpu_norm"] = s["cpu_pct"] / ncpu                 # CPU worker theo thang toàn hệ thống
    cpu_lines = [("worker_cpu_norm", "worker")]
    if "sys_cpu_pct" in s and s["sys_cpu_pct"].notna().any():
        cpu_lines.append(("sys_cpu_pct", "toàn hệ thống"))     # khoảng cách giữa 2 đường = nền
    rows = [(cpu_lines, "CPU (% toàn hệ thống)"), ([("ram_mb", "worker")], "RAM RSS (MB)")]
    if device == "cuda":
        if s["gpu_mb"].notna().any() and base == base and base is not None:
            s["vram_delta"] = (s["gpu_mb"] - base).clip(lower=0)
            rows.append(([("vram_delta", "thiết bị − nền")], "VRAM tăng so với nền (MB)"))
        if s["gpu_util"].notna().any():
            rows.append(([("gpu_util", "toàn thiết bị")], "GPU (%)"))
    colors = {"idle": "#dddddd", "startup": "#eeeeee", "model_load": "#cfe8ff",
              "warmup": "#fff2b3", "evaluation": "#d6f5d6", "long_audio": "#f5d6f5",
              "post_idle": "#e8e0f0"}
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.6 * len(rows)), sharex=True)
    axes = [axes] if len(rows) == 1 else axes
    s["blk"] = (s.phase != s.phase.shift()).cumsum()
    for ax, (lines, lab) in zip(axes, rows):
        for col, name in lines:
            ax.plot(s.t, s[col], lw=1, label=name)
        for _, b in s.groupby("blk"):
            ax.axvspan(b.t.iloc[0], b.t.iloc[-1], color=colors.get(b.phase.iloc[0], "#fff"), alpha=.6)
        ax.set_ylabel(lab); ax.grid(alpha=.3)
        if len(lines) > 1:
            ax.legend(fontsize=7, loc="upper left")
    for ph, c in colors.items():
        axes[0].axvspan(0, 0, color=c, label=ph)
    axes[0].legend(ncol=6, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("giây từ lúc worker bắt đầu")
    fig.suptitle(f"{model} | {device} | {compute_type} | lần {repeat}")
    plt.tight_layout(); plt.show()


def export_predictions(df):
    rows = []
    for _, r in df[df.status == "ok"].iterrows():
        for h in r["hypotheses"]:
            rows.append({"model": r["model"], "device": r["device"],
                         "compute_type": r["compute_type"], "repeat": r["repeat"], **h})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    return out
