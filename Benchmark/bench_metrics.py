"""Tính các chỉ số tài nguyên đã trừ nhiễu nền từ danh sách mẫu (resource_samples).
Dùng chung cho bench_worker (lúc chạy) và bench_common (tính lại cho JSON cũ nếu có đủ trường)."""
import statistics

RUN_PHASES = ("model_load", "warmup", "evaluation", "long_audio")


def _vals(rows, key):
    return [x[key] for x in rows if x.get(key) is not None]


def derive_metrics(s, cpu_count):
    """s: list dict mẫu (t, phase, cpu_pct, ram_mb, gpu_mb, gpu_util, sys_cpu_pct, sys_ram_mb, gpu_proc_mb)."""
    idle = [x for x in s if x.get("phase") == "idle"]
    run = [x for x in s if x.get("phase") in RUN_PHASES]
    ev = [x for x in s if x.get("phase") == "evaluation"]
    post = [x for x in s if x.get("phase") == "post_idle"]
    out = {}

    # --- mức nền đo ở pha idle (trước khi nạp model)
    sc = _vals(idle, "sys_cpu_pct")
    if sc:
        out["sys_cpu_baseline_pct"] = statistics.mean(sc)
        out["sys_cpu_baseline_std_pct"] = statistics.pstdev(sc)
    sr = _vals(idle, "sys_ram_mb")
    if sr:
        out["sys_ram_baseline_mb"] = statistics.mean(sr)
    g_idle = _vals(idle, "gpu_mb")
    if g_idle:
        out["vram_baseline_mb"] = statistics.median(g_idle)
        out["vram_baseline_std_mb"] = statistics.pstdev(g_idle)

    # --- CPU: nền và phần tăng thêm trong pha evaluation (thang toàn hệ thống 0-100%)
    ev_sys = [x for x in ev if x.get("sys_cpu_pct") is not None]
    if ev_sys and cpu_count:
        sys_mean = statistics.mean(x["sys_cpu_pct"] for x in ev_sys)
        worker_norm = statistics.mean(x["cpu_pct"] / cpu_count for x in ev_sys)
        out["bg_cpu_pct_eval"] = max(0.0, sys_mean - worker_norm)          # CPU của các tiến trình khác
        if "sys_cpu_baseline_pct" in out:
            out["cpu_net_pct_eval"] = max(0.0, sys_mean - out["sys_cpu_baseline_pct"])

    # --- RAM: đỉnh RSS của worker trừ RSS lúc idle
    ram_idle, ram_run = _vals(idle, "ram_mb"), _vals(run, "ram_mb")
    if ram_idle and ram_run:
        out["ram_net_peak_mb"] = max(0.0, max(ram_run) - statistics.mean(ram_idle))

    # --- VRAM: theo tiến trình nếu NVML cho, nếu không thì đỉnh thiết bị trừ nền
    gp, g_run = _vals(run, "gpu_proc_mb"), _vals(run, "gpu_mb")
    if gp:
        out["vram_net_peak_mb"], out["vram_net_source"] = max(gp), "process"
    elif g_run and g_idle:
        out["vram_net_peak_mb"] = max(0.0, max(g_run) - statistics.median(g_idle))
        out["vram_net_source"] = "device_delta"

    # --- trôi VRAM nền: sau khi giải phóng model so với trước khi nạp
    g_post = _vals(post, "gpu_mb")
    if g_post and g_idle:
        out["bg_drift_vram_mb"] = statistics.median(g_post) - statistics.median(g_idle)
    return out
