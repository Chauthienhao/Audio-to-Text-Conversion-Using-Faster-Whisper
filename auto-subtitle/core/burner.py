#burn-in subtitle or hardsub
import subprocess
from pathlib import Path


def burn_subtitle(video_path, srt_path, output_path):
    video = Path(video_path).resolve() 
    srt = Path(srt_path).resolve()
    out = Path(output_path).resolve()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"subtitles={srt.name}",
        "-c:a", "aac",
        str(out),
    ]
    result = subprocess.run(
        cmd,
        cwd=srt.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-800:])
    return str(out)