from core.transcriber import transcribe
from core.subtitle import to_srt
import sys
sys.stdout.reconfigure(encoding="utf-8")

segs, lang = transcribe(r"E:\4nd\NNLTHD\auto-subtitle\in-10-minutes-this-room-will-explode.mp4",
                        on_progress=lambda p: print(f"{p:.0%}"))
with open("auto-subtitle/outputs/test.srt", "w", encoding="utf-8") as f:
    f.write(to_srt(segs))
print("Done:", lang)
