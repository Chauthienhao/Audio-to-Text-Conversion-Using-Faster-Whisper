def format_timestamps(seconds):
    ms = int(round(seconds*1000))
    h,ms = divmod(ms,3_600_000) # chia ms cho 3.600.000 phần nguyên cho biến h phần dư lại đưa cho ms. '_' dấu phân cách cho dễ nhìn.
    m,ms = divmod(ms,60_000)
    s,ms = divmod(ms,1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def to_srt(segments):
    blocks = []
    for i,seg in enumerate(segments,start=1):
        start = format_timestamps(seg["start"])
        end = format_timestamps(seg["end"])
        text = seg["text"].strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"