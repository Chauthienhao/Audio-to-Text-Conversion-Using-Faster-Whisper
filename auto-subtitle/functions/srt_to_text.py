#region script to text
def srt_to_text(srt_content):
    lines = []
    for line in srt_content.splitlines():
        line = line.strip()
        if not line or line.isdigit() or " --> " in line:
            continue
        lines.append(line)
    return "\n".join(lines)
#endregion