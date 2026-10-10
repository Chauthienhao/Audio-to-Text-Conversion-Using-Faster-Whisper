from pathlib import Path

# Thư mục lưu file phụ đề
BASE_DIR = Path(__file__).resolve().parent.parent
SUBS_OUTPUT_DIR = BASE_DIR / "outputs"
SUBS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình model
MODEL_SIZE = "large-v3"      # tiny / base / small / medium / large-v3
DEVICE = "cuda"            # "cuda" nếu có GPU NVIDIA
COMPUTE_TYPE = "int8"     # "float16" nếu dùng GPU

# Tên hiển thị -> mã ngôn ngữ (None = tự phát hiện)
LANGUAGES = {
    "Tự động": None,
    "Tiếng Việt": "vi",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
    "中文": "zh",
}
