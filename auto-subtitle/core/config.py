from pathlib import Path

# Thư mục lưu file phụ đề
SUBS_OUTPUT_DIR = Path("auto-subtitle/outputs") #đường dẫn tuyệt đối, nên phải chạy trên thư mục auto-subtile 
SUBS_OUTPUT_DIR.mkdir(exist_ok=True)

# Cấu hình model
MODEL_SIZE = "large"      # tiny / base / small / medium / large-v3
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