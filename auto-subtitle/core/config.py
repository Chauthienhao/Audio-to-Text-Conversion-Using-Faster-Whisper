from pathlib import Path

# Thư mục lưu file phụ đề
BASE_DIR = Path(__file__).resolve().parent.parent
SUBS_OUTPUT_DIR = BASE_DIR / "outputs"
SUBS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình model
MODEL_SIZE = "base"
DEVICE = "cpu"            
COMPUTE_TYPE = "int8" 

# Tên hiển thị -> mã ngôn ngữ (None = tự phát hiện)
LANGUAGES = {
    "Tự động": None,
    "Tiếng Việt": "vi",
    "English": "en"
}
