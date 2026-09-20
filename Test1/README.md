## Hướng Dẫn Cài Đặt Test Ban Đầu

## 1. HƯỚNG DẪN CÀI ĐẶT QUA TERMINAL
### Hướng dẫn cài cuda
#### Check nvidia
```powershell
nvidia-smi
```
Sau đó Vào trang CUDA Toolkit Archive. 
Chọn bản phù hợp
Chọn window x86_64, cài exe
Bật exe cài Express
#### Cách check xem đã cài thành công chưa:
```powershell
nvcc --version
```

Tải cấu hình cuda cuDNN ta chạy lệnh:
```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
Cách check xem đã cài đc cuda cuDNN chưa:
```powershell
python -c "from faster_whisper import WhisperModel; model = WhisperModel('base', device='cuda', compute_type='float16'); print('KICH HOAT GPU THANH CONG!')"
```

#### Cài thư viện
Nâng cấp gói python (Nếu quá cũ)
```powershell
python -m pip install --upgrade pip
```
Cài thư viện
```powershell
pip install torch torchaudio openai-whisper faster-whisper pywhispercpp psutil
```
Nếu hệ thống chưa có sẵn công cụ FFmpeg để giải mã âm thanh, chỉ cần chạy 1 dòng lệnh: 
```powershell
winget install Gyan.FFmpeg
```

[!WARNING]
### LƯU Ý QUAN TRỌNG 
**Pytorch KHÔNG NÊN sử dụng trong Faster-Whisper, tải Pytorch ở đây chỉ là để dùng so sánh với OpenAIWhisper để cho mình thấy trực quan vấn đề.** 
**ĐỪNG NHẦM LẪN VÀ HÃY DOWN CUDA 😭**

## 2. HƯỚNG DẪN CHẠY KIỂM THỬ TRÊN TERMINAL
Di chuyển vào thư mục kiểm thử và chạy file benchmark:
```powershell
cd Test1
python Source/benchmark.py
```
Lần chạy đầu: Chương trình tự động kết nối mạng để tải trọng số các mô hình bản base về máy.
Tiến trình xử lý: Terminal lần lượt duyệt qua 5 file âm thanh sạch (Clean) và 2 file âm thanh thực tế dính nhạc nền (Noise), sau đó in trực tiếp thời gian chạy, bộ nhớ RAM và toàn bộ chuỗi văn bản trích xuất được.


## 3. KẾT QUẢ THU ĐƯỢC
#### Clean(Không có tiếng ồn):
![Image](PicReadme/pic1.png)

#### Noise(Có tiếng ồn):
![Image](PicReadme/pic2.png)
![Image](PicReadme/pic3.png)

## 4. CÁC VẤN ĐỀ KĨ THUẬT

### Lỗi TypeError: unexpected keyword argument 'vad_filter':
Nguyên nhân: Tính năng vad_filter là cơ chế độc quyền của faster-whisper. Thư viện gốc của openai-whisper và whisper.cpp không hỗ trợ cờ này.
#### Giải pháp: Chỉ truyền tham số này vào lời gọi hàm model_fast.transcribe().

### Lỗi Exception: WAV file must be 16000 Hz:
Nguyên nhân: Engine C++ pywhispercpp bắt buộc tần số lấy mẫu phải chuẩn 16 kHz.
#### Giải pháp: Nạp âm thanh qua whisper.load_audio(filepath) để tự động chuẩn hóa tần số lấy mẫu trước khi chuyển sang C++.

### Hiện tượng ảo giác âm thanh (Audio Hallucination):
Nguyên nhân: Khi ép cờ tiếng Việt (language="vi") cho các file YouTube vốn là tiếng Anh dính nhạc nền mạnh, mô hình gốc OpenAI sinh chữ tiếng Trung (加油) và các từ vô nghĩa.
#### Giải pháp: Đặt language=None để mô hình tự động nhận diện đúng ngôn ngữ nguồn.

### Vấn đề nuốt chữ của VAD trên tập Noise:
Hiện tượng: Ở video có nhạc dồn dập và người nói hét nhanh (escape...), Faster-Whisper chạy cực nhanh (1.85s so với 5.58s) nhưng bị sót một số câu thoại ở giữa.
Nguyên nhân: Bộ lọc Silero VAD ở ngưỡng mặc định (0.5) nhầm tiếng người dính nhạc lớn là tạp âm nên tự ý cắt bỏ.
#### Giải pháp: Hạ ngưỡng lọc nhạy hơn vad_parameters=dict(threshold=0.35) hoặc tạm tắt VAD vad_filter=False khi cần giữ đủ 100% câu chữ.
