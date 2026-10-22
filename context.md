# Bối cảnh
Mình là sinh viên (nhóm 4 người) đang viết đồ án "Tìm hiểu và ứng dụng faster-whisper trong nhận dạng tiếng nói và sinh phụ đề tự động". Mình phụ trách Chương 4 (Bản chất công nghệ) và đang chuẩn bị viết **Chương 8: Triển khai, tích hợp và minh chứng kết quả**.

Mã nguồn dùng làm chuẩn cho Chương 8 là nhánh **chau-den-bien** (commit mới nhất tại thời điểm viết: 7d26b80). Hãy checkout đúng nhánh này và chỉ dựa vào nhánh này. Không dùng nhánh main (main chạy cấu hình khác: base/cpu/int8, beam 3, có FFmpeg trích audio, không phải sản phẩm cuối).

# Yêu cầu của Chương 8 (theo đề)
- 8.1 Tích hợp hệ thống: mô tả kiến thức/cơ chế đã học được hiện thực hóa thế nào trong sản phẩm cuối: module nào dùng trực tiếp, phần nào refactor/viết lại, thí nghiệm nào dùng để ra quyết định kỹ thuật. Không cần "copy code"; cần chứng minh chuỗi kiến thức -> quyết định -> sản phẩm.
- 8.2 Kết quả đạt được: minh chứng theo track (mình thuộc track AI/Data: bộ test, bảng số liệu đánh giá, latency/throughput khi phục vụ).
- 8.3 Bảng phủ kỹ thuật, các cột: TT | Nội dung (Phần 2) | Tầng | Dùng ở chức năng nào (Phần 3) | File / commit / version | Minh chứng / số liệu | Trạng thái.
Chỉ ghi những gì THẬT SỰ có trong mã đồ án. Phân biệt rõ: "Dùng trong đồ án" / "Có mã nhưng chưa tích hợp" / "Thử nghiệm độc lập".

# Việc cần làm (chỉ đọc mã, KHÔNG sửa file nguồn, KHÔNG commit)
1. Xác nhận nhánh và commit hiện tại (`git branch --show-current`, `git log -1`).
2. Đọc toàn bộ thư mục auto-subtitle/ (core/config.py, core/transcriber.py, core/subtitle.py, core/burner.py, core/cache.py, functions/*.py, app.py, imports.py, requirements.txt) và lập bản đồ luồng xử lý từ lúc người dùng chọn video đến lúc xuất phụ đề/video: hàm nào được gọi ở đâu, dữ liệu đi qua bước nào.
3. Với mỗi nội dung Tầng 1A của Chương 4, tìm mã tương ứng trong nhánh này và ghi **đường dẫn file + số dòng chính xác + trích đoạn ngắn**:
   (1) Luồng đầu vào audio/video (video được truyền thẳng cho transcribe như thế nào; có chỗ nào tự chạy FFmpeg trên audio không).
   (2) Tải/chạy model: model_size, device, compute_type, cơ chế singleton get_model().
   (3) Segment start/end/text: từ transcribe -> chunking_segments -> to_srt.
   (4) Tham số giải mã: beam_size, language, vad_filter và vad_parameters, condition_on_previous_text, word_timestamps; có dùng BatchedInferencePipeline không và batch_size là bao nhiêu (nếu có).
   (5) Xử lý lỗi ASR thường gặp: có đoạn mã hậu xử lý nào thực sự xử lý lỗi nhận dạng không (nếu không có, nói rõ là không có).
4. Kiểm tra sự nhất quán và báo cáo mọi mâu thuẫn, đặc biệt:
   - requirements.txt ghi `faster-whisper==1.21` (nghi là lỗi đánh máy của 1.2.1): chạy `pip show faster-whisper` nếu có môi trường, và cho biết phiên bản thực tế.
   - Comment trong config.py ("float16 nếu dùng GPU") có khớp với giá trị COMPUTE_TYPE="int8" không.
   - Giá trị cấu hình thật trong mã so với những gì báo cáo Chương 4 đang nói (large-v3, cuda, int8, beam_size=5).
5. Tạo file `chuong8_notes.md` ở thư mục gốc (không thêm vào git) gồm:
   a. Sơ đồ luồng xử lý (dạng văn bản/mermaid).
   b. Bảng 8.3 dạng markdown cho 5 nội dung Tầng 1A ở trên, điền đủ 7 cột. Cột "Trạng thái" phải trung thực. Chỗ nào chưa chắc thì ghi "[cần xác nhận]", không tự suy đoán.
   c. Danh sách mâu thuẫn/điểm cần sửa trong mã hoặc báo cáo (mục 4).
   d. Gợi ý dàn ý cho 8.1 theo chuỗi kiến thức -> quyết định -> sản phẩm cho từng nội dung (ví dụ: benchmark ở mục 4.2.3 -> chọn large-v3/cuda/int8 -> config.py và get_model()).

# Quy tắc
- Không bịa số dòng, tên hàm, kết quả chạy. Nếu không tìm thấy, nói rõ "không tìm thấy".
- Không chạy lệnh có thể sửa/xóa file, không cài thêm gói nếu chưa cần. Không cần xin phép cho các lệnh chỉ đọc (git, cat, grep, ls, pip show).
- Benchmark ở Chương 4 chạy trên WhisperModel thuần (không VAD, không batching, không word_timestamps); mã đồ án dùng cấu hình khác. Khi viết gợi ý cho 8.1, nhấn mạnh rằng chỉ "xu hướng" từ benchmark được chuyển sang đồ án, không phải con số tuyệt đối.
- Trả lời bằng tiếng Việt. Cuối cùng tóm tắt ngắn: đã tìm được gì, còn thiếu gì cần mình bổ sung.