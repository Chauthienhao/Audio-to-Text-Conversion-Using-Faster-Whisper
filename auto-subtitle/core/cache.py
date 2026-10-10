import hashlib #sử dụng thư viện hash của Python


def file_hash(path, chunk_size=1024 * 1024): #nhận file và kích thước đọc mỗi lần (mỗi lần chỉ đọc 1mb)
    h = hashlib.sha256() #SHA-256 biến dữ liệu thành 1 chuỗi 256 bit
    with open(path, "rb") as f: #mở file ở chế độ read binary
        while chunk := f.read(chunk_size): #đọc từng khối 1 MB. Dấu := vừa gán kết quả vào chunk vừa kiểm tra điều kiện.
            h.update(chunk) #đưa chunk vừa đọc vào để 'băm'. Gọi nhiều lần đến khi kết quả giống việc đưa cả file vào.
    return h.hexdigest()[:16] #lấy kết quả dạng chuỗi thập lục phân (64 ký tự), cắt lấy 16 ký tự đầu để tên file gọn.
#f.read() không tham số sẽ nạp toàn bộ file vào RAM. 
# Video 3 GB sẽ ngốn 3 GB bộ nhớ, có thể làm treo máy. 
# Đọc từng khối 1 MB thì RAM chỉ tốn khoảng 1 MB, dù video lớn cỡ nào.

# Đặt tên file theo mã hash + với tên ngôn ngữ, có thể thêm MODEL_SIZE nếu muốn. Nếu đọc được hash + ngôn ngữ đúng thì sẽ dùng lại.
def srt_path_for(video_path, language, output_dir):
    key = f"{file_hash(video_path)}_{language or 'auto'}"
    return output_dir / f"{key}.srt"