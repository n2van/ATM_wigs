import os
import time
import shutil
from datetime import datetime, timedelta
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cleanup_tmp.log"),
        logging.StreamHandler()
    ]
)

def cleanup_old_files(folder_path="./tmp", hours_threshold=3):
    """
    Xóa các file trong thư mục nếu chúng không được sửa đổi trong số giờ quy định
    
    Args:
        folder_path: Đường dẫn đến thư mục cần dọn dẹp
        hours_threshold: Số giờ, các file cũ hơn số giờ này sẽ bị xóa
    """
    if not os.path.exists(folder_path):
        logging.warning(f"Thư mục {folder_path} không tồn tại")
        return
    
    # Thời gian hiện tại
    current_time = time.time()
    # Thời gian ngưỡng (3 giờ trước)
    threshold_time = current_time - (hours_threshold * 3600)
    
    # Đếm số file đã xóa và tổng dung lượng
    deleted_count = 0
    total_size = 0
    
    logging.info(f"Bắt đầu dọn dẹp thư mục {folder_path}")
    logging.info(f"Xóa các file không được sửa đổi trong {hours_threshold} giờ qua")
    
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Kiểm tra xem đây có phải là file không
        if os.path.isfile(file_path):
            # Lấy thời gian sửa đổi cuối cùng
            last_modified_time = os.path.getmtime(file_path)
            
            # Nếu file không được sửa đổi trong 3 giờ qua
            if last_modified_time < threshold_time:
                try:
                    # Lấy kích thước file trước khi xóa
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    # Xóa file
                    os.remove(file_path)
                    deleted_count += 1
                    
                    # Log thông tin
                    last_modified = datetime.fromtimestamp(last_modified_time).strftime('%Y-%m-%d %H:%M:%S')
                    logging.info(f"Đã xóa: {filename} (Sửa đổi lần cuối: {last_modified}, Kích thước: {file_size/1024:.2f} KB)")
                    
                except Exception as e:
                    logging.error(f"Lỗi khi xóa file {filename}: {str(e)}")
        
        # Nếu là thư mục con và rỗng, xóa luôn
        elif os.path.isdir(file_path):
            try:
                # Xóa thư mục nếu rỗng
                os.rmdir(file_path)
                logging.info(f"Đã xóa thư mục rỗng: {filename}")
            except OSError:
                # Thư mục không rỗng, bỏ qua
                pass
    
    # Tổng kết
    if deleted_count > 0:
        logging.info(f"Đã xóa tổng cộng {deleted_count} file, giải phóng {total_size/1024/1024:.2f} MB")
    else:
        logging.info(f"Không có file nào cần xóa")

if __name__ == "__main__":
    cleanup_old_files()