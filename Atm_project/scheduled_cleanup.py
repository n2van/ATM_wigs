import time
import schedule
from cleanup_tmp import cleanup_old_files
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduled_cleanup.log"),
        logging.StreamHandler()
    ]
)

def job():
    logging.info("Đang chạy tác vụ dọn dẹp theo lịch...")
    cleanup_old_files("./tmp", hours_threshold=3)
    logging.info("Hoàn thành tác vụ dọn dẹp")

# Lên lịch chạy mỗi giờ
schedule.every(1).hour.do(job)

logging.info("Đã khởi động tác vụ dọn dẹp tự động")
logging.info("Sẽ chạy mỗi giờ để kiểm tra và xóa các file cũ")

# Chạy ngay lần đầu
job()

# Vòng lặp chính
while True:
    schedule.run_pending()
    time.sleep(60)  # Kiểm tra mỗi phút