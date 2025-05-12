import os
import csv
import datetime
from pathlib import Path

class CustomerService:
    def __init__(self, data_dir="customer_data"):
        """
        Khởi tạo dịch vụ quản lý khách hàng
        
        Args:
            data_dir: Thư mục lưu trữ dữ liệu khách hàng
        """
        self.data_dir = data_dir
        self._ensure_data_dir()
        self.csv_file = os.path.join(self.data_dir, "customer_leads.csv")
        self._ensure_csv_file()
        
    def _ensure_data_dir(self):
        """Đảm bảo thư mục dữ liệu tồn tại"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Đã tạo thư mục {self.data_dir} để lưu trữ dữ liệu khách hàng")
    
    def _ensure_csv_file(self):
        """Đảm bảo file CSV tồn tại với header"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Thời gian', 'Tên khách hàng', 'Số điện thoại', 
                    'Email', 'Hình dạng khuôn mặt', 'Mã sản phẩm quan tâm', 
                    'Ghi chú'
                ])
            print(f"Đã tạo file {self.csv_file} để lưu trữ thông tin khách hàng")
    
    def save_customer_info(self, customer_name, customer_phone, customer_email, 
                          face_shape="", product_code="", notes=""):
        """
        Lưu thông tin khách hàng vào file CSV
        
        Args:
            customer_name: Tên khách hàng
            customer_phone: Số điện thoại khách hàng
            customer_email: Email khách hàng
            face_shape: Hình dạng khuôn mặt (nếu có)
            product_code: Mã sản phẩm quan tâm (nếu có)
            notes: Ghi chú thêm
            
        Returns:
            bool: True nếu lưu thành công, False nếu thất bại
        """
        try:
            # Lấy thời gian hiện tại
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Ghi thông tin vào file CSV
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    current_time, customer_name, customer_phone, 
                    customer_email, face_shape, product_code, 
                    notes
                ])
            
            print(f"Đã lưu thông tin khách hàng {customer_name} thành công")
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu thông tin khách hàng: {str(e)}")
            return False