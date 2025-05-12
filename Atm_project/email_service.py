import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import logging
import dotenv

# Tải biến môi trường từ file .env
dotenv.load_dotenv()

class EmailService:
    def __init__(self):
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "")
        
    def send_email(self, recipient_email, subject, message, image_path=None):
        if not self.sender_email or not self.sender_password:
            print("Chưa cấu hình email người gửi hoặc mật khẩu trong biến môi trường")
            return False
            
        try:
            # Tạo email
            email = MIMEMultipart()
            email["From"] = self.sender_email
            email["To"] = recipient_email
            email["Subject"] = subject
            
            # Thêm nội dung
            email.attach(MIMEText(message, "html"))
            
            # Thêm hình ảnh nếu có
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    img_data = img_file.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-ID', '<image1>')
                    image.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    email.attach(image)
            
            # Kết nối và gửi email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(email)
                
            print(f"Đã gửi email thành công đến {recipient_email}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi gửi email: {str(e)}")
            return False