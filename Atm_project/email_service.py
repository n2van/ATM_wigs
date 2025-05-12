import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import logging

class EmailService:
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587, 
                sender_email="", sender_password=""):
        """
        Khởi tạo dịch vụ email
        
        Args:
            smtp_server: Máy chủ SMTP
            smtp_port: Cổng SMTP
            sender_email: Email người gửi
            sender_password: Mật khẩu email người gửi
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        
    def send_email(self, recipient_email, subject, message, image_path=None):
        """
        Gửi email
        
        Args:
            recipient_email: Email người nhận
            subject: Tiêu đề email
            message: Nội dung email (HTML)
            image_path: Đường dẫn đến hình ảnh đính kèm (nếu có)
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        if not self.sender_email or not self.sender_password:
            print("Chưa cấu hình email người gửi hoặc mật khẩu")
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
    
    def send_customer_notification(self, customer_name, customer_phone, customer_email, 
                                  product_code="", face_shape="", notes="", 
                                  image_path=None, sales_team_email="sansinglong71@gmail.com"):
        """
        Gửi email thông báo về khách hàng mới cho team sale
        
        Args:
            customer_name: Tên khách hàng
            customer_phone: Số điện thoại khách hàng
            customer_email: Email khách hàng
            product_code: Mã sản phẩm quan tâm (nếu có)
            face_shape: Hình dạng khuôn mặt (nếu có)
            notes: Ghi chú thêm (nếu có)
            image_path: Đường dẫn đến hình ảnh khuôn mặt khách hàng (nếu có)
            sales_team_email: Email team sale (mặc định là sansinglong71@gmail.com)
            
        Returns:
            bool: True nếu gửi thành công, False nếu thất bại
        """
        subject = f"[ATMwigs] Khách hàng mới: {customer_name}"
        
        # Tạo nội dung HTML
        message = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #0e1b4d; padding: 20px; text-align: center; color: white; }}
                .content {{ padding: 20px; }}
                .customer-info {{ background-color: #f0f9ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .product-info {{ background-color: #f0fff4; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .footer {{ background-color: #f8f9fa; padding: 10px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>ATMwigs - Thông Báo Khách Hàng Mới</h2>
                </div>
                <div class="content">
                    <p>Xin chào team sale,</p>
                    <p>Có một khách hàng mới vừa để lại thông tin trên hệ thống ATMwigs:</p>
                    
                    <div class="customer-info">
                        <h3>Thông tin khách hàng:</h3>
                        <p><b>Tên:</b> {customer_name}</p>
                        <p><b>Số điện thoại:</b> {customer_phone}</p>
                        <p><b>Email:</b> {customer_email or "Không cung cấp"}</p>
                        <p><b>Hình dạng khuôn mặt:</b> {face_shape or "Chưa phân tích"}</p>
                    </div>
        """
        
        if product_code:
            message += f"""
                    <div class="product-info">
                        <h3>Thông tin sản phẩm:</h3>
                        <p><b>Mã sản phẩm quan tâm:</b> {product_code}</p>
                    </div>
            """
        
        if notes:
            message += f"""
                    <div class="product-info">
                        <h3>Ghi chú từ khách hàng:</h3>
                        <p>{notes}</p>
                    </div>
            """
        
        message += f"""
                    <p>Vui lòng liên hệ với khách hàng sớm nhất có thể.</p>
                    <p>Trân trọng,<br>Hệ thống ATM Wigs</p>
                </div>
                <div class="footer">
                    <p>© {2023} ATM Wigs. Tất cả các quyền được bảo lưu.</p>
                    <p>Email này được gửi tự động từ hệ thống.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Gửi email thông báo cho team sale
        return self.send_email(sales_team_email, subject, message, image_path)