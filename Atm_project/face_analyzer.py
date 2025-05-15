import os
import tempfile
import numpy as np
from PIL import Image

class FaceWigRecommender:
    def __init__(self, face_predictor):
        self.face_predictor = face_predictor
        self.face_shapes = ["Heart", "Oblong", "Oval", "Round", "Square"]
        
    def analyze_face_shape(self, image):
        """
        Phân tích hình dạng khuôn mặt từ ảnh đầu vào
        
        Args:
            image: Ảnh đầu vào (đường dẫn hoặc numpy array)
            
        Returns:
            face_shape: Hình dạng khuôn mặt dự đoán hoặc None nếu không phát hiện được
            result_text: Kết quả phân tích dưới dạng chuỗi văn bản
        """
        if image is None:
            return None, "Vui lòng tải lên ảnh khuôn mặt để nhận diện"
        
        # Lưu ảnh tạm thời nếu là một mảng numpy
        if isinstance(image, np.ndarray):
            temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
            Image.fromarray(image).save(temp_file.name)
            image_path = temp_file.name
        else:
            # Nếu là đường dẫn file
            image_path = image
        
        # Sử dụng phương thức predict
        try:
            if isinstance(image, Image.Image):
                result = self.face_predictor.predict(image=image)
            else:
                result = self.face_predictor.predict(image_path=image_path)
        except Exception as e:
            return None, f"Lỗi phân tích khuôn mặt: {str(e)}"
        finally:
            # Xóa file tạm nếu đã tạo
            if isinstance(image, np.ndarray) and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
        
        if result:
            face_shape = result["predicted_class"]
            confidence = result["confidence"]
            
            result_text = f"Hình dạng khuôn mặt: {face_shape} (Độ tin cậy: {confidence:.2%})"
            return face_shape, result_text
        else:
            return None, "Không thể phân tích hình dạng khuôn mặt"
    
    def get_wigs_for_face_shape(self, face_shape):
        """
        Lấy danh sách tóc giả dựa trên hình dạng khuôn mặt
        
        Args:
            face_shape: Hình dạng khuôn mặt dự đoán
            
        Returns:
            wigs: Danh sách đường dẫn đến các file tóc giả
        """
        if not face_shape or face_shape not in self.face_shapes:
            return self.get_wigs_for_face_shape("Oval")
        
        wigs = []
        face_shape_wig_folder = f"./example_wigs/{face_shape}"
        
        # Kiểm tra xem có thư mục tóc giả cho hình dạng khuôn mặt này không
        if os.path.exists(face_shape_wig_folder):
            for file in os.listdir(face_shape_wig_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    wigs.append(os.path.join(face_shape_wig_folder, file))
        
        # Nếu không có tóc giả trong thư mục hình dạng mặt, 
        # trả về tất cả các file
        if not wigs:
            wigs = self.get_wigs_for_face_shape("Oval")
        
        return wigs
    
