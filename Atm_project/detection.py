import sys
import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import numpy as np

class FaceShapePredictor:
    def __init__(self, model_path="best_model.pth"):
        # Khởi tạo các lớp mặt
        self.class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        
        # Tải model
        try:
            self.model = self.load_model(model_path)
            print("Đã tải model thành công!")
        except Exception as e:
            print(f"Lỗi: Không thể tải model: {e}")
            sys.exit(1)

    def load_model(self, model_path):
        # Khởi tạo mô hình
        model = torchvision.models.efficientnet_b4(pretrained=False)
        # Thay đổi lớp classifier
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(model.classifier[1].in_features, len(self.class_names))
        )
        
        # Tải trọng số từ file PTH
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Chế độ evaluation
        model.eval()
        
        return model
    
    def predict(self, image_path=None, image=None):
        try:
            # Tạo transform
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Đọc và xử lý ảnh
            if image_path:
                if not os.path.exists(image_path):
                    print(f"Lỗi: File ảnh '{image_path}' không tồn tại!")
                    return None
                image = Image.open(image_path).convert('RGB')
            elif image is not None:
                # Nếu đã truyền vào một đối tượng ảnh (từ app.py)
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image).convert('RGB')
            else:
                print("Lỗi: Cần cung cấp đường dẫn ảnh hoặc đối tượng ảnh!")
                return None
                
            input_tensor = transform(image).unsqueeze(0)
            
            # Dự đoán
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                _, predicted = torch.max(output, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence = probabilities[predicted.item()].item()
            
            # Lấy danh sách xác suất của tất cả các lớp
            probs = probabilities.cpu().numpy()
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": {
                    self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
                }
            }
            
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            return None

def main():
    if len(sys.argv) < 2:
        # Mặc định sử dụng best_model.pth nếu không có tham số
        model_path = "best_model.pth"
        if len(sys.argv) == 2:
            image_path = sys.argv[1]
        else:
            print("Sử dụng: python detection.py [<đường_dẫn_tới_model.pth>] <đường_dẫn_tới_ảnh>")
            sys.exit(1)
    else:
        model_path = sys.argv[1]
        if len(sys.argv) < 3:
            print("Sử dụng: python detection.py <đường_dẫn_tới_model.pth> <đường_dẫn_tới_ảnh>")
            sys.exit(1)
        image_path = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"Lỗi: File model '{model_path}' không tồn tại!")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Lỗi: File ảnh '{image_path}' không tồn tại!")
        sys.exit(1)
    
    predictor = FaceShapePredictor(model_path)
    result = predictor.predict(image_path=image_path)
    
    if result:
        print(f"\nKết quả dự đoán:")
        print(f"- Hình dạng khuôn mặt: {result['predicted_class']}")
        print(f"- Độ tin cậy: {result['confidence']:.2%}")
        
        print("\nXác suất của từng lớp:")
        for face_shape, prob in result['probabilities'].items():
            print(f"- {face_shape}: {prob:.2%}")

if __name__ == "__main__":
    main()
