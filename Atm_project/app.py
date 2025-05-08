import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from refacer import Refacer
import argparse
import ngrok
import imageio
import numpy as np
from PIL import Image
import tempfile
import base64
import pyfiglet
import shutil
import time
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T

print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("Development by Van Nguyen") + "\033[0m")

# Face Shape Predictor class from detection.py
class FaceShapePredictor:
    def __init__(self, model_path):
        # Khởi tạo các lớp mặt
        self.class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        
        # Tải model
        try:
            self.model = self.load_model(model_path)
            print("Face shape detection model loaded successfully!")
        except Exception as e:
            print(f"Error: Cannot load face shape model: {e}")
            self.model = None

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
    
    def predict(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' does not exist!")
            return None
        
        if self.model is None:
            return None
        
        try:
            # Tạo transform
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Đọc và xử lý ảnh
            image = Image.open(image_path).convert('RGB')
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
            print(f"Error when predicting: {e}")
            return None

def cleanup_temp(folder_path):
    try:
        shutil.rmtree(folder_path)
        print("Gradio cache cleared successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Prepare temp folder
os.environ["GRADIO_TEMP_DIR"] = "./tmp"
if os.path.exists("./tmp"):
    cleanup_temp(os.environ['GRADIO_TEMP_DIR'])
if not os.path.exists("./tmp"):
    os.makedirs("./tmp")

# Tạo thư mục chứa các hình ảnh mẫu nếu chưa tồn tại
if not os.path.exists("./example_wigs"):
    os.makedirs("./example_wigs")
    print("Đã tạo thư mục 'example_wigs'. Vui lòng thêm các hình ảnh tóc giả mẫu vào thư mục này.")

# Default path for face shape detection model
DEFAULT_MODEL_PATH = "face_shape_model.pth"

# Hàm tải các hình ảnh tóc giả mẫu
def load_example_wigs():
    example_wigs = []
    wig_folder = "./example_wigs"  # Thư mục chứa các mẫu tóc giả
    
    if os.path.exists(wig_folder):
        for file in os.listdir(wig_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                example_wigs.append(os.path.join(wig_folder, file))
    
    # Nếu không tìm thấy file nào, trả về danh sách trống
    return example_wigs

# Parse arguments
parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, default=1)  # Changed from 8 to 1
parser.add_argument("--force_cpu", default=False, action="store_true")
parser.add_argument("--share_gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=1221)
parser.add_argument("--colab_performance", default=False, action="store_true")
parser.add_argument("--ngrok", type=str, default=None)
parser.add_argument("--ngrok_region", type=str, default="us")
parser.add_argument("--face_shape_model", type=str, default=DEFAULT_MODEL_PATH)
args = parser.parse_args()

# Initialize
refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)
num_faces = args.max_num_faces  # This will now be 1

# Initialize face shape predictor
face_shape_predictor = FaceShapePredictor(args.face_shape_model)

def create_dummy_image():
    dummy = Image.new('RGB', (1, 1), color=(255, 255, 255))
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
    dummy.save(temp_file.name)
    return temp_file.name

def load_image_from_path(image_path):
    """Load an image from a file path and return a numpy array."""
    if image_path is None:
        return None
    
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return None
    
    try:
        # Đọc hình ảnh bằng PIL
        image = Image.open(image_path).convert('RGB')
        # Chuyển đổi sang numpy array
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Lỗi khi đọc hình ảnh {image_path}: {e}")
        return None

def run_image(image_path, face_path):
    print(f"START run_image with: image_path={image_path}, face_path={face_path}")
    
    # Kiểm tra đầu vào
    if image_path is None or face_path is None:
        print("image_path hoặc face_path là None")
        return None
    
    # Đảm bảo cả hai đều là đường dẫn file hợp lệ
    if not os.path.exists(image_path) or not os.path.exists(face_path):
        print(f"File không tồn tại: image_path={image_path}, face_path={face_path}")
        return None
    
    # Lưu ảnh kết quả vào file tạm
    timestamp = int(time.time() * 1000)
    output_path = os.path.join("./tmp", f"result_{timestamp}.png")
    
    # Thông số cho xử lý ảnh
    disable_similarity = True
    multiple_faces_mode = False
    partial_reface_ratio = 0.0
    
    # Tạo danh sách faces cho refacer
    faces = [{
        'origin': None,
        'destination': face_path,
        'threshold': 0.0
    }]
    
    print(f"Created faces array: {faces}")
    
    try:
        # Mở và đọc file ảnh bằng PIL trước khi xử lý
        wig_img = Image.open(image_path).convert('RGB')
        wig_array = np.array(wig_img)
        face_img = Image.open(face_path).convert('RGB')
        face_array = np.array(face_img)
        
        print(f"Loaded wig image shape: {wig_array.shape}")
        print(f"Loaded face image shape: {face_array.shape}")
        
        # Lưu tạm các ảnh đã xử lý
        temp_wig_path = os.path.join("./tmp", f"temp_wig_{timestamp}.png")
        temp_face_path = os.path.join("./tmp", f"temp_face_{timestamp}.png")
        wig_img.save(temp_wig_path)
        face_img.save(temp_face_path)
        
        # Cập nhật đường dẫn cho faces
        faces[0]['destination'] = temp_face_path
        
        print("Calling refacer.reface_image...")
        
        # Gọi hàm reface_image với đường dẫn tạm thời
        result = refacer.reface_image(
            temp_wig_path, 
            faces, 
            disable_similarity=disable_similarity,
            multiple_faces_mode=multiple_faces_mode,
            partial_reface_ratio=partial_reface_ratio
        )
        
        print(f"Result type: {type(result)}")
        
        # Xử lý kết quả
        if result is not None:
            if isinstance(result, str) and os.path.exists(result):
                print(f"Returning result path: {result}")
                return result
            elif isinstance(result, np.ndarray):
                print(f"Saving numpy array to {output_path}")
                Image.fromarray(result).save(output_path)
                return output_path
            else:
                print(f"Unexpected result type: {type(result)}")
        else:
            print("Result is None")
        
        return None
    except Exception as e:
        print(f"Lỗi trong run_image: {e}")
        import traceback
        traceback.print_exc()
        return None

def detect_face_shape(image_path):
    print(f"START detect_face_shape with: image_path={image_path}")
    
    if image_path is None:
        print("No face uploaded")
        return "No face uploaded"
    
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return "File không tồn tại"
    
    try:
        # Đảm bảo image_path là đường dẫn hợp lệ
        result = face_shape_predictor.predict(image_path)
        if result is None:
            print("Không thể nhận diện hình dạng khuôn mặt")
            return "Không thể nhận diện hình dạng khuôn mặt"
        
        # Format the result
        face_shape = result["predicted_class"]
        confidence = result["confidence"] * 100
        
        output_text = f"Detected Face Shape: {face_shape} ({confidence:.1f}%)\n\n"
        output_text += "Probability Breakdown:\n"
        
        for shape, prob in result["probabilities"].items():
            output_text += f"- {shape}: {prob*100:.1f}%\n"
        
        print(f"Face shape detection result: {face_shape}")
        return output_text
    except Exception as e:
        print(f"Lỗi trong detect_face_shape: {e}")
        import traceback
        traceback.print_exc()
        return f"Lỗi khi nhận diện: {str(e)}"

def load_first_frame(filepath):
    if filepath is None:
        return None
    frames = imageio.get_reader(filepath)
    return frames.get_data(0)

def extract_faces_auto(filepath, refacer_instance, max_faces=1, isvideo=False):
    if filepath is None:
        return [None] * max_faces

    if isvideo and os.path.getsize(filepath) > 5 * 1024 * 1024:
        print("Video too large for auto-extract, skipping face extraction.")
        return [None] * max_faces

    frame = load_first_frame(filepath)
    if frame is None:
        return [None] * max_faces

    while len(frame.shape) > 3:
        frame = frame[0]

    if frame.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3 (RGB), but got {frame.shape[-1]}")

    temp_image_path = os.path.join("./tmp", f"temp_face_extract_{int(time.time() * 1000)}.png")
    Image.fromarray(frame).save(temp_image_path)

    try:
        faces = refacer_instance.extract_faces_from_image(temp_image_path, max_faces=max_faces)
        result = faces + [None] * (max_faces - len(faces))
        return result
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_image_path}: {e}")

# Simplified for single face
def distribute_faces(filepath):
    faces = extract_faces_auto(filepath, refacer, max_faces=1)
    return faces[0]

# Hàm load wig example để hiển thị trong Select Wigs
def load_wig_example(example_path):
    return example_path

# Check if face shape and image match - returns recommendation
def get_wig_recommendation(face_shape_text):
    print(f"START get_wig_recommendation with: {face_shape_text}")
    
    if face_shape_text is None or face_shape_text == "No face uploaded" or face_shape_text == "File không tồn tại" or face_shape_text == "Không thể nhận diện hình dạng khuôn mặt":
        return "Vui lòng tải lên ảnh khuôn mặt để nhận gợi ý tóc giả phù hợp."
    
    # Trích xuất hình dạng khuôn mặt từ văn bản phát hiện
    try:
        if "Detected Face Shape:" in face_shape_text:
            shape = face_shape_text.split("Detected Face Shape:")[1].split("(")[0].strip()
            print(f"Extracted face shape: {shape}")
        else:
            print("Không thể xác định hình dạng khuôn mặt từ kết quả nhận diện")
            return "Không thể xác định hình dạng khuôn mặt từ kết quả nhận diện."
        
        recommendations = {
            "Heart": "Đối với khuôn mặt trái tim, hãy thử tóc giả có kiểu tóc xếp lớp giúp tăng độ rộng ở vùng xương hàm. Kiểu tóc dài vừa phải với tóc mái rẽ một bên hoạt động rất tốt.",
            "Oblong": "Đối với khuôn mặt thuôn dài, hãy cân nhắc tóc giả có độ phồng ở hai bên để tạo độ rộng. Tránh kiểu tóc quá dài và hãy thử kiểu tóc có mái để rút ngắn khuôn mặt.",
            "Oval": "Đối với khuôn mặt oval, hầu hết các kiểu tóc giả đều phù hợp! Bạn có hình dạng khuôn mặt đa năng phù hợp với mọi độ dài hoặc phong cách.",
            "Round": "Đối với khuôn mặt tròn, hãy thử tóc giả có kiểu tóc xếp lớp hoặc bất đối xứng để tăng chiều cao. Tóc giả dài hơn với các lớp xung quanh khuôn mặt giúp kéo dài khuôn mặt.",
            "Square": "Đối với khuôn mặt vuông, tóc giả mềm mại với các lớp xung quanh khuôn mặt hoạt động rất tốt. Hãy thử kiểu tóc có mái rẽ một bên và tránh kiểu tóc bob cắt thẳng."
        }
        
        if shape in recommendations:
            return recommendations[shape]
        else:
            return "Không có gợi ý cụ thể cho hình dạng khuôn mặt này."
    except Exception as e:
        print(f"Lỗi trong get_wig_recommendation: {e}")
        import traceback
        traceback.print_exc()
        return "Đã xảy ra lỗi khi tạo gợi ý tóc giả."

# --- UI với CSS tùy chỉnh ---
custom_css = """
body {
    background-color: #f8fafc;
    color: #1e293b;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    background-color: #ffffff;
    border-top: 5px solid #0e1b4d; /* Chỉ viền trên với màu xanh navy */
    border-radius: 10px;
    box-shadow: 0 3px 20px rgba(14, 27, 77, 0.1);
    padding: 25px;
}

.header-container {
    padding: 20px;
    margin-bottom: 20px;
    background-color: #0e1b4d;
    color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    display: flex;
    align-items: center;
}

.header-logo {
    margin-right: 20px; /* Khoảng cách giữa logo và text */
}

.header-text {
    flex: 1;
}

.header-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.header-subtitle {
    font-size: 1.2rem;
    color: #bfdbfe;
}

.input-panel {
    background-color: #6194c7;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid #e2e8f0;
}

.output-panel {
    background-color: #6194c7;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #e2e8f0;
    margin: 0 auto; /* Giúp căn giữa panel */
    max-width: 800px; /* Giới hạn chiều rộng khi đứng một mình */
}

.detection-panel {
    background-color: #e6f0ff;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #a0c8ff;
    margin-top: 15px;
}

.recommendation-panel {
    background-color: #f0f9ff;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #bae6fd;
    margin-top: 10px;
}

.control-panel {
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.face-container {
    background-color: #6194c7;
    border-radius: 8px;
    padding: 10px;
    border: 1px solid #e2e8f0;
    margin-bottom: 10px;
}

.section-title {
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 10px;
    color: #0e1b4d; /* Giữ nguyên màu chữ xanh navy đậm */
    padding: 5px 10px; /* Thêm padding để tạo không gian cho khung */
    border: 2px solid #a0c8ff; /* Khung màu xanh dương nhạt */
    border-radius: 5px; /* Bo tròn góc khung */
    background-color: #e6f0ff; /* Nền xanh dương rất nhạt */
    display: inline-block;
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid #e2e8f0;
    font-size: 0.9rem;
    color: #1e293b;
    opacity: 0.7;
}

/* CSS cho gallery hình ảnh mẫu */
.example-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.example-item {
    cursor: pointer;
    border-radius: 5px;
    overflow: hidden;
    border: 2px solid transparent;
    transition: all 0.2s ease;
}

.example-item:hover {
    transform: scale(1.05);
    border-color: #0e1b4d;
    box-shadow: 0 0 10px rgba(14, 27, 77, 0.3);
}

.example-item img {
    width: 100%;
    height: 100px;
    object-fit: cover;
}

/* Thêm màu sắc navy cho các nút */
button.primary {
    background-color: #0e1b4d !important;
}

button.primary:hover {
    background-color: #1a2e6c !important;
}
"""

# Sử dụng theme đơn giản cho các phiên bản Gradio cũ
theme = gr.themes.Base(primary_hue="blue", secondary_hue="blue")

with gr.Blocks(theme=theme, css=custom_css, title="ATMwigs - Try-on Wigs") as demo:
    # Logo and Header
    try:
        with open("Logo.png", "rb") as f:
            icon_data = base64.b64encode(f.read()).decode()
        icon_html = f'<img src="data:image/png;base64,{icon_data}" style="width:100px;height:100px;">'
    except FileNotFoundError:
        icon_html = '<div style="font-size: 3rem; color: white;">💇</div>'
    
    gr.HTML(f"""
    <div class="header-container">
        <div class="header-logo">{icon_html}</div>
        <div class="header-text">
            <div class="header-title">ATMwigs</div>
            <div class="header-subtitle">Virtual Try-on System for Wigs</div>
        </div>
    </div>
    """)

    # --- IMAGE MODE ---
    with gr.Tab("Image Mode"):
        # Hàng đầu tiên: Original Face và Select Wigs
        with gr.Row():
            # Input Column - Face
            with gr.Column(scale=1, elem_classes="face-container"):
                gr.Markdown('<div class="section-title">Original Face</div>')
                dest_img = gr.Image(label="Input Face", height=400, type="filepath")
                
                # Add face detection button
                detect_btn = gr.Button("Detect Face Shape", variant="primary")
                
                # Output for face shape detection
                with gr.Column(elem_classes="detection-panel"):
                    face_shape_output = gr.Textbox(label="Face Shape Detection", lines=8)
                
                # Recommendations based on face shape
                with gr.Column(elem_classes="recommendation-panel"):
                    recommendation_output = gr.Textbox(label="Recommended Wig Styles", lines=4)
            
            # Input Column - Wigs
            with gr.Column(scale=1, elem_classes="input-panel"):
                gr.Markdown('<div class="section-title">Wigs</div>')
                image_input = gr.Image(label="Select Wigs", type="filepath", height=400)
                
                # Hiển thị hình ảnh tóc giả mẫu
                example_wigs = load_example_wigs()
                if example_wigs:
                    gr.Markdown('<div class="section-title">Example Wigs</div>')
                    with gr.Row(elem_classes="example-gallery"):
                        for wig in example_wigs:
                            wig_btn = gr.Button(
                                "",
                                elem_classes="example-item"
                            )
                            wig_btn.style(
                                full_width=False,
                                size="sm",
                                image=wig
                            )
                            # Khi nhấp vào một hình ảnh mẫu, load hình ảnh đó vào ô select wig
                            wig_btn.click(
                                fn=load_wig_example,
                                inputs=[],
                                outputs=[image_input],
                                _js=f"() => '{wig}'"
                            )
        
        # Hàng thứ hai: Nút Try On Wig
        with gr.Row(elem_classes="control-panel"):
            image_btn = gr.Button("Try On Wig", variant="primary", size="lg")
        
        # Hàng thứ ba: Result
        with gr.Row():
            # Output Column - Ở giữa để cân bằng giao diện
            with gr.Column(scale=1, elem_classes="output-panel"):
                gr.Markdown('<div class="section-title">Result</div>')
                image_output = gr.Image(label="After try-on", interactive=False, type="filepath", height=400)
        
        # Connect events - simplified for just one wig
        image_btn.click(
            fn=run_image,
            inputs=[image_input, dest_img],
            outputs=image_output
        )
        
        # Connect face detection
        detect_btn.click(
            fn=detect_face_shape,
            inputs=[dest_img],
            outputs=[face_shape_output]
        )
        
        # Connect recommendation generation
        detect_btn.click(
            fn=get_wig_recommendation,
            inputs=[face_shape_output],
            outputs=[recommendation_output]
        )

    # Footer
    gr.HTML("""
    <div class="footer">
        <p>© 2023 ATMwigs - All rights reserved</p>
        <p>Developed with ❤️ for virtual wig try-on</p>
    </div>
    """)

# --- ngrok connect (optional) ---
if args.ngrok and args.ngrok != "None":
    def connect(token, port, options):
        try:
            public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
            print(f'ngrok URL: {public_url}')
        except Exception as e:
            print(f'ngrok connection aborted: {e}')

    connect(args.ngrok, args.server_port, {'region': args.ngrok_region, 'authtoken_from_env': False})

# --- Launch app ---
if __name__ == "__main__":
    # Loại bỏ tham số enable_api vì không được hỗ trợ trong phiên bản cũ
    demo.queue().launch(
        favicon_path="Logo.png" if os.path.exists("Logo.png") else None,
        show_error=True,
        share=args.share_gradio,
        server_name=args.server_name,
        server_port=args.server_port
    )
    
    # Nếu cần tương thích API, hãy thêm message để hướng dẫn upgrade Gradio
    print("NOTE: To enable API functionality, upgrade Gradio to version 3.32.0 or higher.")
