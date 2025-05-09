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

# Import FaceShapePredictor từ detection.py
from detection import FaceShapePredictor
# Import FaceWigRecommender từ face_analyzer.py
from face_analyzer import FaceWigRecommender

print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("Development by Van Nguyen") + "\033[0m")

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

# Tạo các thư mục con cho từng kiểu khuôn mặt
face_shapes = ["Heart", "Oblong", "Oval", "Round", "Square"]
for shape in face_shapes:
    face_shape_folder = f"./example_wigs/{shape}"
    if not os.path.exists(face_shape_folder):
        os.makedirs(face_shape_folder)
        print(f"Đã tạo thư mục '{face_shape_folder}' cho kiểu khuôn mặt {shape}.")

# Hàm tải các hình ảnh tóc giả mẫu - có thể bỏ và dùng wig_recommender.get_all_wigs()
def load_example_wigs():
    return wig_recommender.get_all_wigs()

# Hàm tải hình ảnh tóc giả theo hình dạng khuôn mặt - có thể bỏ và dùng wig_recommender.get_wigs_for_face_shape()
def load_wigs_for_face_shape(face_shape):
    return wig_recommender.get_wigs_for_face_shape(face_shape)

# Parse arguments
parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, default=1)  # Changed from 8 to 1
parser.add_argument("--force_cpu", default=False, action="store_true")
parser.add_argument("--share_gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=1356)
parser.add_argument("--colab_performance", default=False, action="store_true")
parser.add_argument("--ngrok", type=str, default=None)
parser.add_argument("--ngrok_region", type=str, default="us")
parser.add_argument("--face_model", type=str, default="best_model.pth")
args = parser.parse_args()

# Initialize
refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)
num_faces = args.max_num_faces  # This will now be 1

# Khởi tạo bộ nhận dạng hình dạng khuôn mặt
face_predictor = FaceShapePredictor(args.face_model)
# Khởi tạo bộ đề xuất tóc giả
wig_recommender = FaceWigRecommender(face_predictor)

def create_dummy_image():
    dummy = Image.new('RGB', (1, 1), color=(255, 255, 255))
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
    dummy.save(temp_file.name)
    return temp_file.name

def run_image(image_path, destination):
    # Simplified for single wig mode
    face_mode = "Single Face"
    partial_reface_ratio = 0.0
    disable_similarity = True
    multiple_faces_mode = False

    faces = []
    if destination is not None:
        faces.append({
            'origin': None,
            'destination': destination,
            'threshold': 0.0
        })

    return refacer.reface_image(image_path, faces, disable_similarity=disable_similarity, 
                               multiple_faces_mode=multiple_faces_mode, 
                               partial_reface_ratio=partial_reface_ratio)

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

# Hàm phân tích khuôn mặt - có thể bỏ vì đã có hàm tương tự trong FaceWigRecommender
def analyze_face_shape(image):
    face_shape, result_text = wig_recommender.analyze_face_shape(image)
    return result_text

# Hàm cập nhật hiển thị tóc giả dựa trên kết quả phân tích
def update_wig_examples(face_shape_result):
    if face_shape_result and "Hình dạng khuôn mặt:" in face_shape_result:
        # Trích xuất hình dạng khuôn mặt từ kết quả
        for shape in face_shapes:
            if shape in face_shape_result:
                # Tải tóc giả từ thư mục tương ứng với hình dạng khuôn mặt
                wigs = wig_recommender.get_wigs_for_face_shape(shape)
                return wigs
    
    # Mặc định hiển thị tất cả tóc giả nếu không phân tích được khuôn mặt
    all_wigs = wig_recommender.get_all_wigs()
    return all_wigs

# Hàm lấy đường dẫn wig đã chọn
def get_selected_wig(evt: gr.SelectData, gallery):
    if evt.index < len(gallery):
        return gallery[evt.index]
    return None

# Hàm load wig example để hiển thị trong Select Wigs
def load_wig_example(example_path):
    return example_path

# Hàm phân tích khuôn mặt và hiển thị các tóc giả phù hợp
def analyze_and_recommend(image):
    face_shape, result_text = wig_recommender.analyze_face_shape(image)
    if face_shape:
        # Lấy danh sách tóc giả phù hợp
        wigs = wig_recommender.get_wigs_for_face_shape(face_shape)
        return wigs if wigs else []
    return wig_recommender.get_all_wigs()

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

.face-analysis {
    background-color: #f0f9ff;
    border: 1px solid #a0c8ff;
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
    font-size: 1rem;
}

.face-recommendation {
    background-color: #f0fff4;
    border: 1px solid #a0ffc8;
    border-radius: 8px;
    padding: 12px;
    margin-top: 10px;
    font-size: 1rem;
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

.example-item-btn {
    width: 100%;
    padding: 4px 8px !important;
    margin-top: 5px !important;
    font-size: 0.8rem !important;
    text-align: center;
    background-color: #0e1b4d !important;
    color: white !important;
    border-radius: 4px;
    cursor: pointer;
}

.example-item-btn:hover {
    background-color: #1a2e6c !important;
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
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-title">Original Face</div>')
                dest_img = gr.Image(height=400)  
                
                # Thêm phân tích hình dạng khuôn mặt - chỉ giữ nút phân tích
                analyze_btn = gr.Button("Analyze Face Shape")
                
                # Ẩn kết quả phân tích (để sử dụng trong backend)
                face_shape_result = gr.Textbox(visible=False)
            
            # Input Column - Wigs
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-title">Wigs</div>')
                # Đặt image_input là readonly để người dùng không thể upload
                image_input = gr.Image(type="filepath", height=400, interactive=False)
                
                # Hiển thị hình ảnh tóc giả mẫu
                gr.Markdown('<div class="section-title">Example Wigs</div>')
                # Ban đầu không hiển thị ảnh nào (truyền list rỗng)
                wig_gallery = gr.Gallery(value=[], label="Example Wigs", height=200)
                wig_gallery_placeholder = gr.Markdown("Analyze your face first to see suitable wigs")
                
                # Nút để làm mới tóc giả (hiển thị tất cả)
                refresh_wigs_btn = gr.Button("Show All Wigs")
        
        # Hàng thứ hai: Nút Try On Wig
        with gr.Row():
            image_btn = gr.Button("Try On Wig")
        
        # Hàng thứ ba: Result
        with gr.Row():
            # Output Column - Ở giữa để cân bằng giao diện
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-title">Result</div>')
                image_output = gr.Image(interactive=False, type="filepath", height=400)
        
        # Connect events
        # Nút phân tích khuôn mặt và hiển thị tóc giả phù hợp
        analyze_btn.click(
            fn=wig_recommender.analyze_face_shape,
            inputs=[dest_img],
            outputs=[face_shape_result]
        ).then(
            fn=update_wig_examples,
            inputs=[face_shape_result],
            outputs=[wig_gallery]
        ).then(
            # Khi gallery cập nhật, ẩn placeholder text
            fn=lambda: "",
            inputs=[],
            outputs=[wig_gallery_placeholder]
        )
        
        # Nút làm mới tóc giả (hiển thị tất cả)
        refresh_wigs_btn.click(
            fn=lambda: wig_recommender.get_all_wigs(),
            inputs=[],
            outputs=[wig_gallery]
        ).then(
            # Khi gallery cập nhật, ẩn placeholder text
            fn=lambda: "",
            inputs=[],
            outputs=[wig_gallery_placeholder]
        )
        
        # Khi chọn tóc giả từ gallery - dùng event select
        def select_wig(evt, gallery):
            if evt and hasattr(evt, 'index') and evt.index < len(gallery):
                return gallery[evt.index]
            return None
            
        wig_gallery.select(
            fn=select_wig,
            inputs=[wig_gallery],
            outputs=[image_input]
        )
        
        # Try on wig
        image_btn.click(
            fn=run_image,
            inputs=[image_input, dest_img],
            outputs=image_output
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
