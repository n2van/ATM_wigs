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
import csv

# Import FaceShapePredictor từ detection.py
from detection import FaceShapePredictor
# Import FaceWigRecommender từ face_analyzer.py
from face_analyzer import FaceWigRecommender

from customer_service import CustomerService
from email_service import EmailService

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
    # Dọn dẹp các file cũ trong thư mục tmp
    from cleanup_tmp import cleanup_old_files
    cleanup_old_files("./tmp", hours_threshold=3)
else:
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

# Khởi tạo dịch vụ
customer_service = CustomerService()
email_service = EmailService()

# Hàm lưu thông tin khách hàng và gửi email thông báo
def save_customer_info(customer_name, customer_phone, customer_email, product_code, notes, face_shape, image_path=None):
    if not customer_name or not customer_phone:
        return "Vui lòng nhập đầy đủ tên và số điện thoại khách hàng"
    
    # Lưu thông tin khách hàng
    success = customer_service.save_customer_info(
        customer_name=customer_name,
        customer_phone=customer_phone,
        customer_email=customer_email,
        face_shape=face_shape,
        product_code=product_code,
        notes=notes
    )
    
    # Gửi email thông báo cho team sale
    try:
        email_service.send_customer_notification(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            face_shape=face_shape,
            product_code=product_code,
            notes=notes,
            sales_team_email="sansinglong71@gmail.com"
        )
    except Exception as e:
        print(f"Lỗi khi gửi email: {str(e)}")
    
    if success:
        return "Đã lưu thông tin của bạn thành công! Team sale sẽ liên hệ với bạn sớm nhất."
    else:
        return "Không thể lưu thông tin. Vui lòng thử lại sau."

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

# Hàm cập nhật hiển thị tóc giả dựa trên kết quả phân tích
def update_wig_examples(face_shape_result):
    if face_shape_result and "Hình dạng khuôn mặt:" in face_shape_result:
        # Trích xuất hình dạng khuôn mặt từ kết quả
        for shape in face_shapes:
            if shape in face_shape_result:
                # Tải tóc giả từ thư mục tương ứng với hình dạng khuôn mặt
                return wig_recommender.get_wigs_for_face_shape(shape)
    
    # Mặc định hiển thị tất cả tóc giả nếu không phân tích được khuôn mặt
    return wig_recommender.get_all_wigs()

def update_dropdown(gallery_images):
    if gallery_images and isinstance(gallery_images, list) and len(gallery_images) > 0:
        # Tạo danh sách các tùy chọn: (label: "Wig #N", value: đường dẫn)
        choices = [{"label": f"Wig #{i+1}", "value": i} for i in range(len(gallery_images))]
        return gr.Dropdown.update(
            choices=choices,
            value=None,
            visible=True
        )
    return gr.Dropdown.update(visible=False)

# Cập nhật refresh wigs button
def refresh_wigs():
    try:
        wigs = wig_recommender.get_all_wigs()
        if not wigs or not isinstance(wigs, list):
            print("No wigs found or invalid result from get_all_wigs")
            wigs = []
        print(f"Refreshed wigs: {len(wigs)} found")
        return wigs
    except Exception as e:
        print(f"Error in refresh_wigs: {str(e)}")
        return []

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

# --- CSS tùy chỉnh ---
custom_css = """
// ... existing code ...
"""

# Sử dụng theme đơn giản cho các phiên bản Gradio cũ
theme = gr.themes.Base(primary_hue="blue", secondary_hue="blue")

# Đơn giản hóa WigSelector để áp dụng trực tiếp
class WigSelector:
    def __init__(self):
        self.selected_wig = None
    
    def select_wig_from_gallery(self, evt, gallery):
        """Hàm này vừa chọn wig vừa trả về đường dẫn để hiển thị luôn"""
        try:
            # Xử lý click event
            index = None
            
            # Thử các cách khác nhau để lấy index
            if isinstance(evt, int):
                index = evt
            elif hasattr(evt, 'index'):
                index = evt.index 
            elif isinstance(evt, dict) and 'index' in evt:
                index = evt['index']
            else:
                try:
                    index = int(evt)
                except:
                    print(f"Debug - Cannot parse index from event: {evt}")
                    return None
            
            # Kiểm tra index và gallery
            if isinstance(gallery, list) and 0 <= index < len(gallery):
                self.selected_wig = gallery[index]
                print(f"Selected and applied wig at index {index}: {self.selected_wig}")
                return self.selected_wig
            elif isinstance(gallery, list) and len(gallery) > 0:
                # Nếu không tìm thấy index, trả về ảnh đầu tiên
                self.selected_wig = gallery[0]
                print(f"Fallback: Applied first wig: {self.selected_wig}")
                return self.selected_wig
            else:
                print(f"Invalid gallery: {type(gallery)}")
                return None
        except Exception as e:
            print(f"Error selecting wig: {str(e)}")
            return None

# Khởi tạo WigSelector
wig_selector = WigSelector()

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
                dest_img = gr.Image(height=450, elem_classes=["original-image", "image-container"])  
                
                # Thêm phân tích hình dạng khuôn mặt - chỉ giữ nút phân tích
                analyze_btn = gr.Button("Analyze Face Shape", elem_classes=["try-on-button"])
                
                # Ẩn kết quả phân tích (để sử dụng trong backend)
                face_shape_result = gr.Textbox(visible=False)
            
            # Input Column - Wigs
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-title">Wigs</div>')
                image_input = gr.Image(type="filepath", height=450, elem_classes=["wig-image", "image-container"])
                
                # Hiển thị hình ảnh tóc giả mẫu
                gr.Markdown('<div class="section-title">Example Wigs</div>')
                # Khởi tạo gallery với list rỗng (không hiển thị ảnh nào)
                wig_gallery = gr.Gallery(
                    value=[], 
                    label="Example Wigs", 
                    height=200,
                    columns=5,
                    elem_classes=["gallery-container"]
                )
                
                # Thêm thông báo hướng dẫn
                wig_gallery_placeholder = gr.Markdown(
                    '<div style="text-align: center; padding: 20px; background-color: #f0f9ff; border: 2px dashed #a0c8ff; border-radius: 8px;">Hãy tải lên hình ảnh khuôn mặt và nhấn "Analyze Face Shape" để xem các tóc giả phù hợp</div>',
                    visible=True
                )
                
                # Thêm nút làm mới danh sách tóc giả
                refresh_wigs_btn = gr.Button("Refresh Wigs", elem_classes=["try-on-button"])
        
        # Hàng thứ hai: Kết quả phân tích khuôn mặt và Kết quả thử tóc
        with gr.Row():
            # Output Column - Face Analysis
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-title">Face Analysis</div>')
                face_analysis_output = gr.Markdown(elem_classes=["face-analysis"])
            
            # Output Column - Try-on Result
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-title">Try-on Result</div>')
                result_image = gr.Image(height=450, elem_classes=["result-image", "image-container"])
                
                # Nút thử tóc
                try_on_btn = gr.Button("Try On Wig", elem_classes=["try-on-button"])
        
        # Hàng thứ ba: Thông tin khách hàng
        with gr.Row():
            with gr.Column():
                gr.Markdown('<div class="section-title">Customer Information</div>')
                with gr.Row():
                    customer_name = gr.Textbox(label="Họ và tên", placeholder="Nhập họ và tên của bạn")
                    customer_phone = gr.Textbox(label="Số điện thoại", placeholder="Nhập số điện thoại của bạn")
                    customer_email = gr.Textbox(label="Email", placeholder="Nhập email của bạn (không bắt buộc)")
                
                with gr.Row():
                    product_code = gr.Textbox(label="Mã sản phẩm", placeholder="Nhập mã sản phẩm bạn quan tâm (không bắt buộc)")
                    notes = gr.Textbox(label="Ghi chú", placeholder="Thêm ghi chú (không bắt buộc)")
                
                save_info_btn = gr.Button("Lưu thông tin", elem_classes=["try-on-button"])
                save_result = gr.Markdown("")
        
        # Kết nối các sự kiện
        # 1. Phân tích khuôn mặt khi nhấn nút Analyze
        analyze_btn.click(
            fn=lambda img: wig_recommender.analyze_face_shape(img)[1],  # Lấy result_text từ kết quả
            inputs=dest_img,
            outputs=face_shape_result
        ).then(
            fn=lambda result: result,  # Hiển thị kết quả phân tích
            inputs=face_shape_result,
            outputs=face_analysis_output
        ).then(
            fn=update_wig_examples,  # Cập nhật gallery dựa trên kết quả phân tích
            inputs=face_shape_result,
            outputs=wig_gallery
        ).then(
            fn=lambda: gr.update(visible=False),  # Ẩn placeholder khi đã có kết quả
            outputs=wig_gallery_placeholder
        )
        
        # 2. Làm mới danh sách tóc giả khi nhấn nút Refresh
        refresh_wigs_btn.click(
            fn=refresh_wigs,
            outputs=wig_gallery
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=wig_gallery_placeholder
        )
        
        # 3. Chọn tóc giả từ gallery
        wig_gallery.select(
            fn=wig_selector.select_wig_from_gallery,
            inputs=[wig_gallery, wig_gallery],
            outputs=image_input
        )
        
        # 4. Thử tóc khi nhấn nút Try On
        try_on_btn.click(
            fn=run_image,
            inputs=[dest_img, image_input],
            outputs=result_image
        )
        
        # 5. Lưu thông tin khách hàng
        save_info_btn.click(
            fn=save_customer_info,
            inputs=[customer_name, customer_phone, customer_email, product_code, notes, face_shape_result],
            outputs=save_result
        )
    
    # --- ABOUT TAB ---
    with gr.Tab("About"):
        gr.Markdown("""
        # ATMwigs - Virtual Try-on System for Wigs
        
        ATMwigs là hệ thống thử tóc giả ảo, giúp khách hàng có thể thử các mẫu tóc giả khác nhau trước khi quyết định mua.
        
        ## Cách sử dụng
        
        1. Tải lên hình ảnh khuôn mặt của bạn
        2. Nhấn "Analyze Face Shape" để phân tích hình dạng khuôn mặt
        3. Chọn một mẫu tóc giả từ danh sách được đề xuất
        4. Nhấn "Try On Wig" để xem kết quả
        5. Nếu bạn hài lòng, hãy điền thông tin liên hệ và nhấn "Lưu thông tin"
        
        ## Liên hệ
        
        Để biết thêm thông tin, vui lòng liên hệ:
        
        - Email: sansinglong71@gmail.com
        - Điện thoại: 0123456789
        - Website: [atmwigs.com](https://atmwigs.com)
        """)
    
    # --- FOOTER ---
    gr.HTML("""
    <div class="footer">
        <p>© 2023 ATMwigs. All rights reserved. Developed by Van Nguyen.</p>
    </div>
    """)

# Khởi chạy ứng dụng
if args.ngrok is not None:
    public_url = ngrok.connect(addr=args.server_port, authtoken=args.ngrok, region=args.ngrok_region)
    print(f"Public URL: {public_url}")

demo.launch(
    server_name=args.server_name,
    server_port=args.server_port,
    share=args.share_gradio
)
