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

print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("ATM wigs") + "\033[0m")

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

# Parse arguments
parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, default=8)
parser.add_argument("--force_cpu", default=False, action="store_true")
parser.add_argument("--share_gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=1234)
parser.add_argument("--colab_performance", default=False, action="store_true")
parser.add_argument("--ngrok", type=str, default=None)
parser.add_argument("--ngrok_region", type=str, default="us")
args = parser.parse_args()

# Initialize
refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)
num_faces = args.max_num_faces

def create_dummy_image():
    dummy = Image.new('RGB', (1, 1), color=(255, 255, 255))
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
    dummy.save(temp_file.name)
    return temp_file.name

def run_image(image_path, *args):
    # Extract arguments from *args
    # Bỏ qua origin_images, chỉ lấy destination_images
    destinations = list(args)
    
    # Sử dụng Single Face mode và Reface Ratio = 0 (mặc định)
    face_mode = "Single Face"  # Mặc định Single Face mode
    partial_reface_ratio = 0.0    # Mặc định Full Face
    
    disable_similarity = True  # Single Face => disable_similarity = True
    multiple_faces_mode = False # Single Face => multiple_faces_mode = False

    faces = []
    for k in range(len(destinations)):
        if destinations[k] is not None:
            faces.append({
                'origin': None,  # Không sử dụng origin trong Single Face mode
                'destination': destinations[k],
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

def extract_faces_auto(filepath, refacer_instance, max_faces=5, isvideo=False):
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

def distribute_faces(filepath):
    faces = extract_faces_auto(filepath, refacer, max_faces=num_faces)
    return faces[0], faces[1], faces[2], faces[3], faces[4], faces[5], faces[6], faces[7]

# --- UI với CSS tùy chỉnh ---
custom_css = """
body {
    background-color: #f8fafc;
    color: #1e293b;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
}

.header-container {
    padding: 20px;
    margin-bottom: 20px;
    background-color: #0e1b4d; /* Màu xanh navy đậm theo yêu cầu */
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
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid #e2e8f0;
}

.output-panel {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #e2e8f0;
}

.control-panel {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.face-container {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 10px;
    border: 1px solid #e2e8f0;
    margin-bottom: 10px;
}

.section-title {
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 10px;
    color: #0e1b4d; /* Màu xanh navy đậm cho tiêu đề */
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 5px;
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
        # Main layout
        with gr.Row():
            # Destination (Wigs) Column FIRST
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-title">Wigs to Try</div>')
                
                # Destination faces (shifted to the top)
                dest_images = []
                for i in range(num_faces):
                    with gr.Column(visible=True, elem_classes="face-container"):
                        dest_img = gr.Image(label=f"Wig #{i+1}", height=180)
                        dest_images.append(dest_img)
            
            # Middle column for Input
            with gr.Column(scale=2, elem_classes="input-panel"):
                gr.Markdown('<div class="section-title">Input Image</div>')
                image_input = gr.Image(label="Original face", type="filepath", height=400)
                
                # Process button
                with gr.Row(elem_classes="control-panel"):
                    image_btn = gr.Button("Try On Wig", variant="primary", size="lg")
            
            # Output Column
            with gr.Column(scale=2, elem_classes="output-panel"):
                gr.Markdown('<div class="section-title">Result</div>')
                image_output = gr.Image(label="After try-on", interactive=False, type="filepath", height=400)
        
        # Connect events
        all_inputs = [image_input]
        all_inputs.extend(dest_images)
        
        image_btn.click(
            fn=run_image,
            inputs=all_inputs,
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
