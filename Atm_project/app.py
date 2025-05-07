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
    num_faces = (len(args) - 2) // 2
    origins = list(args[:num_faces])
    destinations = list(args[num_faces:2*num_faces])
    face_mode = args[-2]
    partial_reface_ratio = args[-1]
    
    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    faces = []
    for k in range(num_faces):
        if destinations[k] is not None:
            faces.append({
                'origin': origins[k] if not multiple_faces_mode else None,
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

# Hàm để cập nhật hiển thị của face panels dựa trên chế độ
def update_face_visibility(mode):
    if mode == "Single Face":
        return [gr.update(visible=True)] + [gr.update(visible=False)] * 7
    else:
        return [gr.update(visible=True)] * 8

# --- UI với CSS tùy chỉnh ---
custom_css = """
:root {
    --main-color: #3b82f6;
    --secondary-color: #06b6d4;
    --background-color: #f8fafc;
    --panel-background: #ffffff;
    --text-color: #1e293b;
    --border-color: #e2e8f0;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
}

.header-container {
    text-align: center;
    padding: 20px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
}

.header-logo {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 10px;
}

.header-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: var(--main-color);
    margin-bottom: 5px;
}

.header-subtitle {
    font-size: 1.2rem;
    color: var(--text-color);
    opacity: 0.7;
}

.card {
    background-color: var(--panel-background);
    border-radius: 10px;
    box-shadow: var(--shadow);
    padding: 20px;
    margin-bottom: 20px;
}

.input-panel {
    background-color: var(--panel-background);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid var(--border-color);
}

.output-panel {
    background-color: var(--panel-background);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid var(--border-color);
}

.control-panel {
    background-color: var(--panel-background);
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    border: 1px solid var(--border-color);
}

.face-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.face-container {
    background-color: var(--panel-background);
    border-radius: 8px;
    padding: 10px;
    border: 1px solid var(--border-color);
    margin-bottom: 10px;
}

.section-title {
    font-weight: bold;
    font-size: 1.2rem;
    margin-bottom: 10px;
    color: var(--main-color);
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 5px;
    display: inline-block;
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid var(--border-color);
    font-size: 0.9rem;
    color: var(--text-color);
    opacity: 0.7;
}

button.primary {
    background-color: var(--main-color) !important;
    color: white !important;
}

button.primary:hover {
    background-color: var(--secondary-color) !important;
}

.tab-nav {
    border-bottom: 2px solid var(--border-color);
    margin-bottom: 20px;
}

.tab-nav button {
    font-weight: bold;
    padding: 10px 20px !important;
}

.tab-nav button.selected {
    color: var(--main-color) !important;
    border-bottom: 3px solid var(--main-color) !important;
}
"""

custom_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
)

with gr.Blocks(theme=custom_theme, css=custom_css, title="ATMwigs - Try-on Wigs") as demo:
    # Logo and Header
    try:
        with open("Logo.png", "rb") as f:
            icon_data = base64.b64encode(f.read()).decode()
        icon_html = f'<img src="data:image/png;base64,{icon_data}" style="width:120px;height:120px;">'
    except FileNotFoundError:
        icon_html = '<div style="font-size: 3rem; color: var(--main-color);">💇</div>'
    
    gr.HTML(f"""
    <div class="header-container">
        <div class="header-logo">{icon_html}</div>
        <div class="header-title">ATMwigs</div>
        <div class="header-subtitle">Virtual Try-on System for Wigs</div>
    </div>
    """)

    # --- IMAGE MODE ---
    with gr.Tab("Image Mode"):
        with gr.Row():
            # Input Column
            with gr.Column(scale=1, elem_classes="input-panel"):
                gr.Markdown('<div class="section-title">Input Image</div>')
                image_input = gr.Image(label="Original image", type="filepath", height=400)
            
            # Output Column
            with gr.Column(scale=1, elem_classes="output-panel"):
                gr.Markdown('<div class="section-title">Result</div>')
                image_output = gr.Image(label="Refaced image", interactive=False, type="filepath", height=400)
        
        # Controls
        with gr.Row(elem_classes="control-panel"):
            with gr.Column(scale=1):
                face_mode_image = gr.Radio(
                    ["Single Face", "Multiple Faces", "Faces By Match"], 
                    value="Single Face", 
                    label="Replacement Mode"
                )
            with gr.Column(scale=1):
                partial_reface_ratio_image = gr.Slider(
                    label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", 
                    minimum=0.0, 
                    maximum=0.5, 
                    value=0.0, 
                    step=0.1
                )
            with gr.Column(scale=1):
                image_btn = gr.Button("Process Image", variant="primary", size="lg")

        # Faces Panels in Tabs
        with gr.Tabs():
            with gr.TabItem("Face Configuration"):
                with gr.Row():
                    # Source Faces
                    with gr.Column(elem_classes="face-grid"):
                        gr.Markdown('<div class="section-title">Faces to Replace</div>')
                        face_panels = []
                        
                        # Origin faces
                        origin_images = []
                        for i in range(num_faces):
                            with gr.Column(visible=(i==0), elem_classes="face-container") as panel:
                                face_panels.append(panel)
                                origin_img = gr.Image(label=f"Face #{i+1} to replace", height=180)
                                origin_images.append(origin_img)
                    
                    # Destination Faces
                    with gr.Column(elem_classes="face-grid"):
                        gr.Markdown('<div class="section-title">Destination Faces</div>')
                        dest_panels = []
                        
                        # Destination faces
                        dest_images = []
                        for i in range(num_faces):
                            with gr.Column(visible=(i==0), elem_classes="face-container") as panel:
                                dest_panels.append(panel)
                                dest_img = gr.Image(label=f"Destination face #{i+1}", height=180)
                                dest_images.append(dest_img)
                                
        # Connect events
        all_inputs = [image_input]
        all_inputs.extend(origin_images)
        all_inputs.extend(dest_images)
        all_inputs.extend([face_mode_image, partial_reface_ratio_image])
        
        image_btn.click(
            fn=run_image,
            inputs=all_inputs,
            outputs=image_output
        )
        
        image_input.change(
            fn=distribute_faces,
            inputs=image_input,
            outputs=origin_images
        )
        
        image_input.change(
            fn=lambda _: 0.0,
            inputs=image_input,
            outputs=partial_reface_ratio_image
        )
        
        # Update face panel visibility based on mode
        face_mode_image.change(
            fn=update_face_visibility,
            inputs=face_mode_image,
            outputs=face_panels
        )
        
        face_mode_image.change(
            fn=update_face_visibility,
            inputs=face_mode_image,
            outputs=dest_panels
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
    demo.queue().launch(
        favicon_path="Logo.png" if os.path.exists("Logo.png") else None,
        show_error=True,
        share=args.share_gradio,
        server_name=args.server_name,
        server_port=args.server_port,
        enable_api=True
    )
