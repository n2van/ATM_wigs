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
