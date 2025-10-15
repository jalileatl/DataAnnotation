import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pydicom
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import zipfile
import json
import datetime
import os
import requests
import uuid
import pickle
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SESSIONS_DIR = "sessions"
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.file_bytes = None
    st.session_state.ds = None
    st.session_state.image_orig = None
    st.session_state.image_display = None
    st.session_state.disp_size = (500, 500)
    st.session_state.mask_history = []
    st.session_state.current_mask = None
    st.session_state.chat_history = []
    st.session_state.radiopaedia_cases = []
    st.session_state.zoom_level = 1.0
    st.session_state.text_annotations = []
    st.session_state.rotation = 0
    st.session_state.flip_horizontal = False
    st.session_state.flip_vertical = False
    st.session_state.saved_states = []
    st.session_state.file_type = None
    st.session_state.filename = None
    st.session_state.session_id = None
    st.session_state.drawing_mode = "rect"
    st.session_state.stroke_width = 3
    st.session_state.stroke_color = "#FF0000"
    st.session_state.text_color = "#FF0000"
    st.session_state.show_chat = False
    st.session_state.dark_mode = False

# Helper functions
def image_to_pil(file_bytes):
    """Convert PNG/JPG bytes to PIL Image."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img, None
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

def dicom_to_image(file_bytes):
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(float)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        if np.nanmin(arr) != np.nanmax(arr):
            arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        return img, ds
    except Exception as e:
        raise ValueError(f"Failed to process DICOM: {e}")

def resize_for_display(img, max_size=500):
    w, h = img.size
    if w > h:
        new_w = max_size
        new_h = int(max_size * h / w)
    else:
        new_h = max_size
        new_w = int(max_size * w / h)
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR), (new_w, new_h)

def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def add_text_to_image(img, text, position, font_size=16, color=(255, 0, 0), font_style="Arial"):
    try:
        img_w, img_h = img.size
        scale_factor = max(1, min(img_w, img_h) / 500)
        scaled_font_size = int(font_size * scale_factor)
        
        try:
            font = ImageFont.truetype(f"{font_style.lower()}.ttf", scaled_font_size)
        except:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(img)
    draw.text(position, text, fill=color, font=font)
    return img

def compute_mask_stats(mask_array):
    """Compute statistics for mask array."""
    if mask_array is None or not mask_array.any():
        return {"area_pixels": 0, "bbox": None, "percent": 0.0}
    
    ys, xs = np.where(mask_array > 0)
    if ys.size == 0:
        return {"area_pixels": 0, "bbox": None, "percent": 0.0}
    
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    area = int(np.sum(mask_array > 0))
    pct = area / mask_array.size * 100
    
    return {
        "area_pixels": area,
        "bbox": [x0, y0, x1, y1],
        "percent": pct
    }

def make_export_zip(image_bytes, mask_bytes, metadata, filename="image"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{filename}", image_bytes)
        z.writestr("mask.png", mask_bytes)
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
    buf.seek(0)
    return buf

def chat_with_openai(prompt, history, image_data=None, api_key=OPENAI_API_KEY):
    if not api_key:
        return "OpenAI API key not set."
    try:
        system_message = """
        You are an experienced radiologist assisting medical students in radiology image practice.
        Always provide helpful analysis and explanations.
        Answer in structured sections:
        1. Summary
        2. Observations
        3. Recommendations
        4. Teaching Points
        Use bullet points.
        """
        messages = [{"role": "system", "content": system_message}]
        messages.extend(history[-6:])
        
        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                            "detail": "low"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

def save_session(state_data):
    """Save current session to file and return session ID."""
    session_id = str(uuid.uuid4())[:8]
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.pkl")
    
    with open(session_file, 'wb') as f:
        pickle.dump(state_data, f)
    
    return session_id

# Page config
st.set_page_config(
    page_title="Medical Image Annotation Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
    .stats-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Medical Image Annotation Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Medical image annotation and analysis platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Dark mode toggle
    st.session_state.dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    
    st.divider()
    
    # File upload
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a DICOM, PNG, or JPG file",
        type=['dcm', 'png', 'jpg', 'jpeg'],
        help="Upload a medical image for annotation"
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        file_ext = filename.lower().split('.')[-1]
        
        try:
            if file_ext in ['png', 'jpg', 'jpeg']:
                img_orig, ds = image_to_pil(file_bytes)
                st.session_state.file_type = file_ext
                modality = file_ext.upper()
            elif file_ext == 'dcm':
                img_orig, ds = dicom_to_image(file_bytes)
                st.session_state.file_type = "dcm"
                modality = getattr(ds, 'Modality', 'DICOM')
            
            img_disp, disp_size = resize_for_display(img_orig)
            
            st.session_state.file_bytes = file_bytes
            st.session_state.ds = ds
            st.session_state.image_orig = img_orig
            st.session_state.image_display = img_disp
            st.session_state.disp_size = disp_size
            st.session_state.filename = filename
            
            st.success(f"✅ Loaded: {filename}")
            st.info(f"Type: {modality} | Size: {img_orig.size[0]}x{img_orig.size[1]}")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    
    st.divider()
    
    # Drawing tools
    st.subheader("🎨 Drawing Tools")
    
    tool_options = {
        "Rectangle": "rect",
        "Circle": "circle",
        "Polygon": "polygon",
        "Freehand": "freedraw",
        "Line": "line"
    }
    
    selected_tool = st.selectbox(
        "Select Tool",
        options=list(tool_options.keys()),
        index=0
    )
    st.session_state.drawing_mode = tool_options[selected_tool]
    
    st.session_state.stroke_width = st.slider("Stroke Width", 1, 20, 3)
    st.session_state.stroke_color = st.color_picker("Stroke Color", "#FF0000")
    
    st.divider()
    
    # Image transformations
    st.subheader("🔄 Transformations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("↺ Rotate Left"):
            if st.session_state.image_orig:
                st.session_state.image_orig = st.session_state.image_orig.rotate(90, expand=True)
                st.session_state.rotation = (st.session_state.rotation - 90) % 360
                st.rerun()
        
        if st.button("↔️ Flip H"):
            if st.session_state.image_orig:
                st.session_state.image_orig = st.session_state.image_orig.transpose(Image.FLIP_LEFT_RIGHT)
                st.session_state.flip_horizontal = not st.session_state.flip_horizontal
                st.rerun()
    
    with col2:
        if st.button("↻ Rotate Right"):
            if st.session_state.image_orig:
                st.session_state.image_orig = st.session_state.image_orig.rotate(-90, expand=True)
                st.session_state.rotation = (st.session_state.rotation + 90) % 360
                st.rerun()
        
        if st.button("↕️ Flip V"):
            if st.session_state.image_orig:
                st.session_state.image_orig = st.session_state.image_orig.transpose(Image.FLIP_TOP_BOTTOM)
                st.session_state.flip_vertical = not st.session_state.flip_vertical
                st.rerun()
    
    st.divider()
    
    # Zoom controls
    st.subheader("🔍 Zoom")
    st.session_state.zoom_level = st.slider(
        "Zoom Level",
        min_value=0.5,
        max_value=3.0,
        value=st.session_state.zoom_level,
        step=0.1
    )
    st.info(f"Current: {int(st.session_state.zoom_level * 100)}%")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🖼️ Image Viewer")
    
    if st.session_state.image_display is not None:
        # Canvas for drawing
        canvas_width = int(st.session_state.disp_size[0] * st.session_state.zoom_level)
        canvas_height = int(st.session_state.disp_size[1] * st.session_state.zoom_level)
        
        # Convert PIL image to format suitable for canvas
        img_array = np.array(st.session_state.image_display)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=st.session_state.stroke_width,
            stroke_color=st.session_state.stroke_color,
            background_image=st.session_state.image_display,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=st.session_state.drawing_mode,
            key="canvas",
        )
        
        # Store canvas data
        if canvas_result.image_data is not None:
            st.session_state.current_mask = canvas_result.image_data
        
        # Action buttons
        st.divider()
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("💾 Save Annotation"):
                save_state = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "mask": st.session_state.current_mask,
                    "text_annotations": st.session_state.text_annotations.copy(),
                    "rotation": st.session_state.rotation,
                    "zoom_level": st.session_state.zoom_level
                }
                st.session_state.saved_states.append(save_state)
                st.success("✅ Annotation saved!")
        
        with btn_col2:
            if st.button("🗑️ Clear Canvas"):
                st.session_state.current_mask = None
                st.rerun()
        
        with btn_col3:
            if st.button("📤 Export ZIP"):
                if st.session_state.image_orig:
                    img_bytes = io.BytesIO()
                    st.session_state.image_orig.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    
                    mask_bytes = b""
                    if st.session_state.current_mask is not None:
                        mask_img = Image.fromarray(st.session_state.current_mask.astype(np.uint8))
                        mask_buf = io.BytesIO()
                        mask_img.save(mask_buf, format='PNG')
                        mask_bytes = mask_buf.getvalue()
                    
                    metadata = {
                        "filename": st.session_state.filename,
                        "file_type": st.session_state.file_type,
                        "image_size": st.session_state.image_orig.size,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "zoom_level": st.session_state.zoom_level,
                        "rotation": st.session_state.rotation
                    }
                    
                    zip_buf = make_export_zip(img_bytes, mask_bytes, metadata, st.session_state.filename)
                    
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=zip_buf.getvalue(),
                        file_name=f"annotation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
        
        with btn_col4:
            if st.button("🔗 Share Session"):
                if st.session_state.image_orig:
                    session_data = {
                        "file_bytes": st.session_state.file_bytes,
                        "image_orig": st.session_state.image_orig,
                        "current_mask": st.session_state.current_mask,
                        "text_annotations": st.session_state.text_annotations,
                        "zoom_level": st.session_state.zoom_level,
                        "rotation": st.session_state.rotation
                    }
                    session_id = save_session(session_data)
                    st.session_state.session_id = session_id
                    st.success(f"Session ID: {session_id}")
                    st.info(f"Share URL: http://localhost:8501/?session={session_id}")
    
    else:
        st.info("👆 Please upload an image file from the sidebar to begin annotation")

with col2:
    st.subheader("📊 Statistics")
    
    if st.session_state.current_mask is not None:
        stats = compute_mask_stats(st.session_state.current_mask)
        
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("Area (pixels)", f"{stats['area_pixels']:,}")
        st.metric("Coverage", f"{stats['percent']:.2f}%")
        
        if stats['bbox']:
            x0, y0, x1, y1 = stats['bbox']
            st.text(f"BBox: ({x0}, {y0}) to ({x1}, {y1})")
            st.text(f"Dimensions: {x1-x0} × {y1-y0} px")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No annotation created yet")
    
    st.divider()
    
    # Text annotation
    st.subheader("📝 Add Text")
    
    with st.expander("Text Annotation Settings"):
        text_input = st.text_area("Text", placeholder="Enter annotation text...")
        font_size = st.slider("Font Size", 10, 48, 16)
        text_color = st.color_picker("Text Color", "#FF0000")
        
        col_x, col_y = st.columns(2)
        with col_x:
            text_x = st.number_input("X Position", min_value=0, value=50)
        with col_y:
            text_y = st.number_input("Y Position", min_value=0, value=50)
        
        if st.button("Add Text to Image"):
            if text_input and st.session_state.image_orig:
                hex_color = text_color.lstrip('#')
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                
                img = st.session_state.image_orig.copy()
                img = add_text_to_image(img, text_input, (text_x, text_y), font_size, rgb_color)
                st.session_state.image_orig = img
                
                img_disp, disp_size = resize_for_display(img)
                st.session_state.image_display = img_disp
                st.session_state.disp_size = disp_size
                
                st.session_state.text_annotations.append({
                    "text": text_input,
                    "position": (text_x, text_y),
                    "font_size": font_size,
                    "color": text_color,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                st.success("✅ Text added!")
                st.rerun()
    
    st.divider()
    
    # AI Analysis
    st.subheader("🤖 AI Analysis")
    
    if st.button("🔍 Analyze Image"):
        st.session_state.show_chat = True
    
    if st.session_state.show_chat:
        with st.expander("AI Assistant", expanded=True):
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Assistant:** {msg['content']}")
            
            # Chat input
            user_input = st.text_input("Ask about the image...", key="chat_input")
            
            if st.button("Send") and user_input:
                if st.session_state.image_display:
                    img_b64 = pil_to_b64(st.session_state.image_display)
                    
                    with st.spinner("Analyzing..."):
                        response = chat_with_openai(
                            user_input,
                            st.session_state.chat_history,
                            img_b64
                        )
                    
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    st.rerun()

# Footer
st.divider()
st.markdown(
    '<p style="text-align: center; color: #666; font-size: 0.9rem;">Medical Image Annotation Dashboard | Built with Streamlit</p>',
    unsafe_allow_html=True
)
