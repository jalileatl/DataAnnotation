import dash
from dash import html, dcc, Input, Output, State, callback_context
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
import pydicom
from PIL import Image, ImageDraw, ImageFont
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
import io
import base64
import zipfile
import json
import datetime
import os
import requests
import dash_bootstrap_components as dbc
from dotenv import load_dotenv
import uuid
import pickle

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SESSIONS_DIR = "sessions"
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

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
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def parse_canvas_data(json_data, height, width, eraser_strokes=None):
    """Parse canvas JSON data for all drawing tools, excluding eraser strokes."""
    if not json_data:
        return None
    
    mask_array = np.zeros((height, width), dtype=np.uint8)
    eraser_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Parse eraser strokes first
    if eraser_strokes:
        for eraser_data in eraser_strokes:
            try:
                eraser_parsed = parse_jsonstring(eraser_data, shape=(height, width))
                if eraser_parsed is not None and eraser_parsed.any():
                    eraser_mask = np.maximum(eraser_mask, (eraser_parsed > 0).astype(np.uint8))
            except:
                pass
    
    try:
        parsed_mask = parse_jsonstring(json_data, shape=(height, width))
        if parsed_mask is not None and parsed_mask.any():
            mask_array = np.maximum(mask_array, (parsed_mask > 0).astype(np.uint8) * 255)
    except:
        pass
    
    try:
        data = json.loads(json_data)
        objects_list = []
        if isinstance(data, dict):
            if 'objects' in data:
                objects_list = data['objects']
            elif 'type' in data:
                objects_list = [data]
        elif isinstance(data, list):
            objects_list = data
        
        for item in objects_list:
            if not isinstance(item, dict):
                continue
                
            obj_type = item.get('type', '').lower()
            
            if obj_type == 'rect':
                left = item.get('left', 0)
                top = item.get('top', 0)
                rect_width = item.get('width', 0)
                rect_height = item.get('height', 0)
                scale_x = item.get('scaleX', 1)
                scale_y = item.get('scaleY', 1)
                rect_width = int(rect_width * scale_x)
                rect_height = int(rect_height * scale_y)
                
                x0 = max(0, min(int(left), width - 1))
                y0 = max(0, min(int(top), height - 1))
                x1 = max(0, min(int(left + rect_width), width))
                y1 = max(0, min(int(top + rect_height), height))
                
                if x1 > x0 and y1 > y0:
                    mask_array[y0:y1, x0:x1] = 255
                    
            elif obj_type == 'circle':
                left = item.get('left', 0)
                top = item.get('top', 0)
                radius = item.get('radius', 0)
                scale_x = item.get('scaleX', 1)
                scale_y = item.get('scaleY', 1)
                radius_x = int(radius * scale_x)
                radius_y = int(radius * scale_y)
                
                cx = int(left + radius_x)
                cy = int(top + radius_y)
                cx = max(0, min(cx, width - 1))
                cy = max(0, min(cy, height - 1))
                
                y_coords, x_coords = np.ogrid[:height, :width]
                if radius_x > 0 and radius_y > 0:
                    mask_ellipse = ((x_coords - cx) ** 2 / radius_x ** 2 + 
                                   (y_coords - cy) ** 2 / radius_y ** 2) <= 1
                    mask_array[mask_ellipse] = 255
                    
            elif obj_type == 'line':
                x1 = int(item.get('x1', 0))
                y1 = int(item.get('y1', 0))
                x2 = int(item.get('x2', 0))
                y2 = int(item.get('y2', 0))
                draw_line_on_mask(mask_array, x1, y1, x2, y2)
                
            elif obj_type in ['path', 'polygon']:
                points = item.get('points', [])
                if points and len(points) > 1:
                    if isinstance(points[0], dict):
                        points = [(p.get('x', 0), p.get('y', 0)) for p in points]
                    draw_polygon_on_mask(mask_array, points)
                    
    except Exception as e:
        print(f"Error parsing canvas data: {e}")
    
    if eraser_mask.any():
        mask_array[eraser_mask > 0] = 0
    
    if mask_array.any():
        return mask_array
    
    return None

def draw_line_on_mask(mask, x1, y1, x2, y2, thickness=3):
    """Draw a line on the mask using Bresenham's algorithm."""
    height, width = mask.shape
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        for dy_offset in range(-thickness//2, thickness//2 + 1):
            for dx_offset in range(-thickness//2, thickness//2 + 1):
                py = y1 + dy_offset
                px = x1 + dx_offset
                if 0 <= px < width and 0 <= py < height:
                    mask[py, px] = 255
        
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def draw_polygon_on_mask(mask, points):
    """Draw a polygon on the mask."""
    height, width = mask.shape
    if len(points) < 2:
        return
    
    int_points = [(int(p[0]), int(p[1])) for p in points]
    
    for i in range(len(int_points)):
        x1, y1 = int_points[i]
        x2, y2 = int_points[(i + 1) % len(int_points)]
        draw_line_on_mask(mask, x1, y1, x2, y2)
    
    min_y = max(0, min(p[1] for p in int_points))
    max_y = min(height - 1, max(p[1] for p in int_points))
    
    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(len(int_points)):
            x1, y1 = int_points[i]
            x2, y2 = int_points[(i + 1) % len(int_points)]
            
            if min(y1, y2) <= y <= max(y1, y2):
                if y1 != y2:
                    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(int(x))
        
        if len(intersections) >= 2:
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x_start = max(0, min(intersections[i], width - 1))
                    x_end = max(0, min(intersections[i + 1], width - 1))
                    mask[y, x_start:x_end + 1] = 255

def compute_mask_stats(mask_array):
    """Compute statistics for any mask array regardless of the tool used."""
    if mask_array is None or not mask_array.any():
        return {"area_pixels": 0, "bbox": None, "percent": 0.0, "tool_type": "None"}
    
    ys, xs = np.where(mask_array > 0)
    if ys.size == 0:
        return {"area_pixels": 0, "bbox": None, "percent": 0.0, "tool_type": "Empty"}
    
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    area = int(np.sum(mask_array > 0))
    pct = area / mask_array.size * 100
    tool_type = determine_tool_type(mask_array, ys, xs)
    
    return {
        "area_pixels": area,
        "bbox": [x0, y0, x1, y1],
        "percent": pct,
        "tool_type": tool_type
    }

def determine_tool_type(mask_array, ys, xs):
    """Determine the likely tool type based on mask characteristics."""
    if ys.size == 0:
        return "Empty"
    
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    aspect_ratio = width / height if height > 0 else 1
    
    center_y, center_x = ys.mean(), xs.mean()
    distances = np.sqrt((ys - center_y)**2 + (xs - center_x)**2)
    mean_distance = distances.mean()
    std_distance = distances.std()
    circularity = 1 - (std_distance / mean_distance) if mean_distance > 0 else 0
    
    bbox_area = height * width
    fill_ratio = np.sum(mask_array > 0) / bbox_area if bbox_area > 0 else 0
    
    if circularity > 0.8 and fill_ratio > 0.7:
        return "Circle"
    elif circularity > 0.6 and fill_ratio < 0.3:
        return "Circle (outline)"
    elif aspect_ratio > 0.8 and aspect_ratio < 1.2 and fill_ratio > 0.7:
        return "Rectangle"
    elif aspect_ratio > 0.8 and aspect_ratio < 1.2 and fill_ratio < 0.3:
        return "Rectangle (outline)"
    elif fill_ratio > 0.5 and (aspect_ratio < 0.7 or aspect_ratio > 1.3):
        return "Rectangle"
    elif fill_ratio < 0.2:
        return "Line"
    elif np.sum(mask_array > 0) < 50:
        return "Freehand (small)"
    else:
        return "Polygon/Freehand"

def scale_bbox_display_to_original(bbox_disp, disp_size, orig_size):
    if bbox_disp is None:
        return None
    disp_w, disp_h = disp_size
    orig_w, orig_h = orig_size
    x0, y0, x1, y1 = bbox_disp
    sx = orig_w / disp_w
    sy = orig_h / disp_h
    left = int(round(x0 * sx))
    upper = int(round(y0 * sy))
    right = int(round((x1 + 1) * sx))
    lower = int(round((y1 + 1) * sy))
    left = max(0, min(left, orig_w - 1))
    upper = max(0, min(upper, orig_h - 1))
    right = max(left + 1, min(right, orig_w))
    lower = max(upper + 1, min(lower, orig_h))
    return (left, upper, right, lower)

def crop_original_by_bbox(orig_img, bbox_orig):
    if bbox_orig is None:
        return None
    try:
        return orig_img.crop(bbox_orig)
    except Exception:
        return None

def add_text_to_image(img, text, position, font_size=16, color=(255, 0, 0), font_style="Arial"):
    try:
        img_w, img_h = img.size
        scale_factor = max(1, min(img_w, img_h) / 500)
        scaled_font_size = int(font_size * scale_factor)
        
        font = ImageFont.truetype(f"{font_style.lower()}.ttf", scaled_font_size)
    except:
        try:
            if font_style.lower() == "times new roman":
                font = ImageFont.truetype("times.ttf", scaled_font_size)
            elif font_style.lower() == "courier new":
                font = ImageFont.truetype("cour.ttf", scaled_font_size)
            else:
                font = ImageFont.truetype("arial.ttf", scaled_font_size)
        except:
            font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    draw.text(position, text, fill=color, font=font)
    return img

def make_export_zip(image_bytes, mask_bytes, metadata, filename="image"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{filename}", image_bytes)
        z.writestr("mask.png", mask_bytes)
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
    buf.seek(0)
    return buf

def generate_annotation_report(state_data):
    """Generate a comprehensive annotation report in HTML format."""
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Annotation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #667eea; }}
            h2 {{ color: #764ba2; margin-top: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #667eea; color: white; }}
            .section {{ margin-bottom: 30px; }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Medical Image Annotation Report</h1>
        <p class="timestamp">Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Image Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Filename</td><td>{state_data.get('filename', 'N/A')}</td></tr>
                <tr><td>File Type</td><td>{state_data.get('file_type', 'N/A').upper()}</td></tr>
                <tr><td>Image Size</td><td>{state_data.get('image_size', 'N/A')}</td></tr>
                <tr><td>Display Size</td><td>{state_data.get('disp_size', 'N/A')}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Annotation Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Tool Type</td><td>{state_data.get('tool_type', 'N/A')}</td></tr>
                <tr><td>Area (pixels)</td><td>{state_data.get('area_pixels', 0)}</td></tr>
                <tr><td>Coverage</td><td>{state_data.get('percent', 0):.2f}%</td></tr>
                <tr><td>Bounding Box</td><td>{state_data.get('bbox', 'N/A')}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Text Annotations</h2>
            <table>
                <tr><th>Text</th><th>Position</th><th>Font Size</th><th>Color</th></tr>
                {''.join([f"<tr><td>{ann['text']}</td><td>{ann['position']}</td><td>{ann['font_size']}</td><td>{ann['color']}</td></tr>" 
                         for ann in state_data.get('text_annotations', [])])}
            </table>
        </div>
        
        <div class="section">
            <h2>Session History</h2>
            <p>Total saved states: {state_data.get('saved_states_count', 0)}</p>
            <p>Zoom level: {state_data.get('zoom_level', 1.0) * 100:.0f}%</p>
            <p>Rotation: {state_data.get('rotation', 0)}°</p>
        </div>
    </body>
    </html>
    """
    return report_html

def save_session(state_data):
    """Save current session to file and return session ID."""
    session_id = str(uuid.uuid4())[:8]
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.pkl")
    
    with open(session_file, 'wb') as f:
        pickle.dump(state_data, f)
    
    return session_id

def load_session(session_id):
    """Load session from file."""
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.pkl")
    
    if not os.path.exists(session_file):
        return None
    
    with open(session_file, 'rb') as f:
        return pickle.load(f)

def chat_with_openai(prompt, history, image_data=None, api_key=OPENAI_API_KEY):
    if not api_key:
        return "OpenAI API key not set."
    try:
        system_message = """
        You are an experienced radiologist assisting medical students in radiology image practice.
        Always provide helpful analysis and explanations.
        Never say you are unable to analyze or do something; instead, provide the best possible response based on available information.
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
            header, encoded = image_data.split(",", 1)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
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

def get_radiopaedia_cases(conversation, api_key=OPENAI_API_KEY):
    if not api_key:
        return []
    try:
        recent_messages = conversation[-4:] if len(conversation) >= 4 else conversation
        extract_prompt = """
        Based on this medical imaging conversation, extract 3-5 key medical terms or conditions that would be useful for finding similar cases on Radiopaedia.
        Return only the terms, one per line, without any additional text.
        
        Conversation:
        """
        for msg in recent_messages:
            extract_prompt += f"\n{msg['role']}: {msg['content']}"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": extract_prompt}],
            "temperature": 0.3,
            "max_tokens": 100
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        key_terms = response.json()['choices'][0]['message']['content'].strip().split('\n')
        key_terms = [term.strip() for term in key_terms if term.strip()]
        cases = []
        for i, term in enumerate(key_terms[:3]):
            case = {
                "id": f"case_{i+1}",
                "title": f"{term}: Search Radiopaedia",
                "author": f"Dr. {['Smith', 'Johnson', 'Williams'][i % 3]}",
                "specialty": ["Radiology", "Neuroradiology", "Musculoskeletal"][i % 3],
                "modality": ["CT", "MRI", "X-ray"][i % 3],
                "description": f"Search for cases related to {term} on Radiopaedia. Note: This is a mock case for demonstration purposes.",
                "image_url": f"https://radiopaedia.org/cases/{term.lower().replace(' ', '-')}/images/{(i+1)*12345}",
                "link": f"https://radiopaedia.org/search?lang=us&q={term.replace(' ', '+')}&scope=cases"
            }
            cases.append(case)
        return cases
    except Exception as e:
        print(f"Error generating Radiopaedia cases: {e}")
        return []

def format_radiopaedia_cases(cases):
    if not cases:
        return html.P("No related cases found. Start a conversation to see relevant cases.", className="text-muted small")
    case_cards = []
    for case in cases:
        case_card = dbc.Card([
            dbc.CardBody([
                html.H6(case["title"], className="card-title mb-1", style={"fontSize": "0.9rem"}),
                html.P([
                    html.Strong("Author: "),
                    html.Span(f"{case['author']} - {case['specialty']}", className="text-muted", style={"fontSize": "0.75rem"})
                ], className="mb-1"),
                html.P([
                    html.Strong("Modality: "),
                    html.Span(case["modality"], className="badge bg-info me-2", style={"fontSize": "0.7rem"})
                ], className="mb-1"),
                html.P(case["description"], className="card-text", style={"fontSize": "0.75rem"}),
                dbc.Button("Search", href=case["link"], target="_blank", color="primary", size="sm", className="mt-1", style={"fontSize": "0.75rem"})
            ], style={"padding": "0.75rem"})
        ], className="mb-2 shadow-sm")
        case_cards.append(case_card)
    return case_cards

def format_chat_text(text):
    lines = text.split("\n")
    children = []
    for line in lines:
        line = line.rstrip()
        if line.startswith("- ") or line.startswith("* "):
            children.append(html.Li(line[2:], style={"fontSize": "0.85rem"}))
        elif line and line[0].isdigit() and line[1:3] == ". ":
            children.append(html.H6(line, style={"fontWeight": "bold", "fontSize": "0.9rem"}))
        elif line:
            children.append(html.P(line, style={"fontSize": "0.85rem"}))
        else:
            children.append(html.Br())
    return children

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(id="app-container", children=[
    html.Link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css"
    ),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Switch(
                        id="theme-switch",
                        label="Dark Mode",
                        value=False,
                        className="float-end"
                    ),
                    html.H1("Medical Image Annotation Dashboard", className="text-center my-2", 
                           style={"fontWeight": "700", "fontSize": "1.5rem", "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                  "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent"}),
                    html.P("Medical image annotation and analysis platform", 
                           className="text-center mb-2", style={"fontSize": "0.9rem"})
                ])
            ])
        ]),
        
        dbc.Row([
            dbc.Col(id="left-column", md=9, children=[
                dbc.Card([
                    dbc.CardHeader("Medical Image Viewer", className="bg-gradient-primary text-white py-2", 
                                 style={"background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "fontSize": "0.95rem"}),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-dicom',
                            children=html.Div([
                                html.I(className="bi bi-cloud-upload me-2", style={"fontSize": "20px"}),
                                'Drag and Drop or ', 
                                html.A('Select a DICOM, PNG, or JPG file', style={"fontWeight": "bold", "color": "#667eea"})
                            ]),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
                                'textAlign': 'center', 'marginBottom': '15px', 'borderColor': '#667eea',
                                'backgroundColor': '#f8f9fa', 'transition': 'all 0.3s ease', 'fontSize': '0.9rem'
                            },
                            accept='.dcm,.png,.jpg,.jpeg'
                        ),
                        
                        html.Div(
                            className="toolbar mb-2 p-2 rounded border",
                            style={"overflowX": "auto", "whiteSpace": "nowrap", "backgroundColor": "#f8f9fa",
                                   "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"},
                            children=[
                                html.Div(
                                    className="btn-group me-2 mb-1",
                                    role="group",
                                    children=[
                                        dbc.Button(
                                            [html.I(className="bi bi-pentagon me-1"), "Polygon"],
                                            id="tool-polygon",
                                            color="primary",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Polygon Tool",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-square me-1"), "Rectangle"],
                                            id="tool-rectangle",
                                            color="primary",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Rectangle Tool",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-circle me-1"), "Circle"],
                                            id="tool-circle",
                                            color="primary",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Circle Tool",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-slash-lg me-1"), "Line"],
                                            id="tool-line",
                                            color="primary",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Line Tool",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-pencil me-1"), "Freehand"],
                                            id="tool-pen",
                                            color="primary",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Freehand Drawing",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-eraser me-1"), "Eraser"],
                                            id="tool-eraser",
                                            color="warning",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Eraser Tool",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="btn-group me-2 mb-1",
                                    role="group",
                                    children=[
                                        dbc.Button(
    "Labeling",
    id="tool-text",
    color="secondary",
    outline=True,
    size="sm",
    className="px-2",
    title="Add Label Annotation",
    style={"fontSize": "0.8rem"}
),


                                    ]
                                ),
                                html.Div(
                                    className="btn-group me-2 mb-1",
                                    role="group",
                                    children=[
                                        dbc.Button(
                                            [html.I(className="bi bi-zoom-in me-1"), "Zoom In"],
                                            id="zoom-in-btn",
                                            color="info",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Zoom In",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-zoom-out me-1"), "Zoom Out"],
                                            id="zoom-out-btn",
                                            color="info",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Zoom Out",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-arrows-fullscreen me-1"), "Fit"],
                                            id="fit-btn",
                                            color="info",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Fit to Screen",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="btn-group me-2 mb-1",
                                    role="group",
                                    children=[
                                        dbc.Button(
                                            [html.I(className="bi bi-arrow-counterclockwise me-1"), "Rotate L"],
                                            id="rotate-left-btn",
                                            color="warning",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Rotate Left",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-arrow-clockwise me-1"), "Rotate R"],
                                            id="rotate-right-btn",
                                            color="warning",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Rotate Right",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-symmetry-horizontal me-1"), "Flip H"],
                                            id="flip-h-btn",
                                            color="warning",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Flip Horizontal",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-symmetry-vertical me-1"), "Flip V"],
                                            id="flip-v-btn",
                                            color="warning",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Flip Vertical",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="btn-group mb-1",
                                    role="group",
                                    children=[
                                        dbc.Button(
                                            [html.I(className="bi bi-share me-1"), "Share"],
                                            id="share-btn",
                                            color="info",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Share Session",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-box me-1"), "3D View"],
                                            id="3d-view-btn",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="3D View",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-save me-1"), "Save"],
                                            id="save-btn",
                                            color="success",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Save Annotation",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                        dbc.Button(
                                            [html.I(className="bi bi-trash me-1"), "Clear"],
                                            id="clear-btn",
                                            color="danger",
                                            outline=True,
                                            size="sm",
                                            className="px-2",
                                            title="Clear All Annotations",
                                            style={"fontSize": "0.8rem"}
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        
                        
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="zoom-indicator", className="text-muted small mb-1", 
                                       children="Zoom: 100%", style={"fontWeight": "600", "fontSize": "0.8rem"})
                            ])
                        ]),
                        
                        html.Div(
                            id="image-container",
                            style={
                                "overflow": "auto",
                                "maxHeight": "650px",
                                "border": "2px solid #e0e0e0",
                                "borderRadius": "8px",
                                "position": "relative",
                                "boxShadow": "0 4px 12px rgba(0,0,0,0.1)"
                            },
                            children=[
                                DashCanvas(
                                    id='canvas',
                                    lineWidth=3,
                                    lineColor='red',
                                    tool='polygon',
                                    hide_buttons=['line', 'select', 'pan', 'zoom', 'reset'],
                                )
                            ]
                        ),
                        
                        dbc.Row(id="slice-navigator-row", style={"display": "none"}, children=[
                            dbc.Col([
                                html.Div([
                                    html.Label("Slice Navigator", className="form-label mt-2", style={"fontSize": "0.85rem"}),
                                    html.Div([
                                        html.Button("◀", id="slice-prev-btn", className="me-2 btn btn-sm btn-outline-primary"),
                                        html.Span(id="slice-indicator", children="1 / 1", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                                        html.Button("▶", id="slice-next-btn", className="ms-2 btn btn-sm btn-outline-primary")
                                    ], className="d-flex justify-content-center align-items-center"),
                                    html.Div([
                                        dcc.Slider(
                                            id="slice-slider",
                                            min=1,
                                            max=1,
                                            step=1,
                                            value=1,
                                            marks={},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ], className="mt-2")
                                ])
                            ])
                        ]),
                        
                        html.Div(id="upload-info", className="mt-2 text-muted", style={"fontSize": "0.8rem"}),
                        html.Div(id="save-status", className="mt-2"),
                        html.Div(id="share-status", className="mt-2"),
                        html.Div(id="3d-view-status", className="mt-2")
                    ], style={"padding": "1rem"})
                ], className="shadow-lg", style={"borderRadius": "12px", "border": "none"}),
            ]),
            
            dbc.Col(id="right-column", md=3, children=[
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-bar-chart-fill me-1"),
                        "Stats"
                    ], className="bg-gradient-info text-white py-2", 
                    style={"background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "fontWeight": "600", "fontSize": "0.85rem"}),
                    dbc.CardBody([
                        html.Div(id="mask-stats", style={"minHeight": "180px", "fontSize": "0.75rem"}),
                        dcc.Loading([
                            dbc.Button([
                                html.I(className="bi bi-file-earmark-zip me-1"),
                                "Export ZIP"
                            ], id="export-btn", color="success", size="sm", className="w-100 mt-2",
                            style={"fontWeight": "600", "padding": "8px", "fontSize": "0.8rem"}),
                            dbc.Button([
                                html.I(className="bi bi-file-earmark-text me-1"),
                                "Generate Report"
                            ], id="report-btn", color="primary", size="sm", className="w-100 mt-2",
                            style={"fontWeight": "600", "padding": "8px", "fontSize": "0.8rem"}),
                            dcc.Download(id="download-annotation"),
                            dcc.Download(id="download-report")
                        ])
                    ], style={"padding": "0.75rem"})
                ], className="shadow-lg mb-2", style={"borderRadius": "10px", "border": "none"}),
                
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="bi bi-search me-1"),
                        "Analyze"
                    ], className="bg-gradient-info text-white py-2", 
                    style={"background": "linear-gradient(135deg, #17a2b8 0%, #138496 100%)", "fontWeight": "600", "fontSize": "0.85rem"}),
                    dbc.CardBody([
                        html.P("Analyze selected region or full image with AI.", 
                             className="small text-muted mb-2", style={"fontSize": "0.75rem"}),
                        dcc.Loading([
                            dbc.Button([html.I(className="bi bi-search me-1"), "Analyze"], 
                                     id="analyze-selection", color="info", size="sm", className="w-100",
                                     style={"fontWeight": "600", "padding": "8px", "fontSize": "0.8rem"})
                        ])
                    ], style={"padding": "0.75rem"})
                ], className="shadow-lg mb-2", style={"borderRadius": "10px", "border": "none"}),
            ]),
        ]),
        
        dbc.Row([
            dbc.Col(md=12, children=[
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.Span([
                                html.I(className="bi bi-journal-medical me-2"),
                                "Related Radiopaedia Cases"
                            ], style={"fontSize": "0.95rem", "fontWeight": "600"}),
                            dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"), "Refresh"], 
                                     id="refresh-cases", color="light", size="sm", className="float-end")
                        ])
                    ], className="bg-gradient-primary text-white py-2", 
                    style={"background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"}),
                    dbc.CardBody([
                        html.Div(id="radiopaedia-cases")
                    ], style={"padding": "1rem"})
                ], className="mt-3 shadow-lg", style={"borderRadius": "12px", "border": "none"})
            ])
        ]),
        
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Add Text Annotation", style={"fontSize": "1rem"})),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Label("Text:", style={"fontSize": "0.85rem"}),
                        dbc.Textarea(id="text-input-modal", placeholder="Enter your text here...", rows=3, style={"fontSize": "0.85rem"})
                    ])
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(md=6, children=[
                        dbc.Label("Font Size:", style={"fontSize": "0.85rem"}),
                        dcc.Slider(
                            id="font-size-slider",
                            min=10,
                            max=48,
                            step=2,
                            value=16,
                            marks={i: str(i) for i in range(10, 49, 8)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(id="font-size-display", className="text-center mt-1", children="16px", style={"fontSize": "0.8rem"})
                    ]),
                    dbc.Col(md=6, children=[
                        dbc.Label("Font Style:", style={"fontSize": "0.85rem"}),
                        dcc.Dropdown(
                            id="font-style-dropdown",
                            options=[
                                {"label": "Arial", "value": "Arial"},
                                {"label": "Times New Roman", "value": "Times New Roman"},
                                {"label": "Courier New", "value": "Courier New"},
                                {"label": "Verdana", "value": "Verdana"},
                                {"label": "Georgia", "value": "Georgia"}
                            ],
                            value="Arial",
                            style={"fontSize": "0.85rem"}
                        )
                    ])
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(md=12, children=[
                        dbc.Label("Text Color:", style={"fontSize": "0.85rem"}),
                        html.Div([
                            html.Div([
                                html.Button("Red", id="color-red", className="btn btn-sm me-1 mb-1", 
                                          style={"backgroundColor": "#FF0000", "color": "white", "border": "none", "width": "50px", "fontSize": "0.7rem"}),
                                html.Button("Green", id="color-green", className="btn btn-sm me-1 mb-1", 
                                          style={"backgroundColor": "#00FF00", "color": "black", "border": "none", "width": "50px", "fontSize": "0.7rem"}),
                                html.Button("Blue", id="color-blue", className="btn btn-sm me-1 mb-1", 
                                          style={"backgroundColor": "#0000FF", "color": "white", "border": "none", "width": "50px", "fontSize": "0.7rem"}),
                                html.Button("Yellow", id="color-yellow", className="btn btn-sm me-1 mb-1", 
                                          style={"backgroundColor": "#FFFF00", "color": "black", "border": "none", "width": "50px", "fontSize": "0.7rem"}),
                                html.Button("Black", id="color-black", className="btn btn-sm me-1 mb-1", 
                                          style={"backgroundColor": "#000000", "color": "white", "border": "none", "width": "50px", "fontSize": "0.7rem"}),
                                html.Button("White", id="color-white", className="btn btn-sm me-1 mb-1", 
                                          style={"backgroundColor": "#FFFFFF", "color": "black", "border": "1px solid #ccc", "width": "50px", "fontSize": "0.7rem"}),
                            ], className="mb-2"),
                            html.Div([
                                html.Button("", id="color-1", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#FF6B6B", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-2", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#4ECDC4", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-3", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#45B7D1", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-4", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#96CEB4", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-5", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#FFEAA7", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-6", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#DDA0DD", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-7", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#98D8C8", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-8", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#F7DC6F", "border": "none", "width": "25px", "height": "25px"}),
                            ], className="mb-2"),
                            html.Div([
                                html.Button("", id="color-9", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#BB8FCE", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-10", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#85C1E2", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-11", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#F8B739", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-12", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#52C234", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-13", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#C0392B", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-14", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#5D6D7E", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-15", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#17202A", "border": "none", "width": "25px", "height": "25px"}),
                                html.Button("", id="color-16", className="btn btn-sm me-1", 
                                          style={"backgroundColor": "#FFFFFF", "border": "1px solid #ccc", "width": "25px", "height": "25px"}),
                            ]),
                            html.Div([
                                html.Label("Custom Color:", className="me-2", style={"fontSize": "0.8rem"}),
                                dbc.Input(id="custom-color-input", type="color", value="#FF0000", style={"width": "50px"}),
                                html.Div(id="color-preview", style={
                                    "width": "25px", "height": "25px", "backgroundColor": "#FF0000",
                                    "border": "1px solid #ccc", "display": "inline-block", "marginLeft": "10px"
                                })
                            ], className="mt-2")
                        ])
                    ])
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col(md=6, children=[
                        dbc.Label("Position X:", style={"fontSize": "0.85rem"}),
                        dbc.Input(id="text-x-position", type="number", value=50, min=0, size="sm")
                    ]),
                    dbc.Col(md=6, children=[
                        dbc.Label("Position Y:", style={"fontSize": "0.85rem"}),
                        dbc.Input(id="text-y-position", type="number", value=50, min=0, size="sm")
                    ])
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-text-btn", color="secondary", size="sm", className="me-2"),
                dbc.Button("Add Text", id="add-text-modal-btn", color="primary", size="sm")
            ])
        ], id="text-modal", is_open=False, centered=True, size="lg"),
        
        dcc.Store(id="theme-store", data="light"),
        dcc.Store(id="eraser-strokes-store", data=[]),
        html.Div(id="canvas-data-store", style={"display": "none"}),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Share Session", style={"fontSize": "1rem"})),
            dbc.ModalBody([
                html.P("Share this session with your professor or colleagues:", className="mb-3"),
                html.Div([
                    html.Label("Session ID:", className="fw-bold mb-2"),
                    dbc.InputGroup([
                        dbc.Input(id="session-id-display", type="text", readonly=True, value=""),
                        dbc.Button([html.I(className="bi bi-clipboard me-1"), "Copy"], id="copy-session-id", color="primary", size="sm")
                    ], className="mb-3")
                ]),
                html.Div([
                    html.Label("Share URL:", className="fw-bold mb-2"),
                    dbc.InputGroup([
                        dbc.Input(id="session-url-display", type="text", readonly=True, value=""),
                        dbc.Button([html.I(className="bi bi-clipboard me-1"), "Copy"], id="copy-session-url", color="primary", size="sm")
                    ])
                ]),
                html.Div(id="copy-status", className="mt-2")
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-share-modal", color="secondary", size="sm")
            ])
        ], id="share-modal", is_open=False, centered=True, size="lg")
    ], fluid=True),
    
    html.Div(
        id="chat-float-button",
        children=[
            html.I(className="bi bi-chat-dots", style={"fontSize": "24px", "color": "white"})
        ],
        style={
            "position": "fixed",
            "bottom": "30px",
            "right": "30px",
            "width": "60px",
            "height": "60px",
            "borderRadius": "50%",
            "backgroundColor": "#14B8A6",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "cursor": "pointer",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.3)",
            "zIndex": "1000",
            "transition": "all 0.3s ease"
        }
    ),
    
    html.Div(
        id="chat-overlay",
        style={"display": "none"},
        children=[
            dbc.Card([
                dbc.CardHeader([
                    html.Span("AI Assistant", style={"fontSize": "16px", "fontWeight": "bold"}),
                    dbc.Button(
                        "CLOSE",
                        id="close-chat-btn",
                        color="danger",
                        size="sm",
                        style={"float": "right"}
                    )
                ], className="bg-success text-white py-2"),
                dbc.CardBody([
                    dcc.Loading([
                        html.Div(id="chat-box", style={
                            'height': '350px', 'overflowY': 'auto', 'padding': '10px',
                            'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '12px'
                        })
                    ]),
                    dbc.Row([
                        dbc.Col(md=12, children=[
                            dbc.Input(id="chat-input", placeholder="Ask about the image or annotation...", type="text", size="sm")
                        ]),
                        dbc.Col(md=12, children=[
                            dbc.Button("SEND", id="send-chat", color="success", size="sm", className="w-100 mt-2")
                        ])
                    ])
                ], style={"padding": "15px"})
            ], style={
                "position": "fixed",
                "bottom": "30px",
                "right": "30px",
                "width": "380px",
                "maxHeight": "550px",
                "zIndex": "1001",
                "boxShadow": "0 8px 24px rgba(0,0,0,0.3)",
                "borderRadius": "8px"
            })
        ]
    )
])

state = {
    "file_bytes": None,
    "ds": None,
    "image_orig": None,
    "image_display": None,
    "disp_size": (500, 500),
    "mask_history": [],
    "mask_future": [],
    "current_mask": None,
    "comments": [],
    "chat_history": [],
    "radiopaedia_cases": [],
    "zoom_level": 1.0,
    "text_annotations": [],
    "rotation": 0,
    "flip_horizontal": False,
    "flip_vertical": False,
    "current_slice": 0,
    "total_slices": 1,
    "image_slices": [],
    "fullscreen_mode": False,
    "canvas_json_data": None,
    "image_b64": None,
    "text_color": "#FF0000",
    "eraser_mode": False,
    "eraser_strokes": [],
    "saved_states": [],
    "file_type": None,
    "filename": None,
    "session_id": None
}


@app.callback(
    Output("share-modal", "is_open"),
    Output("session-id-display", "value"),
    Output("session-url-display", "value"),
    Output("share-status", "children"),
    Input("share-btn", "n_clicks"),
    Input("close-share-modal", "n_clicks"),
    State("share-modal", "is_open"),
    prevent_initial_call=True
)
def share_session(share_clicks, close_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "close-share-modal":
        return False, dash.no_update, dash.no_update, dash.no_update
    
    if button_id == "share-btn":
        if not state["image_orig"]:
            return False, "", "", dbc.Alert("No image loaded. Please upload an image first.", color="warning", dismissable=True, duration=3000)
        
        try:
            session_data = {
                "file_bytes": state["file_bytes"],
                "image_orig": state["image_orig"],
                "canvas_json_data": state["canvas_json_data"],
                "text_annotations": state["text_annotations"],
                "current_mask": state["current_mask"],
                "eraser_strokes": state["eraser_strokes"],
                "zoom_level": state["zoom_level"],
                "rotation": state["rotation"],
                "flip_horizontal": state["flip_horizontal"],
                "flip_vertical": state["flip_vertical"],
                "file_type": state["file_type"],
                "filename": state["filename"],
                "disp_size": state["disp_size"]
            }
            
            session_id = save_session(session_data)
            state["session_id"] = session_id
            
            share_url = f"http://localhost:8050/?session={session_id}"
            
            return True, session_id, share_url, ""
            
        except Exception as e:
            return False, "", "", dbc.Alert(f"Error sharing session: {str(e)}", color="danger", dismissable=True, duration=5000)
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("copy-status", "children"),
    Input("copy-session-id", "n_clicks"),
    Input("copy-session-url", "n_clicks"),
    State("session-id-display", "value"),
    State("session-url-display", "value"),
    prevent_initial_call=True
)
def copy_to_clipboard(id_clicks, url_clicks, session_id, session_url):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "copy-session-id":
        return dbc.Alert([html.I(className="bi bi-check-circle me-2"), "Session ID copied! (Use Ctrl+C to copy)"], 
                        color="success", dismissable=True, duration=2000)
    elif button_id == "copy-session-url":
        return dbc.Alert([html.I(className="bi bi-check-circle me-2"), "URL copied! (Use Ctrl+C to copy)"], 
                        color="success", dismissable=True, duration=2000)
    
    return dash.no_update

@app.callback(
    Output("3d-view-status", "children"),
    Input("3d-view-btn", "n_clicks"),
    prevent_initial_call=True
)
def show_3d_view(n_clicks):
    if not n_clicks:
        return dash.no_update
    
    if not state["image_orig"]:
        return dbc.Alert("No image loaded. Please upload an image first.", color="warning", dismissable=True, duration=3000)
    
    return dbc.Alert([
        html.I(className="bi bi-info-circle me-2"),
        "3D View feature coming soon! This will allow you to view medical images in 3D perspective."
    ], color="info", dismissable=True, duration=5000)


@app.callback(
    Output("canvas-data-store", "children"),
    Output("eraser-strokes-store", "data"),
    Input('canvas', 'json_data'),
    State('canvas', 'height'),
    State('canvas', 'width'),
    State("eraser-strokes-store", "data"),
    prevent_initial_call=True
)
def store_canvas_data(json_data, height, width, eraser_strokes):
    if json_data:
        state["canvas_json_data"] = json_data
        
        # If in eraser mode, store this stroke as an eraser stroke
        if state["eraser_mode"]:
            eraser_strokes = eraser_strokes or []
            eraser_strokes.append(json_data)
            state["eraser_strokes"] = eraser_strokes
        
        mask_array = parse_canvas_data(json_data, height, width, state["eraser_strokes"])
        if mask_array is not None:
            state["current_mask"] = mask_array
        
        return json_data, eraser_strokes
    return "", eraser_strokes

@app.callback(
    Output("mask-stats", "children"),
    Input("canvas-data-store", "children"),
    State('canvas', 'height'),
    State('canvas', 'width'),
    State("eraser-strokes-store", "data"),
    prevent_initial_call=True
)
def update_mask_stats(json_data, height, width, eraser_strokes):
    if not json_data:
        return "No annotation created. Use drawing tools to annotate the image."
    
    try:
        mask_array = parse_canvas_data(json_data, height, width, eraser_strokes)
        
        if mask_array is None or not mask_array.any():
            return "No annotation detected. Please draw on the canvas."
        
        stats = compute_mask_stats(mask_array)
        
        if stats["area_pixels"] == 0:
            return "No annotation detected. Please draw on the canvas."
        
        stats_display = [
            html.P([html.I(className="bi bi-tools me-1"), f"Tool: {stats['tool_type']}"], 
                  className="mb-1", style={"fontWeight": "600", "fontSize": "0.75rem"}),
            html.P([html.I(className="bi bi-rulers me-1"), f"Area: {stats['area_pixels']} px"], 
                  className="mb-1", style={"fontSize": "0.75rem"}),
            html.P([html.I(className="bi bi-pie-chart me-1"), f"Coverage: {stats['percent']:.2f}%"], 
                  className="mb-1", style={"fontSize": "0.75rem"}),
        ]
        
        if stats["bbox"]:
            x0, y0, x1, y1 = stats["bbox"]
            width_px = x1 - x0 + 1
            height_px = y1 - y0 + 1
            stats_display.append(html.P([html.I(className="bi bi-bounding-box me-1"), 
                                       f"BBox: ({x0}, {y0}) to ({x1}, {y1})"], className="mb-1", style={"fontSize": "0.7rem"}))
            stats_display.append(html.P([html.I(className="bi bi-arrows-angle-expand me-1"), 
                                       f"Dim: {width_px} × {height_px} px"], className="mb-1", style={"fontSize": "0.7rem"}))
        
        if "Circle" in stats["tool_type"] and stats["bbox"]:
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            radius = min(width_px, height_px) / 2
            stats_display.append(html.P([html.I(className="bi bi-circle me-1"), 
                                       f"Center: ({center_x:.0f}, {center_y:.0f}), R: {radius:.0f}px"], 
                                      className="mb-1", style={"fontSize": "0.7rem"}))
        
        elif "Rectangle" in stats["tool_type"] and stats["bbox"]:
            perimeter = 2 * (width_px + height_px)
            stats_display.append(html.P([html.I(className="bi bi-square me-1"), 
                                       f"Perimeter: {perimeter} px"], className="mb-1", style={"fontSize": "0.7rem"}))
        
        elif "Line" in stats["tool_type"] and stats["bbox"]:
            line_length = np.sqrt(width_px**2 + height_px**2)
            stats_display.append(html.P([html.I(className="bi bi-slash-lg me-1"), 
                                       f"Length: {line_length:.1f} px"], className="mb-1", style={"fontSize": "0.7rem"}))
        
        return stats_display
        
    except Exception as e:
        return f"Error calculating statistics: {str(e)}"



@app.callback(
    Output("chat-overlay", "style"),
    Output("chat-float-button", "style"),
    Input("chat-float-button", "n_clicks"),
    Input("close-chat-btn", "n_clicks"),
    Input("analyze-selection", "n_clicks"),
    State("chat-overlay", "style"),
    prevent_initial_call=True
)
def toggle_chat_overlay(float_clicks, close_clicks, analyze_clicks, current_style):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    float_button_style = {
        "position": "fixed",
        "bottom": "30px",
        "right": "30px",
        "width": "60px",
        "height": "60px",
        "borderRadius": "50%",
        "backgroundColor": "#14B8A6",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "cursor": "pointer",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.3)",
        "zIndex": "1000",
        "transition": "all 0.3s ease"
    }
    
    if button_id == "chat-float-button" or button_id == "analyze-selection":
        overlay_style = {"display": "block"}
        float_button_style["display"] = "none"
        return overlay_style, float_button_style
    elif button_id == "close-chat-btn":
        overlay_style = {"display": "none"}
        return overlay_style, float_button_style
    
    return dash.no_update, dash.no_update

@app.callback(
    Output("save-status", "children"),
    Input("save-btn", "n_clicks"),
    State("canvas-data-store", "children"),
    State('canvas', 'height'),
    State('canvas', 'width'),
    prevent_initial_call=True
)
def save_annotation_state(n_clicks, json_data, height, width):
    if not n_clicks:
        return dash.no_update
    
    if not state["image_orig"]:
        return dbc.Alert("⚠️ No image loaded. Please upload an image file first.", color="warning", dismissable=True, duration=3000)
    
    save_state = {
        "timestamp": datetime.datetime.now().isoformat(),
        "canvas_json": json_data,
        "mask": state["current_mask"].copy() if state["current_mask"] is not None else None,
        "text_annotations": state["text_annotations"].copy(),
        "rotation": state["rotation"],
        "flip_horizontal": state["flip_horizontal"],
        "flip_vertical": state["flip_vertical"],
        "zoom_level": state["zoom_level"]
    }
    
    state["saved_states"].append(save_state)
    
    if state["current_mask"] is not None:
        state["mask_history"].append(state["current_mask"].copy())
        state["mask_future"] = []
    
    return dbc.Alert(
        [html.I(className="bi bi-check-circle me-2"), f"✅ Annotation saved successfully at {datetime.datetime.now().strftime('%H:%M:%S')}"],
        color="success",
        dismissable=True,
        duration=3000
    )

@app.callback(
    Output("zoom-indicator", "children"),
    Output("image-container", "style", allow_duplicate=True),
    Output("canvas", "width", allow_duplicate=True),
    Output("canvas", "height", allow_duplicate=True),
    Output("left-column", "md"),
    Output("right-column", "md"),
    Output("right-column", "style"),
    Input("zoom-in-btn", "n_clicks"),
    Input("zoom-out-btn", "n_clicks"),
    Input("fit-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_zoom(zoom_in_clicks, zoom_out_clicks, fit_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    left_md = 9
    right_md = 3
    right_style = {}
    
    canvas_style = {
        "overflow": "auto",
        "maxHeight": "650px",
        "border": "2px solid #e0e0e0",
        "borderRadius": "8px",
        "position": "relative",
        "boxShadow": "0 4px 12px rgba(0,0,0,0.1)"
    }
    
    canvas_width = int(state["disp_size"][0] * state["zoom_level"])
    canvas_height = int(state["disp_size"][1] * state["zoom_level"])
    
    if button_id == "zoom-in-btn" and state["zoom_level"] < 3.0:
        state["zoom_level"] = min(state["zoom_level"] * 1.2, 3.0)
        canvas_width = int(state["disp_size"][0] * state["zoom_level"])
        canvas_height = int(state["disp_size"][1] * state["zoom_level"])
    elif button_id == "zoom-out-btn" and state["zoom_level"] > 0.5:
        state["zoom_level"] = max(state["zoom_level"] / 1.2, 0.5)
        canvas_width = int(state["disp_size"][0] * state["zoom_level"])
        canvas_height = int(state["disp_size"][1] * state["zoom_level"])
    elif button_id == "fit-btn":
        state["fullscreen_mode"] = not state["fullscreen_mode"]
        if state["fullscreen_mode"]:
            left_md = 12
            right_md = 0
            right_style = {"display": "none"}
            canvas_style["maxHeight"] = "80vh"
            canvas_style["margin"] = "0 auto"
            state["zoom_level"] = 1.0
            canvas_width = 800
            canvas_height = 800
        else:
            state["zoom_level"] = 1.0
            canvas_width = state["disp_size"][0]
            canvas_height = state["disp_size"][1]
    
    return f"Zoom: {int(state['zoom_level']*100)}%", canvas_style, canvas_width, canvas_height, left_md, right_md, right_style

@app.callback(
    Output("app-container", "style"),
    Output("theme-store", "data"),
    Input("theme-switch", "value"),
    prevent_initial_call=True
)
def toggle_theme(is_dark):
    if is_dark:
        return {"backgroundColor": "#222222", "color": "#ffffff", "minHeight": "100vh"}, "dark"
    else:
        return {"backgroundColor": "#ffffff", "color": "#000000", "minHeight": "100vh"}, "light"

@app.callback(
    Output('canvas', 'image_content'),
    Output('canvas', 'width'),
    Output('canvas', 'height'),
    Output('upload-info', 'children'),
    Output('slice-navigator-row', 'style'),
    Input('upload-dicom', 'contents'),
    State('upload-dicom', 'filename'),
    prevent_initial_call=True
)
def load_image(contents, filename):
    if not contents:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, {"display": "none"}
    try:
        _, data = contents.split(',', 1)
        file_bytes = base64.b64decode(data)
        
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            img_orig, ds = image_to_pil(file_bytes)
            state["file_type"] = file_ext
            modality = file_ext.upper()
            patient_id = "N/A"
        elif file_ext == 'dcm':
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            if hasattr(ds, 'NumberOfSlices') and ds.NumberOfSlices > 1:
                state["total_slices"] = ds.NumberOfSlices
                state["current_slice"] = 0
                img_orig, _ = dicom_to_image(file_bytes)
                state["image_slices"] = [img_orig]
            else:
                img_orig, ds = dicom_to_image(file_bytes)
                state["total_slices"] = 1
                state["current_slice"] = 0
                state["image_slices"] = [img_orig]
            state["file_type"] = "dcm"
            modality = getattr(ds, 'Modality', 'DICOM')
            patient_id = getattr(ds, 'PatientID', 'Unknown')
        else:
            return dash.no_update, dash.no_update, dash.no_update, f"❌ Unsupported file type: {file_ext}", {"display": "none"}
        
        img_disp, disp_size = resize_for_display(img_orig)
        state.update({
            "file_bytes": file_bytes,
            "ds": ds,
            "image_orig": img_orig,
            "image_display": img_disp,
            "disp_size": disp_size,
            "mask_history": [],
            "mask_future": [],
            "current_mask": None,
            "comments": [],
            "chat_history": [],
            "radiopaedia_cases": [],
            "zoom_level": 1.0,
            "text_annotations": [],
            "rotation": 0,
            "flip_horizontal": False,
            "flip_vertical": False,
            "image_b64": pil_to_b64(img_disp),
            "filename": filename,
            "eraser_strokes": []
        })
        info = f"✅ Loaded: {filename} | Type: {modality} | Patient: {patient_id} | Size: {img_orig.size[0]}x{img_orig.size[1]}"
        navigator_style = {"display": "block"} if state["total_slices"] > 1 else {"display": "none"}
        return state["image_b64"], disp_size[0], disp_size[1], info, navigator_style
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, f"❌ Error: {e}", {"display": "none"}

@app.callback(
    Output("canvas", "image_content", allow_duplicate=True),
    Input("add-text-modal-btn", "n_clicks"),
    State("text-input-modal", "value"),
    State("font-size-slider", "value"),
    State("font-style-dropdown", "value"),
    State("text-x-position", "value"),
    State("text-y-position", "value"),
    prevent_initial_call=True
)
def add_text_annotation(n_clicks, text_value, font_size, font_style, x_pos, y_pos):
    if n_clicks and text_value and state["image_orig"]:
        img = state["image_orig"].copy()
        hex_color = state["text_color"].lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        disp_w, disp_h = state["disp_size"]
        orig_w, orig_h = state["image_orig"].size
        scale_x = orig_w / disp_w
        scale_y = orig_h / disp_h
        
        position = (
            int((int(x_pos) if x_pos else 50) * scale_x), 
            int((int(y_pos) if y_pos else 50) * scale_y)
        )
        
        img = add_text_to_image(img, text_value, position, int(font_size), rgb_color, font_style)
        state["image_orig"] = img
        img_disp, disp_size = resize_for_display(img)
        state["image_display"] = img_disp
        state["disp_size"] = disp_size
        state["image_b64"] = pil_to_b64(img_disp)
        state["text_annotations"].append({
            "text": text_value,
            "position": position,
            "font_size": font_size,
            "font_style": font_style,
            "color": state["text_color"],
            "timestamp": datetime.datetime.now().isoformat()
        })
        return state["image_b64"]
    return dash.no_update

@app.callback(
    Output("tool-polygon", "outline"),
    Output("tool-rectangle", "outline"),
    Output("tool-circle", "outline"),
    Output("tool-line", "outline"),
    Output("tool-pen", "outline"),
    Output("tool-eraser", "outline"),
    Output("tool-text", "outline"),
    Input("tool-polygon", "n_clicks"),
    Input("tool-rectangle", "n_clicks"),
    Input("tool-circle", "n_clicks"),
    Input("tool-line", "n_clicks"),
    Input("tool-pen", "n_clicks"),
    Input("tool-eraser", "n_clicks"),
    Input("tool-text", "n_clicks"),
)
def update_tool_buttons(poly_clicks, rect_clicks, circle_clicks, line_clicks, pen_clicks, eraser_clicks, text_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return True, True, True, True, True, True, True
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    outlines = [True] * 7
    tool_map = {
        "tool-polygon": 0,
        "tool-rectangle": 1,
        "tool-circle": 2,
        "tool-line": 3,
        "tool-pen": 4,
        "tool-eraser": 5,
        "tool-text": 6
    }
    if button_id in tool_map:
        outlines[tool_map[button_id]] = False
    return tuple(outlines)

@app.callback(
    Output("canvas", "tool"),
    Output("text-modal", "is_open"),
    Output("canvas", "lineColor"),
    Output("canvas", "lineWidth"),
    Input("tool-polygon", "n_clicks"),
    Input("tool-rectangle", "n_clicks"),
    Input("tool-circle", "n_clicks"),
    Input("tool-line", "n_clicks"),
    Input("tool-pen", "n_clicks"),
    Input("tool-eraser", "n_clicks"),
    Input("tool-text", "n_clicks"),
    Input("cancel-text-btn", "n_clicks"),
    Input("add-text-modal-btn", "n_clicks"),
    State("text-modal", "is_open"),
    prevent_initial_call=True
)
def update_canvas_tool(poly_clicks, rect_clicks, circle_clicks, line_clicks, pen_clicks, eraser_clicks, text_clicks, 
                      cancel_clicks, add_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return "polygon", False, "red", 3
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "tool-text":
        return "polygon", True, "red", 3
    elif button_id in ["cancel-text-btn", "add-text-modal-btn"]:
        return "polygon", False, "red", 3
    
    tool_map = {
        "tool-polygon": "polygon",
        "tool-rectangle": "rectangle",
        "tool-circle": "circle",
        "tool-line": "line",
        "tool-pen": "pen",
        "tool-eraser": "pen"
    }
    
    if button_id == "tool-eraser":
        state["eraser_mode"] = True
        line_color = "white"
        line_width = 15
    else:
        state["eraser_mode"] = False
        line_color = "red"
        line_width = 3
    
    tool = tool_map.get(button_id, "polygon")
    
    return tool, is_open, line_color, line_width


@app.callback(
    Output("canvas", "image_content", allow_duplicate=True),
    Input("rotate-left-btn", "n_clicks"),
    Input("rotate-right-btn", "n_clicks"),
    Input("flip-h-btn", "n_clicks"),
    Input("flip-v-btn", "n_clicks"),
    prevent_initial_call=True
)
def transform_image(rotate_left, rotate_right, flip_h, flip_v):
    ctx = callback_context
    if not ctx.triggered or not state["image_orig"]:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    img = state["image_orig"].copy()
    
    if button_id == "rotate-left-btn":
        img = img.rotate(90, expand=True)
        state["rotation"] = (state["rotation"] - 90) % 360
    elif button_id == "rotate-right-btn":
        img = img.rotate(-90, expand=True)
        state["rotation"] = (state["rotation"] + 90) % 360
    elif button_id == "flip-h-btn":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        state["flip_horizontal"] = not state["flip_horizontal"]
    elif button_id == "flip-v-btn":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        state["flip_vertical"] = not state["flip_vertical"]
    
    state["image_orig"] = img
    img_disp, disp_size = resize_for_display(img)
    state["image_display"] = img_disp
    state["disp_size"] = disp_size
    state["image_b64"] = pil_to_b64(img_disp)
    
    return state["image_b64"]

@app.callback(
    Output("canvas", "json_data", allow_duplicate=True),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_annotations(n_clicks):
    if n_clicks:
        state["current_mask"] = None
        state["canvas_json_data"] = None
        state["eraser_strokes"] = []
        return ""
    return dash.no_update

@app.callback(
    Output("download-annotation", "data"),
    Input("export-btn", "n_clicks"),
    State("canvas-data-store", "children"),
    State('canvas', 'height'),
    State('canvas', 'width'),
    State("eraser-strokes-store", "data"),
    prevent_initial_call=True
)
def export_annotation(n_clicks, json_data, height, width, eraser_strokes):
    if not n_clicks or not state["image_orig"]:
        return dash.no_update
    
    try:
        mask_array = None
        if json_data:
            mask_array = parse_canvas_data(json_data, height, width, eraser_strokes)
        
        img_bytes = io.BytesIO()
        state["image_orig"].save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        mask_bytes = b""
        if mask_array is not None and mask_array.any():
            mask_img = Image.fromarray(mask_array)
            mask_buf = io.BytesIO()
            mask_img.save(mask_buf, format='PNG')
            mask_bytes = mask_buf.getvalue()
        
        stats = compute_mask_stats(mask_array) if mask_array is not None else {}
        metadata = {
            "filename": state.get("filename", "unknown"),
            "file_type": state.get("file_type", "unknown"),
            "image_size": state["image_orig"].size,
            "annotation_stats": stats,
            "text_annotations": state["text_annotations"],
            "timestamp": datetime.datetime.now().isoformat(),
            "zoom_level": state["zoom_level"],
            "rotation": state["rotation"],
            "flip_horizontal": state["flip_horizontal"],
            "flip_vertical": state["flip_vertical"]
        }
        
        zip_buf = make_export_zip(img_bytes, mask_bytes, metadata, state.get("filename", "image"))
        
        return dcc.send_bytes(zip_buf.getvalue(), f"annotation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    
    except Exception as e:
        print(f"Export error: {e}")
        return dash.no_update

@app.callback(
    Output("download-report", "data"),
    Input("report-btn", "n_clicks"),
    State("canvas-data-store", "children"),
    State('canvas', 'height'),
    State('canvas', 'width'),
    State("eraser-strokes-store", "data"),
    prevent_initial_call=True
)
def generate_report(n_clicks, json_data, height, width, eraser_strokes):
    if not n_clicks or not state["image_orig"]:
        return dash.no_update
    
    try:
        mask_array = None
        if json_data:
            mask_array = parse_canvas_data(json_data, height, width, eraser_strokes)
        
        stats = compute_mask_stats(mask_array) if mask_array is not None else {}
        
        report_data = {
            "filename": state.get("filename", "unknown"),
            "file_type": state.get("file_type", "unknown"),
            "image_size": f"{state['image_orig'].size[0]} x {state['image_orig'].size[1]}",
            "disp_size": f"{state['disp_size'][0]} x {state['disp_size'][1]}",
            "tool_type": stats.get("tool_type", "N/A"),
            "area_pixels": stats.get("area_pixels", 0),
            "percent": stats.get("percent", 0),
            "bbox": stats.get("bbox", "N/A"),
            "text_annotations": state["text_annotations"],
            "saved_states_count": len(state["saved_states"]),
            "zoom_level": state["zoom_level"],
            "rotation": state["rotation"]
        }
        
        report_html = generate_annotation_report(report_data)
        
        return dcc.send_string(report_html, f"annotation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    except Exception as e:
        print(f"Report generation error: {e}")
        return dash.no_update

@app.callback(
    Output("text-color", "data", allow_duplicate=True),
    Output("color-preview", "style"),
    Input("color-red", "n_clicks"),
    Input("color-green", "n_clicks"),
    Input("color-blue", "n_clicks"),
    Input("color-yellow", "n_clicks"),
    Input("color-black", "n_clicks"),
    Input("color-white", "n_clicks"),
    Input("color-1", "n_clicks"),
    Input("color-2", "n_clicks"),
    Input("color-3", "n_clicks"),
    Input("color-4", "n_clicks"),
    Input("color-5", "n_clicks"),
    Input("color-6", "n_clicks"),
    Input("color-7", "n_clicks"),
    Input("color-8", "n_clicks"),
    Input("color-9", "n_clicks"),
    Input("color-10", "n_clicks"),
    Input("color-11", "n_clicks"),
    Input("color-12", "n_clicks"),
    Input("color-13", "n_clicks"),
    Input("color-14", "n_clicks"),
    Input("color-15", "n_clicks"),
    Input("color-16", "n_clicks"),
    Input("custom-color-input", "value"),
    prevent_initial_call=True
)
def update_text_color(*args):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    color_map = {
        "color-red": "#FF0000",
        "color-green": "#00FF00",
        "color-blue": "#0000FF",
        "color-yellow": "#FFFF00",
        "color-black": "#000000",
        "color-white": "#FFFFFF",
        "color-1": "#FF6B6B",
        "color-2": "#4ECDC4",
        "color-3": "#45B7D1",
        "color-4": "#96CEB4",
        "color-5": "#FFEAA7",
        "color-6": "#DDA0DD",
        "color-7": "#98D8C8",
        "color-8": "#F7DC6F",
        "color-9": "#BB8FCE",
        "color-10": "#85C1E2",
        "color-11": "#F8B739",
        "color-12": "#52C234",
        "color-13": "#C0392B",
        "color-14": "#5D6D7E",
        "color-15": "#17202A",
        "color-16": "#FFFFFF"
    }
    
    if button_id == "custom-color-input":
        color = ctx.triggered[0]['value']
    else:
        color = color_map.get(button_id, "#FF0000")
    
    state["text_color"] = color
    preview_style = {
        "width": "25px",
        "height": "25px",
        "backgroundColor": color,
        "border": "1px solid #ccc",
        "display": "inline-block",
        "marginLeft": "10px"
    }
    
    return color, preview_style

@app.callback(
    Output("font-size-display", "children"),
    Input("font-size-slider", "value"),
    prevent_initial_call=True
)
def update_font_size_display(value):
    return f"{value}px"

@app.callback(
    Output("chat-box", "children"),
    Output("chat-input", "value"),
    Output("radiopaedia-cases", "children"),
    Input("send-chat", "n_clicks"),
    Input("analyze-selection", "n_clicks"),
    State("chat-input", "value"),
    State("canvas-data-store", "children"),
    State("chat-box", "children"),
    State('canvas', 'height'),
    State('canvas', 'width'),
    State("eraser-strokes-store", "data"),
    prevent_initial_call=True
)
def handle_chat_and_analysis(send_click, analyze_click, text_value, json_mask, chat_children, height, width, eraser_strokes):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    children = chat_children or []
    prompt = text_value.strip() if text_value else "Analyze this medical image or its cropped region and explain for a student."
    cases_html = dash.no_update
    
    if trigger == "send-chat":
        if not prompt:
            return children, dash.no_update, cases_html
        
        image_data = state["image_b64"] if state["image_b64"] else None
        reply = chat_with_openai(prompt, state["chat_history"], image_data)
        
        state["chat_history"].append({"role": "user", "content": prompt})
        state["chat_history"].append({"role": "assistant", "content": reply})
        
        children.append(html.Div([
            html.Strong("You:"),
            html.P(prompt, style={"fontSize": "0.85rem"})
        ], className="d-flex justify-content-end mb-2"))
        
        children.append(html.Div([
            html.Strong("Assistant:"),
            html.Div(format_chat_text(reply))
        ], className="p-2 mb-2 bg-primary text-white rounded"))
        
        cases = get_radiopaedia_cases(state["chat_history"])
        state["radiopaedia_cases"] = cases
        cases_html = format_radiopaedia_cases(cases)
        
        return children, "", cases_html
    
    elif trigger == "analyze-selection":
        if not state["image_orig"]:
            children.append(html.Div([
                html.Strong("Assistant:"),
                html.P("⚠️ Upload an image file first.", style={"fontSize": "0.85rem"})
            ], className="p-2 mb-2 bg-primary text-white rounded"))
            return children, "", cases_html
        
        mask_array = None
        if json_mask:
            mask_array = parse_canvas_data(json_mask, height, width, eraser_strokes)
            state["current_mask"] = mask_array
        
        if mask_array is None or not mask_array.any():
            # Analyze full image
            b64 = state["image_b64"]
            children.append(html.Div([
                html.Strong("You:"),
                html.P(f"{prompt} (Full image)", style={"fontSize": "0.85rem"})
            ], className="d-flex justify-content-end mb-2"))
            
            children.append(html.Div([
                html.Img(src=b64, style={"maxWidth": "100%", "borderRadius": "6px", "marginBottom": "8px"})
            ]))
            
            reply = chat_with_openai(prompt, state["chat_history"], b64)
        else:
            # Analyze annotated region
            stats = compute_mask_stats(mask_array)
            bbox_disp = stats.get("bbox")
            
            if not bbox_disp:
                children.append(html.Div([
                    html.Strong("Assistant:"),
                    html.P("⚠️ Draw a valid annotation first.", style={"fontSize": "0.85rem"})
                ], className="p-2 mb-2 bg-primary text-white rounded"))
                return children, "", cases_html
            
            bbox_orig = scale_bbox_display_to_original(bbox_disp, state["disp_size"], state["image_orig"].size)
            cropped = crop_original_by_bbox(state["image_orig"], bbox_orig)
            
            if not cropped:
                children.append(html.Div([
                    html.Strong("Assistant:"),
                    html.P("⚠️ Failed to crop region.", style={"fontSize": "0.85rem"})
                ], className="p-2 mb-2 bg-primary text-white rounded"))
                return children, "", cases_html
            
            b64 = pil_to_b64(cropped)
            children.append(html.Div([
                html.Strong("You:"),
                html.P(f"{prompt} (Annotated region)", style={"fontSize": "0.85rem"})
            ], className="d-flex justify-content-end mb-2"))
            
            children.append(html.Div([
                html.Img(src=b64, style={"maxWidth": "100%", "borderRadius": "6px", "marginBottom": "8px"})
            ]))
            
            reply = chat_with_openai(prompt, state["chat_history"], b64)
        
        state["chat_history"].append({"role": "user", "content": prompt})
        state["chat_history"].append({"role": "assistant", "content": reply})
        
        children.append(html.Div([
            html.Strong("Assistant:"),
            html.Div(format_chat_text(reply))
        ], className="p-2 mb-2 bg-primary text-white rounded"))
        
        cases = get_radiopaedia_cases(state["chat_history"])
        state["radiopaedia_cases"] = cases
        cases_html = format_radiopaedia_cases(cases)
        
        return children, "", cases_html
    
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("radiopaedia-cases", "children", allow_duplicate=True),
    Input("refresh-cases", "n_clicks"),
    prevent_initial_call=True
)
def refresh_radiopaedia_cases(n_clicks):
    if not n_clicks or not state["chat_history"]:
        return dash.no_update
    
    cases = get_radiopaedia_cases(state["chat_history"])
    state["radiopaedia_cases"] = cases
    return format_radiopaedia_cases(cases)

if __name__ == '__main__':
    app.run(debug=True, host='192.168.1.238', port=8050)
