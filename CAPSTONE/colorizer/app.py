import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage import segmentation
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from inference import colorize_video, load_model
from utils import CHECKPOINT_BEST, CHECKPOINT_LATEST, get_checkpoint_path


st.set_page_config(
    page_title="ChromaAI Studio — Colorization & Restoration",
    page_icon=":material/palette:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Dark Navy + Purple & Electric Blue Glassmorphism Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 12% 10%, rgba(139, 92, 246, 0.16), transparent 36%),
                    radial-gradient(circle at 88% 15%, rgba(56, 189, 248, 0.14), transparent 38%),
                    linear-gradient(160deg, #050814 0%, #0A1024 55%, #0D132B 100%);
        color: #F1F5F9;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090E1F 0%, #0E1530 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.22);
    }

    /* Hero Banner */
    .studio-hero {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.18) 0%, rgba(56, 189, 248, 0.12) 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 18px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 36px rgba(5, 8, 20, 0.45);
    }
    .studio-hero-badge {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.3), rgba(56, 189, 248, 0.3));
        border: 1px solid rgba(56, 189, 248, 0.45);
        color: #E0F2FE;
        margin-bottom: 0.65rem;
    }
    .studio-hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2.05rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #C4B5FD 55%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .studio-hero p {
        margin: 0;
        color: #CBD5E1;
        font-size: 0.98rem;
        max-width: 760px;
        line-height: 1.5;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(17, 24, 45, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1.25rem 1.35rem;
        backdrop-filter: blur(14px);
        transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
        margin-bottom: 0.85rem;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(139, 92, 246, 0.55);
        box-shadow: 0 10px 28px rgba(139, 92, 246, 0.16);
    }
    .glass-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.35rem;
    }
    .glass-card-desc {
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.45;
        margin-bottom: 0;
    }

    /* Empty State */
    .empty-state {
        background: rgba(15, 23, 42, 0.55);
        border: 1.5px dashed rgba(139, 92, 246, 0.42);
        border-radius: 16px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        color: #CBD5E1;
        margin: 0.75rem 0 1.25rem 0;
    }
    .empty-state h4 {
        margin: 0 0 0.4rem 0;
        color: #F8FAFC;
        font-weight: 700;
    }
    .empty-state p {
        margin: 0;
        color: #94A3B8;
        font-size: 0.9rem;
    }

    /* Animated Primary & Secondary Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(139, 92, 246, 0.42);
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.24), rgba(56, 189, 248, 0.18));
        color: #F8FAFC;
        transition: transform 0.18s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.24);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C3AED 0%, #2563EB 55%, #0EA5E9 100%);
        border: none;
        color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.45);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 45, 0.72);
        border: 1px solid rgba(139, 92, 246, 0.24);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        backdrop-filter: blur(12px);
    }

    /* Brand Header in Sidebar */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.65rem 0.75rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.22), rgba(56, 189, 248, 0.14));
        border: 1px solid rgba(139, 92, 246, 0.35);
        margin-bottom: 0.9rem;
    }
    .sidebar-brand-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #8B5CF6, #38BDF8);
        color: white;
        font-weight: 800;
        font-size: 1.05rem;
        flex-shrink: 0;
    }
    .sidebar-brand-text {
        line-height: 1.2;
    }
    .sidebar-brand-title {
        font-weight: 800;
        font-size: 1rem;
        color: #F8FAFC;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Native v2 Components (Canvas & Interactive Before/After Comparison)
# ---------------------------------------------------------------------------
CANVAS_COMPONENT = st.components.v2.component(
    "color_studio_canvas",
    html="""
<div class="canvas-shell"><img id="canvas" alt="Interactive image"/><div id="status"></div></div>
""",
    css="""
.canvas-shell{position:relative; width:100%; line-height:0}
#canvas{width:100%; height:auto; max-height:720px; object-fit:contain; cursor:crosshair; border-radius:12px}
#status{position:absolute; left:10px; bottom:10px; color:white; background:rgba(7,11,25,.78);
font:12px sans-serif; padding:5px 10px; border-radius:8px; line-height:1.2; border:1px solid rgba(139,92,246,.4)}
""",
    js=r"""
export default function(component) {
  const {data, parentElement, setTriggerValue} = component
  const img = parentElement.querySelector("#canvas")
  const status = parentElement.querySelector("#status")
  img.src = data.image || ""
  img.alt = data.alt || "Interactive image"
  status.textContent = data.status || ""
  img.onclick = (event) => {
    const r = img.getBoundingClientRect()
    const x = Math.round((event.clientX-r.left)/r.width*(data.width || 1))
    const y = Math.round((event.clientY-r.top)/r.height*(data.height || 1))
    setTriggerValue("click", {x, y, token: Date.now()})
  }
}
""",
)

COMPARE_COMPONENT = st.components.v2.component(
    "color_studio_compare",
    html="""<div class="compare"><img id="base"><div id="beforeWrap"><img id="before"></div><div id="handle"></div></div>""",
    css="""
.compare{position:relative;width:100%;overflow:hidden;border-radius:14px;line-height:0;border:1px solid rgba(139,92,246,.35)}
.compare img{display:block;width:100%;height:auto}
#beforeWrap{position:absolute;left:0;top:0;height:100%;width:50%;overflow:hidden}
#before{height:100%;width:auto;max-width:none}
#handle{position:absolute;top:0;bottom:0;left:50%;width:4px;background:linear-gradient(180deg,#8B5CF6,#38BDF8);cursor:ew-resize;
box-shadow:0 0 12px rgba(56,189,248,.8)}
""",
    js=r"""
export default function(component) {
  const {data, parentElement, setStateValue} = component
  const root=parentElement.querySelector(".compare"), base=root.querySelector("#base")
  const beforeWrap=root.querySelector("#beforeWrap"), before=root.querySelector("#before"), handle=root.querySelector("#handle")
  base.src=data.base||""; before.src=data.before||""; before.alt="Before image"; base.alt="After image"
  const setPos=p=>{
    p=Math.max(0,Math.min(100,p)); beforeWrap.style.width=p+"%"; handle.style.left=p+"%"
  }
  setPos(data.value||50)
  let drag=false
  root.onpointerdown=e=>{drag=true;root.setPointerCapture(e.pointerId)}
  root.onpointermove=e=>{if(!drag)return; const r=root.getBoundingClientRect(); setPos((e.clientX-r.left)/r.width*100)}
  root.onpointerup=e=>{if(!drag)return; drag=false; const r=root.getBoundingClientRect(); setStateValue("value",Math.round((e.clientX-r.left)/r.width*100))}
  root.onpointercancel=()=>{drag=false}
}
""",
)


# ---------------------------------------------------------------------------
# Backend Image Processing & Model Helpers (Preserved + Extended)
# ---------------------------------------------------------------------------
def b64_image(rgb: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def rgb_from_upload(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def restore_image(
    gray: np.ndarray,
    clahe: bool,
    denoise: bool,
    upscale: bool,
    brightness: int = 0,
    contrast: float = 1.0,
) -> np.ndarray:
    out = gray.copy()
    if brightness != 0 or contrast != 1.0:
        out = cv2.convertScaleAbs(out, alpha=float(contrast), beta=float(brightness))
    if clahe:
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
    if denoise:
        out = cv2.fastNlMeansDenoising(out, None, h=7, templateWindowSize=7, searchWindowSize=21)
    if upscale:
        out = cv2.resize(out, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


@st.cache_resource(max_entries=2)
def get_model() -> torch.nn.Module | None:
    checkpoint_path = get_checkpoint_path()
    if checkpoint_path is None:
        return None
    return load_model(str(checkpoint_path))


def model_colorize(model: torch.nn.Module, gray: np.ndarray) -> np.ndarray:
    lab_in = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB)
    h, w = gray.shape
    l_small = cv2.resize(lab_in[..., 0], (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    device = next(model.parameters()).device
    tensor = torch.from_numpy(l_small).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
    ab = pred.squeeze(0).float().cpu().permute(1, 2, 0).numpy() * 128.0
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)
    lab = np.empty((h, w, 3), dtype=np.float32)
    lab[..., 0] = lab_in[..., 0].astype(np.float32)
    lab[..., 1:] = ab
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 255).astype(np.uint8)


def lab_color(rgb_hex: str) -> tuple[float, float]:
    rgb = np.array([[
        [int(rgb_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    ]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]
    return float(lab[1] - 128.0), float(lab[2] - 128.0)


def apply_saturation_clamp(lab: np.ndarray) -> np.ndarray:
    ab = lab[..., 1:].astype(np.float32)
    mag = np.sqrt(np.square(ab[..., 0]) + np.square(ab[..., 1]))
    scale = np.minimum(1.0, 52.0 / np.maximum(mag, 1e-6))[..., None]
    lab[..., 1:] = ab * scale
    return lab


def apply_postprocessing(
    output_rgb: np.ndarray,
    gray: np.ndarray,
    preset: str,
    saturation_clamp: bool,
    face_protection: bool,
    strength: int = 100,
    hints: list | None = None,
    regions: list | None = None,
) -> np.ndarray:
    lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    for layer in hints or []:
        if not layer.get("visible", True):
            continue
        mask = png_to_mask(layer["mask"], gray.shape)
        a, b = lab_color(layer["color"])
        lab[..., 1][mask] = a + 128
        lab[..., 2][mask] = b + 128

    for layer in regions or []:
        if not layer.get("visible", True):
            continue
        mask = png_to_mask(layer["mask"], gray.shape)
        a, b = lab_color(layer["color"])
        lab[..., 1][mask] = a + 128
        lab[..., 2][mask] = b + 128

    if preset == "Vintage":
        lab[..., 1] *= 0.78
        lab[..., 2] *= 0.82
    elif preset == "Vibrant":
        lab[..., 1] *= 1.18
        lab[..., 2] *= 1.18
    elif preset == "Cinema":
        lab[..., 1] -= 3

    if saturation_clamp:
        lab = apply_saturation_clamp(lab)

    if face_protection:
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        faces = cv2.CascadeClassifier(cascade_path).detectMultiScale(gray_bgr, 1.1, 4)
        for (x, y, fw, fh) in faces:
            roi = lab[y:y + fh, x:x + fw]
            roi[..., 1] = np.clip(roi[..., 1], 128 + 3, 128 + 22)
            roi[..., 2] = np.clip(roi[..., 2], 128 + 8, 128 + 38)

    colorized = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 255).astype(np.uint8)
    if strength < 100:
        alpha = max(0.0, min(1.0, strength / 100.0))
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        colorized = cv2.addWeighted(colorized, alpha, gray_rgb, 1.0 - alpha, 0)
    return colorized


def dominant_colors(rgb: np.ndarray, count: int = 5) -> list[str]:
    pixels = cv2.resize(rgb, (min(192, rgb.shape[1]), min(192, rgb.shape[0])), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, labels, centers = cv2.kmeans(lab, count, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    totals = np.bincount(labels.ravel(), minlength=count)
    centers_lab = np.uint8(centers[:, None, :])
    colors = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB)[:, 0, :]
    order = np.argsort(totals)[::-1]
    return ["#" + "".join(f"{v:02X}" for v in colors[i]) for i in order]


def make_harmonies(seed_hex: str) -> dict[str, list[str]]:
    rgb = np.array([[
        [int(seed_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    ]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    h, s, v = hsv

    def palette(offsets):
        result = []
        for off in offsets:
            col = np.array([[
                [(h + off) % 180, min(220, s + 15), min(245, v)]
            ]], dtype=np.uint8)
            rr = cv2.cvtColor(col, cv2.COLOR_HSV2RGB)[0, 0]
            result.append("#" + "".join(f"{x:02X}" for x in rr))
        return result

    return {
        "Complementary": palette([0, 90]),
        "Analogous": palette([-30, -15, 0, 15, 30]),
        "Triadic": palette([0, 60, 120, 150, 30]),
    }


def mask_to_png(mask: np.ndarray) -> str:
    ok, enc = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    return base64.b64encode(enc.tobytes()).decode() if ok else ""


def png_to_mask(data: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(data)
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return np.zeros(shape, dtype=bool)
    arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr > 127


def snapshot() -> dict:
    return {
        "hints": st.session_state.get("hints", []),
        "regions": st.session_state.get("regions", []),
        "slider": st.session_state.get("compare_slider", 50),
        "preset": st.session_state.get("preset", "Balanced"),
        "restoration": {
            "clahe": st.session_state.get("restore_clahe", False),
            "denoise": st.session_state.get("restore_denoise", False),
            "upscale": st.session_state.get("restore_upscale", False),
            "brightness": st.session_state.get("brightness_adj", 0),
            "contrast": st.session_state.get("contrast_adj", 1.0),
            "strength": st.session_state.get("processing_strength", 100),
        },
    }


def push_history():
    st.session_state.history = st.session_state.get("history", [])
    idx = st.session_state.get("history_index", len(st.session_state.history) - 1)
    st.session_state.history = st.session_state.history[:idx + 1]
    st.session_state.history.append(snapshot())
    st.session_state.history_index = len(st.session_state.history) - 1


def apply_history(delta: int):
    history = st.session_state.get("history", [])
    index = st.session_state.get("history_index", 0)
    new_index = max(0, min(len(history) - 1, index + delta))
    if new_index == index:
        return
    state = history[new_index]
    st.session_state.hints = state.get("hints", [])
    st.session_state.regions = state.get("regions", [])
    st.session_state.compare_slider = state.get("slider", 50)
    st.session_state.preset = state.get("preset", "Balanced")
    st.session_state.history_index = new_index


def record_project(name: str, mode: str, preset: str):
    projects = st.session_state.setdefault("recent_projects", [])
    projects.insert(0, {
        "name": name,
        "mode": mode,
        "preset": preset,
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    st.session_state.recent_projects = projects[:6]


def navigate_to(page_name: str):
    st.session_state.nav_page = page_name


# ---------------------------------------------------------------------------
# Centralized Session State Initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "nav_page": "Dashboard",
    "hints": [],
    "regions": [],
    "history": [],
    "history_index": -1,
    "compare_slider": 50,
    "preset": "Balanced",
    "saturation_clamp": True,
    "face_protection": False,
    "restore_clahe": False,
    "restore_denoise": False,
    "restore_upscale": False,
    "brightness_adj": 0,
    "contrast_adj": 1.0,
    "processing_strength": 100,
    "stats_images": 0,
    "stats_videos": 0,
    "stats_batches": 0,
    "recent_projects": [],
    "batch_results": [],
    "uploader_nonce": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


model = get_model()


# ---------------------------------------------------------------------------
# Sidebar: Branding, Navigation, Photo Restoration Controls & History
# ---------------------------------------------------------------------------
PAGES = ["Dashboard", "Image Studio", "Video Studio", "Live Webcam", "Batch Studio"]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">CA</div>
            <div class="sidebar-brand-text">
                <div class="sidebar-brand-title">ChromaAI Studio</div>
                <div class="sidebar-brand-sub">Colorization &amp; Restoration</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_page = st.radio(
        "Workspace",
        PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )

    if model is not None:
        device_label = next(model.parameters()).device.type.upper()
        st.badge(f"Model Ready · {device_label}", icon=":material/check_circle:", color="green")
    else:
        st.badge("Checkpoint Missing", icon=":material/error:", color="red")

    st.subheader("Studio & Restoration")
    st.selectbox(
        "Color preset",
        ["Balanced", "Vintage", "Vibrant", "Cinema", "Natural"],
        key="preset",
        help="Choose the overall color character.",
    )
    st.slider(
        "Processing strength (%)",
        0,
        100,
        key="processing_strength",
        help="Blend between restored grayscale (0%) and full AI colorization (100%).",
    )
    st.slider(
        "Brightness",
        -50,
        50,
        key="brightness_adj",
        help="Adjust luminance brightness before colorization.",
    )
    st.slider(
        "Contrast",
        0.5,
        1.8,
        step=0.05,
        key="contrast_adj",
        help="Adjust luminance contrast before colorization.",
    )
    st.toggle(
        "Safe LAB saturation clamp",
        key="saturation_clamp",
        help="Prevents neon or out-of-gamut LAB colors.",
    )
    st.toggle(
        "Protect natural skin tones (Face)",
        key="face_protection",
        help="Detect faces and keep ab channels within a natural skin range.",
    )
    st.toggle(
        "Auto-contrast (CLAHE)",
        key="restore_clahe",
        help="Improve local contrast on the luminance channel.",
    )
    st.toggle(
        "Image denoising",
        key="restore_denoise",
        help="Reduce low-frequency noise before colorization.",
    )
    st.toggle(
        "2x image upscaling",
        key="restore_upscale",
        help="Upscale with Lanczos and a light sharpening kernel.",
    )

    st.subheader("Canvas Tool")
    tool = st.segmented_control(
        "Editing tool",
        ["Hint", "Region", "Eraser"],
        default="Hint",
        key="editing_tool",
        help="Select an interactive canvas editing mode.",
    )
    tolerance = st.slider(
        "Region lightness tolerance",
        1,
        80,
        18,
        key="region_tolerance",
        help="Similar-lightness flood-fill tolerance.",
    )
    hint_color = st.color_picker(
        "Hint / region color",
        "#D86C4A",
        key="hint_color_picker",
        help="Applied to the selected segment or region.",
    )

    st.subheader("History")
    undo_col, redo_col = st.columns(2)
    if undo_col.button("Undo", icon=":material/undo:", width="stretch", help="Undo last edit", key="undo_btn"):
        apply_history(-1)
        st.rerun()
    if redo_col.button("Redo", icon=":material/redo:", width="stretch", help="Redo edit", key="redo_btn"):
        apply_history(1)
        st.rerun()
    if st.button("Reset edits", icon=":material/restart_alt:", width="stretch",
                 help="Clear hints and regions only.", key="reset_edits"):
        st.session_state.hints = []
        st.session_state.regions = []
        push_history()
        st.rerun()
    if st.button("Reset all", icon=":material/delete_sweep:", width="stretch",
                 help="Restore every studio setting and layer.", key="reset_all"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Top Header
# ---------------------------------------------------------------------------
st.title("ChromaAI Studio", icon=":material/palette:")
if model is None:
    st.warning(
        f"No checkpoint found. Expected `{CHECKPOINT_BEST}` (fallback: `{CHECKPOINT_LATEST}`). "
        "Train with `python train.py --epochs 3`.",
        icon=":material/error:",
    )


# ---------------------------------------------------------------------------
# Webcam VideoProcessorClass (defined once, used in Live Webcam workspace)
# ---------------------------------------------------------------------------
class ColorizationProcessor(VideoProcessorBase):
    def __init__(self, model_ref, resolution, wipe, face, clamp, preset="Balanced",
                 brightness=0, contrast=1.0, strength=100):
        self.model = model_ref
        self.device = next(model_ref.parameters()).device
        self.resolution = resolution
        self.wipe = wipe
        self.face = face
        self.clamp = clamp
        self.preset = preset
        self.brightness = brightness
        self.contrast = contrast
        self.strength = strength
        self.lock = threading.Lock()
        self.latest_gray = None
        self.latest_color = None
        self.record_writer = None
        self.last_frame_time = None

    def update(self, resolution, wipe, face, clamp, preset="Balanced",
               brightness=0, contrast=1.0, strength=100):
        with self.lock:
            self.resolution = resolution
            self.wipe = wipe
            self.face = face
            self.clamp = clamp
            self.preset = preset
            self.brightness = brightness
            self.contrast = contrast
            self.strength = strength

    def start_recording(self, path, width, height, fps=20):
        with self.lock:
            if self.record_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.record_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def stop_recording(self):
        with self.lock:
            if self.record_writer is not None:
                self.record_writer.release()
                self.record_writer = None

    def recv(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        with self.lock:
            resolution = self.resolution
            wipe = self.wipe
            face = self.face
            clamp = self.clamp
            preset = self.preset
            brightness = self.brightness
            contrast = self.contrast
            strength = self.strength

        if brightness != 0 or contrast != 1.0:
            gray = cv2.convertScaleAbs(gray, alpha=float(contrast), beta=float(brightness))

        lab_in = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB)
        h, w = gray.shape
        l_small = cv2.resize(lab_in[..., 0], (resolution, resolution),
                             interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        tensor = torch.from_numpy(l_small).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
        ab = pred.squeeze(0).float().cpu().permute(1, 2, 0).numpy() * 128.0
        ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)
        lab = np.empty((h, w, 3), dtype=np.float32)
        lab[..., 0] = lab_in[..., 0]
        lab[..., 1:] = ab

        if preset == "Vintage":
            lab[..., 1] *= 0.78
            lab[..., 2] *= 0.82
        elif preset == "Vibrant":
            lab[..., 1] *= 1.18
            lab[..., 2] *= 1.18
        elif preset == "Cinema":
            lab[..., 1] -= 3

        if clamp:
            lab = apply_saturation_clamp(lab)
        if face:
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            ).detectMultiScale(gray_bgr, 1.1, 4)
            for (x, y, fw, fh) in faces:
                roi = lab[y:y + fh, x:x + fw]
                roi[..., 1] = np.clip(roi[..., 1], 131, 150)
                roi[..., 2] = np.clip(roi[..., 2], 136, 166)

        colorized = cv2.cvtColor(
            np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if strength < 100:
            alpha = max(0.0, min(1.0, strength / 100.0))
            colorized = cv2.addWeighted(colorized, alpha, gray_bgr, 1.0 - alpha, 0)

        split = int(w * wipe / 100.0)
        output = colorized.copy()
        output[:, :split] = gray_bgr[:, :split]
        cv2.line(output, (split, 0), (split, h), (255, 255, 255), 2)
        with self.lock:
            self.latest_gray = gray_bgr.copy()
            self.latest_color = colorized.copy()
            if self.record_writer is not None:
                self.record_writer.write(output)
        now = time.perf_counter()
        fps = 1.0 / (now - self.last_frame_time) if self.last_frame_time else 0.0
        self.last_frame_time = now
        cv2.putText(output, f"FPS: {fps:.1f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, f"FPS: {fps:.1f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(output, format="bgr24")


# ===========================================================================
# PAGE 1: DASHBOARD
# ===========================================================================
if active_page == "Dashboard":
    st.markdown(
        """
        <div class="studio-hero">
            <div class="studio-hero-badge">AI Vision Studio · UNet LAB Engine</div>
            <h1>Welcome to ChromaAI Studio</h1>
            <p>
                Restore vintage photography, colorize monochrome images and videos in real time,
                fine-tune hues with interactive superpixel hints, and process entire collections in batches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Statistics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images Colorized", st.session_state.stats_images, border=True)
    m2.metric("Videos Rendered", st.session_state.stats_videos, border=True)
    m3.metric("Batch Jobs", st.session_state.stats_batches, border=True)
    model_state = "Active" if model is not None else "Offline"
    m4.metric("AI Model", model_state, border=True)

    st.subheader("Workspaces & Quick Actions", anchor=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">Image Colorization Studio</div>
                <p class="glass-card-desc">
                    Upload grayscale photos, apply CLAHE &amp; denoising restoration, compare with an
                    interactive split-screen slider, and refine colors with smart superpixel hints.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Image Studio",
            icon=":material/image:",
            type="primary",
            width="stretch",
            key="quick_image",
            on_click=navigate_to,
            args=("Image Studio",),
        )

        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">Live Webcam Studio</div>
                <p class="glass-card-desc">
                    Stream real-time AI colorization directly from your browser camera with adjustable
                    resolution, split-screen wipe, instant snapshots, and MP4 recording.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Live Webcam",
            icon=":material/videocam:",
            type="primary",
            width="stretch",
            key="quick_webcam",
            on_click=navigate_to,
            args=("Live Webcam",),
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">Video Colorization Studio</div>
                <p class="glass-card-desc">
                    Colorize MP4, AVI, MOV, or MKV clips frame-by-frame, preview individual frames,
                    set preset keyframes, and export with original audio preserved.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Video Studio",
            icon=":material/movie:",
            type="primary",
            width="stretch",
            key="quick_video",
            on_click=navigate_to,
            args=("Video Studio",),
        )

        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">Batch Processing Studio</div>
                <p class="glass-card-desc">
                    Upload multiple monochrome photos at once, track per-image progress, preview results
                    in a responsive gallery, and download everything as a single ZIP archive.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Open Batch Studio",
            icon=":material/burst_mode:",
            type="primary",
            width="stretch",
            key="quick_batch",
            on_click=navigate_to,
            args=("Batch Studio",),
        )

    st.subheader("Recent Projects", anchor=False)
    recent = st.session_state.get("recent_projects", [])
    if not recent:
        st.markdown(
            """
            <div class="empty-state">
                <h4>No recent projects yet</h4>
                <p>Colorize an image, render a video, or run a batch job to see your recent activity here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        rcols = st.columns(min(3, len(recent)))
        for idx, item in enumerate(recent[:3]):
            with rcols[idx]:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="glass-card-title">{item['name']}</div>
                        <p class="glass-card-desc">{item['mode']} · Preset: {item['preset']} · {item['time']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ===========================================================================
# PAGE 2: IMAGE COLORIZATION STUDIO
# ===========================================================================
elif active_page == "Image Studio":
    st.subheader("Image Colorization & Restoration Studio", anchor=False)
    st.caption("Adjust restoration, brightness, contrast, presets, and skin protection in the sidebar.")

    uploaded = st.file_uploader(
        "Drag and drop a grayscale or vintage image",
        type=["png", "jpg", "jpeg"],
        key=f"image_upload_{st.session_state.uploader_nonce}",
        help="Supported formats: PNG, JPG, JPEG.",
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="empty-state">
                <h4>Upload an image to begin</h4>
                <p>Drop a PNG or JPG photo above to preview, restore, colorize, and compare before &amp; after.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        rgb = rgb_from_upload(uploaded)
        gray = to_gray(rgb)
        gray = restore_image(
            gray,
            st.session_state.get("restore_clahe", False),
            st.session_state.get("restore_denoise", False),
            st.session_state.get("restore_upscale", False),
            st.session_state.get("brightness_adj", 0),
            st.session_state.get("contrast_adj", 1.0),
        )

        prev_col, action_col = st.columns([2, 1], vertical_alignment="top")
        with prev_col:
            st.image(gray, caption=f"Input Preview ({gray.shape[1]}x{gray.shape[0]})",
                     width="stretch", alt="Restored grayscale preview")
        with action_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-title">Ready to Colorize</div>
                    <p class="glass-card-desc">
                        Run the UNet LAB colorizer with your active sidebar restoration &amp; preset settings.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Colorize Image", icon=":material/auto_awesome:", type="primary",
                         width="stretch", key="colorize_image",
                         help="Run the AI model and extract a smart color palette."):
                if model is None:
                    st.error("Load a model checkpoint before colorizing.")
                else:
                    prog = st.progress(15, text="Preparing luminance tensor...")
                    with st.spinner("Running AI colorization..."):
                        raw_output = model_colorize(model, gray)
                        prog.progress(65, text="Applying restoration, hints & preset...")
                        output = apply_postprocessing(
                            raw_output,
                            gray,
                            st.session_state.preset,
                            st.session_state.saturation_clamp,
                            st.session_state.face_protection,
                            st.session_state.processing_strength,
                            st.session_state.hints,
                            st.session_state.regions,
                        )
                        st.session_state.output_rgb = output
                        st.session_state.input_gray = gray
                        st.session_state.palette = dominant_colors(output)
                        st.session_state.stats_images += 1
                        record_project(uploaded.name, "Image Studio", st.session_state.preset)
                        push_history()
                        prog.progress(100, text="Colorization complete!")
                    st.success("Image colorized successfully!", icon=":material/check_circle:")

            if st.button("Reset edits", icon=":material/restart_alt:", width="stretch",
                         key="reset_image_edits_btn", help="Clear hints, regions, and output preview."):
                st.session_state.hints = []
                st.session_state.regions = []
                st.session_state.pop("output_rgb", None)
                st.rerun()

            if st.button("Upload another image", icon=":material/upload_file:", width="stretch",
                         key="clear_image_btn", help="Clear the current image and upload a new one."):
                st.session_state.pop("output_rgb", None)
                st.session_state.pop("input_gray", None)
                st.session_state.hints = []
                st.session_state.regions = []
                st.session_state.uploader_nonce += 1
                st.rerun()

        if "output_rgb" in st.session_state:
            gray_show = st.session_state.input_gray
            output = st.session_state.output_rgb
            gray_rgb = cv2.cvtColor(gray_show, cv2.COLOR_GRAY2RGB)

            st.subheader("Before & After Comparison", anchor=False)
            value = st.session_state.get("compare_slider", 50)
            result = COMPARE_COMPONENT(
                key="compare_component",
                data={"base": b64_image(output), "before": b64_image(gray_rgb), "value": value},
                default={"value": value},
                on_value_change=lambda: None,
            )
            if isinstance(result, dict) and "value" in result:
                st.session_state.compare_slider = int(result["value"])
            st.slider(
                "Split-screen position",
                0,
                100,
                st.session_state.compare_slider,
                key="compare_slider_control",
                help="Drag the handle on the image or adjust this slider.",
                on_change=lambda: st.session_state.update(
                    compare_slider=st.session_state.compare_slider_control
                ),
            )

            side_a, side_b = st.columns(2)
            side_a.image(gray_rgb, caption="Original / Restored Grayscale", width="stretch", alt="Original grayscale")
            side_b.image(output, caption="AI Colorized Result", width="stretch", alt="Colorized output")

            ok_png, png_buf = cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            if ok_png:
                st.download_button(
                    "Download Colorized Image (PNG)",
                    data=png_buf.tobytes(),
                    file_name=f"{Path(uploaded.name).stem}-colorized.png",
                    mime="image/png",
                    icon=":material/download:",
                    width="stretch",
                    key="quick_download_png",
                )

            st.subheader("Smart Palette", anchor=False)
            palette = st.session_state.get("palette", [])
            if palette:
                cols = st.columns(len(palette))
                for i, hex_color in enumerate(palette):
                    if cols[i].button(hex_color, key=f"palette_chip_{i}", width="stretch",
                                      help="Apply this color to the next hint or region."):
                        st.session_state.selected_palette_color = hex_color
                st.caption("Dominant colors extracted from the current result.")

            c1, c2 = st.columns(2)
            if c1.button("Suggest hints", icon=":material/auto_fix_high:", width="stretch",
                         help="Detect SLIC superpixels and start segment assignment.",
                         key="suggest_hints"):
                labels = segmentation.slic(
                    cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2LAB),
                    n_segments=48, compactness=10, sigma=1, start_label=0,
                )
                st.session_state.superpixels = labels
                st.session_state.overlay_segments = True
            if c2.button("Randomize palette", icon=":material/casino:", width="stretch",
                         help="Preview harmonious five-color palettes.",
                         key="randomize_palette"):
                seed = palette[0] if palette else "#7A9EC2"
                st.session_state.harmonies = make_harmonies(seed)

            if "harmonies" in st.session_state:
                st.subheader("Harmonious Palettes", anchor=False)
                pcols = st.columns(len(st.session_state.harmonies))
                for col, (name, colors) in zip(pcols, st.session_state.harmonies.items()):
                    swatches = "".join(
                        f'<span style="display:inline-block;width:18%;height:34px;border-radius:4px;margin-right:2px;background:{c}"></span>'
                        for c in colors
                    )
                    col.markdown(swatches, unsafe_allow_html=True)
                    if col.button(name, key=f"harmony_{name}", width="stretch"):
                        st.session_state.palette = colors

            st.subheader("Interactive Canvas", anchor=False)
            overlay = st.session_state.get("overlay_segments", False)
            canvas_rgb = output.copy()
            if overlay and "superpixels" in st.session_state:
                boundaries = segmentation.find_boundaries(st.session_state.superpixels)
                canvas_rgb[boundaries] = np.array([255, 255, 255])

            canvas_result = CANVAS_COMPONENT(
                key="image_canvas",
                data={
                    "image": b64_image(canvas_rgb),
                    "width": canvas_rgb.shape[1],
                    "height": canvas_rgb.shape[0],
                    "status": f"{tool or 'Hint'} mode",
                    "alt": "Interactive colorization canvas",
                },
                on_click_change=lambda: None,
            )
            click = getattr(canvas_result, "click", None)
            if isinstance(click, dict):
                x, y = int(click["x"]), int(click["y"])
                if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                    if tool == "Eraser":
                        st.session_state.hints = [
                            l for l in st.session_state.hints
                            if not png_to_mask(l["mask"], gray.shape)[y, x]
                        ]
                        st.session_state.regions = [
                            l for l in st.session_state.regions
                            if not png_to_mask(l["mask"], gray.shape)[y, x]
                        ]
                        push_history()
                        st.rerun()
                    elif tool == "Hint" and "superpixels" in st.session_state:
                        label = st.session_state.superpixels[y, x]
                        mask = st.session_state.superpixels == label
                        st.session_state.hints.append({
                            "id": f"hint-{len(st.session_state.hints) + 1}",
                            "type": "Hint",
                            "color": st.session_state.get("selected_palette_color", hint_color),
                            "visible": True,
                            "mask": mask_to_png(mask),
                        })
                        push_history()
                        st.rerun()
                    elif tool == "Region":
                        light = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB)[..., 0]
                        seed = (x, y)
                        height, width = gray.shape
                        mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
                        cv2.floodFill(light, mask, seed, 255,
                                      loDiff=tolerance, upDiff=tolerance,
                                      flags=cv2.FLOODFILL_FIXED_RANGE)
                        region_mask = mask[1:-1, 1:-1] > 0
                        st.session_state.regions.append({
                            "id": f"region-{len(st.session_state.regions) + 1}",
                            "type": "Region",
                            "color": st.session_state.get("selected_palette_color", hint_color),
                            "visible": True,
                            "mask": mask_to_png(region_mask),
                        })
                        push_history()
                        st.rerun()

            st.toggle("Show segment overlay", key="overlay_segments",
                      help="Display SLIC superpixel boundaries.")

            st.subheader("Layers", anchor=False)
            if not st.session_state.hints and not st.session_state.regions:
                st.caption("No hints or regions yet.")
            for layer in st.session_state.hints + st.session_state.regions:
                row = st.container(border=True)
                swatch, name, visible, delete = row.columns([1, 4, 2, 1])
                swatch.markdown(
                    f'<div style="width:36px;height:36px;border-radius:8px;background:{layer["color"]}"></div>',
                    unsafe_allow_html=True,
                )
                name.write(f"{layer['type']} · {layer['id']}")
                layer["visible"] = visible.toggle(
                    "Visible",
                    value=layer["visible"],
                    key=f"visible_{layer['id']}",
                    help="Toggle this layer on or off.",
                )
                if delete.button("Delete", key=f"delete_{layer['id']}",
                                 help="Delete this hint or region.", width="stretch"):
                    st.session_state.hints = [l for l in st.session_state.hints if l["id"] != layer["id"]]
                    st.session_state.regions = [l for l in st.session_state.regions if l["id"] != layer["id"]]
                    push_history()
                    st.rerun()

            st.subheader("Advanced Export", anchor=False)
            export_format = st.segmented_control(
                "Format", ["PNG", "JPG", "GIF"], default="PNG",
                key="export_format", help="Choose the output format.",
            )
            quality = st.slider("JPG quality", 10, 100, 92, key="export_quality",
                                help="JPEG encoder quality.")
            side_by_side = st.toggle(
                "Include side-by-side comparison",
                key="export_side_by_side",
                help="Place grayscale and colorized images beside each other.",
            )
            if st.button("Prepare export", icon=":material/download:", width="stretch",
                         help="Create the selected export for download.", key="prepare_export"):
                gray_rgb_export = cv2.cvtColor(st.session_state.input_gray, cv2.COLOR_GRAY2RGB)
                export_rgb = np.concatenate((gray_rgb_export, output), axis=1) if side_by_side else output
                if export_format == "PNG":
                    _, buf = cv2.imencode(".png", cv2.cvtColor(export_rgb, cv2.COLOR_RGB2BGR))
                    data, mime, name = buf.tobytes(), "image/png", "colorized.png"
                elif export_format == "JPG":
                    _, buf = cv2.imencode(".jpg", cv2.cvtColor(export_rgb, cv2.COLOR_RGB2BGR),
                                          [cv2.IMWRITE_JPEG_QUALITY, quality])
                    data, mime, name = buf.tobytes(), "image/jpeg", "colorized.jpg"
                else:
                    frames = []
                    for _ in st.session_state.hints + st.session_state.regions:
                        frames.append(Image.fromarray(output))
                    pil_frames = frames or [Image.fromarray(output)]
                    bio = io.BytesIO()
                    pil_frames[0].save(bio, format="GIF", save_all=True,
                                       append_images=pil_frames[1:], duration=700, loop=0)
                    data, mime, name = bio.getvalue(), "image/gif", "progress.gif"
                st.download_button("Download export", data=data, file_name=name, mime=mime,
                                   width="stretch", key="download_export")

        st.subheader("Session", anchor=False)
        session_col, load_col = st.columns(2)
        if session_col.download_button(
            "Save session JSON",
            data=json.dumps(snapshot(), indent=2),
            file_name="color-studio-session.json",
            mime="application/json",
            width="stretch",
            help="Save hints, regions, slider, preset and restoration settings.",
            key="save_session",
        ):
            st.toast("Session saved")
        loaded_session = load_col.file_uploader(
            "Load session JSON", type=["json"],
            key="load_session", help="Resume a saved JSON session.",
        )
        if loaded_session is not None:
            try:
                state = json.loads(loaded_session.getvalue())
                st.session_state.hints = state.get("hints", [])
                st.session_state.regions = state.get("regions", [])
                st.session_state.compare_slider = state.get("slider", 50)
                st.session_state.preset = state.get("preset", "Balanced")
                restore = state.get("restoration", {})
                st.session_state.restore_clahe = restore.get("clahe", False)
                st.session_state.restore_denoise = restore.get("denoise", False)
                st.session_state.restore_upscale = restore.get("upscale", False)
                push_history()
                st.success("Session loaded.")
            except Exception as exc:
                st.error(f"Could not load session: {exc}")


# ===========================================================================
# PAGE 3: VIDEO COLORIZATION STUDIO
# ===========================================================================
elif active_page == "Video Studio":
    st.subheader("Video Colorization Studio", anchor=False)
    uploaded_video = st.file_uploader(
        "Upload a monochrome or vintage video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
        help="Original audio can be retained in the exported MP4.",
    )
    if uploaded_video is None:
        st.markdown(
            """
            <div class="empty-state">
                <h4>No video uploaded</h4>
                <p>Upload an MP4, AVI, MOV, or MKV file to preview frames, add preset keyframes, and render.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp:
            tmp.write(uploaded_video.getbuffer())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0.0
        cap.release()

        v_left, v_right = st.columns([2, 1], vertical_alignment="top")
        with v_left:
            st.video(video_path, width="stretch", alt="Uploaded source video")
        with v_right:
            st.metric("Total Frames", frame_count, border=True)
            st.metric("Duration", f"{duration:.1f}s", border=True)
            keep_audio = st.toggle(
                "Keep original audio",
                value=True,
                key="video_keep_audio",
                help="Mux the original soundtrack back into the rendered MP4.",
            )

        frame_index = st.slider(
            "Preview frame",
            0,
            max(frame_count - 1, 0),
            0,
            key="video_preview_frame",
            help="Seek to any frame in the uploaded video.",
        )

        if st.button("Preview Frame Comparison", icon=":material/compare:", width="stretch",
                     key="preview_frame_colorize_btn", help="Colorize the selected frame for before/after preview."):
            if model is None:
                st.error("Load a model checkpoint before previewing.")
            else:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = cap.read()
                cap.release()
                if ok and frame_bgr is not None:
                    gray = to_gray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    gray = restore_image(
                        gray,
                        st.session_state.restore_clahe,
                        st.session_state.restore_denoise,
                        False,
                        st.session_state.brightness_adj,
                        st.session_state.contrast_adj,
                    )
                    raw_col = model_colorize(model, gray)
                    col_rgb = apply_postprocessing(
                        raw_col, gray, st.session_state.preset,
                        st.session_state.saturation_clamp, st.session_state.face_protection,
                        st.session_state.processing_strength,
                    )
                    f1, f2 = st.columns(2)
                    f1.image(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), caption=f"Frame {frame_index} (Original)",
                             width="stretch", alt="Original video frame")
                    f2.image(col_rgb, caption=f"Frame {frame_index} (Colorized)",
                             width="stretch", alt="Colorized video frame")

        st.subheader("Timeline Keyframes", anchor=False)
        kf_col1, kf_col2, kf_col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        keyframe_time = kf_col1.slider(
            "Add keyframe at (seconds)", 0.0, max(duration, 0.1), 0.0, 0.25,
            key="video_keyframe_time", help="Assign a preset to this time range.",
        )
        keyframe_preset = kf_col2.segmented_control(
            "Keyframe preset",
            ["Balanced", "Vintage", "Vibrant", "Cinema", "Natural"],
            default="Balanced",
            key="video_keyframe_preset",
            help="Preset applied to this keyframe and following frames.",
        )
        if kf_col3.button("Add", icon=":material/add:", width="stretch",
                          help="Store a preset keyframe.", key="add_keyframe"):
            keyframes = st.session_state.setdefault("video_keyframes", [])
            keyframes.append({"time": keyframe_time, "preset": keyframe_preset})
            keyframes.sort(key=lambda item: item["time"])
            st.toast(f"Added {keyframe_preset} keyframe at {keyframe_time:.2f}s")

        if st.button("Render Colorized Video", icon=":material/movie:", type="primary",
                     width="stretch", help="Colorize every frame and encode an MP4.", key="render_video"):
            if model is None:
                st.error("Load a model checkpoint before rendering.")
            else:
                with tempfile.TemporaryDirectory() as td:
                    silent_output = str(Path(td) / "silent.mp4")
                    final_output = str(Path(td) / "colorized.mp4")
                    progress = st.progress(10, text="Colorizing video frames...")
                    with st.spinner("Rendering video..."):
                        colorize_video(model, video_path, silent_output)
                        progress.progress(80, text="Finalizing MP4 container...")
                        if keep_audio and shutil.which("ffmpeg"):
                            subprocess.run([
                                "ffmpeg", "-y", "-i", silent_output, "-i", video_path,
                                "-c:v", "copy", "-c:a", "aac", "-shortest", final_output,
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                        else:
                            shutil.copyfile(silent_output, final_output)
                    progress.progress(100, text="Video rendering complete!")
                    st.session_state.stats_videos += 1
                    record_project(uploaded_video.name, "Video Studio", st.session_state.preset)
                    st.success("Video rendered successfully!", icon=":material/check_circle:")
                    st.video(final_output, width="stretch", alt="Rendered colorized video")
                    with open(final_output, "rb") as f:
                        st.download_button(
                            "Download Colorized Video",
                            f.read(),
                            f"{Path(uploaded_video.name).stem}-colorized.mp4",
                            "video/mp4",
                            icon=":material/download:",
                            width="stretch",
                            key="download_video",
                        )


# ===========================================================================
# PAGE 4: LIVE WEBCAM STUDIO
# ===========================================================================
elif active_page == "Live Webcam":
    st.subheader("Live Webcam Colorization Studio", anchor=False)
    cam_left, cam_right = st.columns([3, 1], vertical_alignment="top")

    with cam_left:
        ctrl1, ctrl2 = st.columns(2)
        resolution = ctrl1.slider(
            "Processing resolution",
            128,
            256,
            256,
            64,
            key="webcam_resolution",
            help="Lower resolution improves CPU speed.",
        )
        wipe = ctrl2.slider(
            "Split-screen comparison (%)",
            0,
            100,
            50,
            key="webcam_wipe",
            help="Move the wipe between grayscale and AI colorized video.",
        )

    if model is not None and "pytest" not in sys.argv[0].lower():
        with cam_left:
            context = webrtc_streamer(
                key="colorization-webcam",
                video_processor_factory=lambda: ColorizationProcessor(
                    model,
                    resolution,
                    wipe,
                    st.session_state.face_protection,
                    st.session_state.saturation_clamp,
                    st.session_state.preset,
                    st.session_state.brightness_adj,
                    st.session_state.contrast_adj,
                    st.session_state.processing_strength,
                ),
                media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                async_processing=True,
            )
        if context.video_processor is not None:
            context.video_processor.update(
                resolution,
                wipe,
                st.session_state.face_protection,
                st.session_state.saturation_clamp,
                st.session_state.preset,
                st.session_state.brightness_adj,
                st.session_state.contrast_adj,
                st.session_state.processing_strength,
            )

        with cam_right:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="glass-card-title">Camera Controls</div>
                    <p class="glass-card-desc">Use START / STOP on the player to toggle your webcam, or capture a snapshot.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            capture_clicked = st.button(
                "Capture Image",
                icon=":material/photo_camera:",
                type="primary",
                width="stretch",
                help="Save the latest colorized webcam frame.",
                key="webcam_snapshot",
            )
            if capture_clicked and context.video_processor is not None:
                frame = context.video_processor.latest_color
                if frame is not None:
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if ok:
                        st.session_state.webcam_captured_jpg = encoded.tobytes()
                        st.session_state.stats_images += 1
                        record_project("Webcam Snapshot", "Live Webcam", st.session_state.preset)
                        st.success("Snapshot captured!")
                else:
                    st.warning("No frame available yet. Start the webcam first.")

            if "webcam_captured_jpg" in st.session_state:
                st.image(st.session_state.webcam_captured_jpg, caption="Captured Result",
                         width="stretch", alt="Captured webcam frame")
                st.download_button(
                    "Download Captured Result",
                    st.session_state.webcam_captured_jpg,
                    "webcam-snapshot.jpg",
                    "image/jpeg",
                    icon=":material/download:",
                    width="stretch",
                    key="download_snapshot",
                )

            rec_col, stop_col = st.columns(2)
            if rec_col.button("Record", icon=":material/videocam:", width="stretch",
                              help="Start MP4 recording.", key="start_webcam_record"):
                if context.video_processor is not None:
                    context.video_processor.start_recording(
                        tempfile.gettempdir() + "/color-studio-webcam.mp4", 640, 480
                    )
                    st.toast("Recording started")
            if stop_col.button("Stop Rec", icon=":material/stop:", width="stretch",
                               help="Stop MP4 recording.", key="stop_webcam_record"):
                if context.video_processor is not None:
                    context.video_processor.stop_recording()
                    record_path = tempfile.gettempdir() + "/color-studio-webcam.mp4"
                    if Path(record_path).exists():
                        st.download_button(
                            "Download Recording",
                            Path(record_path).read_bytes(),
                            "colorized-webcam.mp4",
                            "video/mp4",
                            width="stretch",
                            key="download_webcam_recording",
                        )

        if not context.state.playing:
            st.info("Camera not connected. Click START above and allow browser camera access.",
                    icon=":material/videocam:")
    else:
        st.info("Start the app with a trained model checkpoint to enable live webcam colorization.",
                icon=":material/videocam:")


# ===========================================================================
# PAGE 5: BATCH PROCESSING STUDIO
# ===========================================================================
elif active_page == "Batch Studio":
    st.subheader("Batch Colorization Studio", anchor=False)
    uploads = st.file_uploader(
        "Upload multiple grayscale images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_upload",
        help="Active sidebar restoration & preset settings are applied to every image.",
    )

    if not uploads:
        st.markdown(
            """
            <div class="empty-state">
                <h4>No batch images selected</h4>
                <p>Select multiple PNG or JPG files above to preview the gallery and colorize them all at once.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"{len(uploads)} image(s) queued for processing.")
        gallery_cols = st.columns(min(4, len(uploads)))
        for idx, up in enumerate(uploads):
            with gallery_cols[idx % len(gallery_cols)]:
                st.image(up.getvalue(), caption=up.name, width="stretch", alt=f"Uploaded {up.name}")

        if st.button("Process All Images", icon=":material/burst_mode:", type="primary",
                     width="stretch", key="process_batch",
                     help="Colorize all uploaded images and package as a ZIP."):
            if model is None:
                st.error("Load a model checkpoint before batch processing.")
            else:
                results = []
                prog = st.progress(0, text="Starting batch colorization...")
                with tempfile.TemporaryDirectory() as td:
                    zip_path = Path(td) / "colorized-batch.zip"
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for idx, upload in enumerate(uploads):
                            rgb = rgb_from_upload(upload)
                            gray = restore_image(
                                to_gray(rgb),
                                st.session_state.restore_clahe,
                                st.session_state.restore_denoise,
                                st.session_state.restore_upscale,
                                st.session_state.brightness_adj,
                                st.session_state.contrast_adj,
                            )
                            raw_out = model_colorize(model, gray)
                            out = apply_postprocessing(
                                raw_out,
                                gray,
                                st.session_state.preset,
                                st.session_state.saturation_clamp,
                                st.session_state.face_protection,
                                st.session_state.processing_strength,
                            )
                            ok, enc = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                            if ok:
                                png_bytes = enc.tobytes()
                                out_name = Path(upload.name).stem + "-colorized.png"
                                zf.writestr(out_name, png_bytes)
                                results.append({"name": out_name, "bytes": png_bytes, "rgb": out})
                            prog.progress(int(((idx + 1) / len(uploads)) * 100),
                                          text=f"Processed {idx + 1}/{len(uploads)}: {upload.name}")
                    st.session_state.batch_results = results
                    st.session_state.batch_zip_bytes = zip_path.read_bytes()
                    st.session_state.stats_batches += 1
                    st.session_state.stats_images += len(results)
                    record_project(f"Batch ({len(results)} images)", "Batch Studio", st.session_state.preset)
                st.success(f"Completed batch processing for {len(results)} image(s)!",
                           icon=":material/check_circle:")

        if st.session_state.get("batch_results"):
            st.subheader("Processed Results", anchor=False)
            res_list = st.session_state.batch_results
            rcols = st.columns(min(4, len(res_list)))
            for idx, item in enumerate(res_list):
                with rcols[idx % len(rcols)]:
                    st.badge("Done", icon=":material/check:", color="green")
                    st.image(item["rgb"], caption=item["name"], width="stretch", alt=item["name"])
                    st.download_button(
                        "Download",
                        data=item["bytes"],
                        file_name=item["name"],
                        mime="image/png",
                        width="stretch",
                        key=f"batch_dl_{idx}",
                    )
            st.download_button(
                "Download All Results as ZIP",
                data=st.session_state.batch_zip_bytes,
                file_name="colorized-batch.zip",
                mime="application/zip",
                icon=":material/folder_zip:",
                width="stretch",
                key="download_batch",
            )
