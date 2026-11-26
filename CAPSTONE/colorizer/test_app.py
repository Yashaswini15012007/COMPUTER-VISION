import io
import sys
import tempfile
import zipfile
from pathlib import Path

import av
import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).parent
APP_PATH = str(APP_DIR / "app.py")
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from inference import colorize_video, load_model
import app as studio_app


def test_app_starts():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert at.title


def test_sidebar_controls_render():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert at.selectbox
    assert at.toggle


def test_switch_all_workspaces():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    for page in ["Dashboard", "Image Studio", "Video Studio", "Live Webcam", "Batch Studio"]:
        at.radio(key="nav_page").set_value(page).run()
        assert not at.exception, f"Exception on page {page}: {at.exception}"


def test_image_colorization_pipeline():
    model = load_model()
    gray = np.tile(np.linspace(30, 220, 128, dtype=np.uint8), (96, 1))
    restored = studio_app.restore_image(gray, clahe=True, denoise=False, upscale=False, brightness=5, contrast=1.05)
    assert restored.shape == gray.shape
    assert restored.dtype == np.uint8

    raw_rgb = studio_app.model_colorize(model, restored)
    assert raw_rgb.shape == (96, 128, 3)
    assert raw_rgb.dtype == np.uint8
    assert 40.0 < float(raw_rgb.mean()) < 215.0
    assert not np.isnan(raw_rgb).any()
    assert raw_rgb.max() > 0, "Colorized image must not be completely black"

    final_rgb = studio_app.apply_postprocessing(
        raw_rgb, restored, preset="Balanced", saturation_clamp=True, face_protection=True, strength=100
    )
    assert final_rgb.shape == (96, 128, 3)
    assert final_rgb.dtype == np.uint8
    assert final_rgb.max() > 0


def test_colorization_multi_image_types():
    """Verify portrait, landscape, object, checkerboard, and float32 inputs are never black."""
    model = load_model()

    # 1. Portrait-like image
    portrait = np.zeros((150, 120), dtype=np.uint8)
    cv2.ellipse(portrait, (60, 75), (35, 45), 0, 0, 360, 180, -1)
    res_portrait = studio_app.model_colorize(model, portrait)
    assert res_portrait.shape == (150, 120, 3)
    assert res_portrait.mean() > 10.0

    # 2. Landscape-like gradient
    landscape = np.zeros((100, 160), dtype=np.uint8)
    for i in range(50):
        landscape[i, :] = int(180 + i * 1.2)
    for i in range(50, 100):
        landscape[i, :] = int(50 + (i - 50) * 1.5)
    res_land = studio_app.model_colorize(model, landscape)
    assert res_land.shape == (100, 160, 3)
    assert res_land.mean() > 20.0

    # 3. Object-like image with shapes
    obj = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(obj, (60, 60), 40, 200, -1)
    res_obj = studio_app.model_colorize(model, obj)
    assert res_obj.shape == (120, 120, 3)
    assert res_obj.mean() > 15.0

    # 4. Checkerboard pattern
    checker = np.zeros((128, 128), dtype=np.uint8)
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                checker[i*32:(i+1)*32, j*32:(j+1)*32] = 220
            else:
                checker[i*32:(i+1)*32, j*32:(j+1)*32] = 30
    res_check = studio_app.model_colorize(model, checker)
    assert res_check.shape == (128, 128, 3)
    assert res_check.mean() > 30.0
    assert res_check.max() > 100

    # 5. Float32 image [0.0, 1.0]
    f32_img = (checker / 255.0).astype(np.float32)
    res_f32 = studio_app.model_colorize(model, f32_img)
    assert res_f32.shape == (128, 128, 3)
    assert res_f32.dtype == np.uint8
    assert res_f32.mean() > 30.0


def test_video_colorization_pipeline():
    model = load_model()
    with tempfile.TemporaryDirectory() as td:
        in_mp4 = str(Path(td) / "input.mp4")
        out_mp4 = str(Path(td) / "output.mp4")

        writer = cv2.VideoWriter(in_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (96, 64))
        for i in range(6):
            frame = np.full((64, 96, 3), 60 + i * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        progress_calls = []
        colorize_video(model, in_mp4, out_mp4, progress_callback=lambda d, t: progress_calls.append((d, t)))
        assert len(progress_calls) == 6
        assert progress_calls[-1] == (6, 6)
        assert Path(out_mp4).exists() and Path(out_mp4).stat().st_size > 0

        cap = cv2.VideoCapture(out_mp4)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        assert len(frames) == 6
        assert frames[0].shape == (64, 96, 3)
        assert 40.0 < float(frames[0].mean()) < 220.0


def test_webcam_processor_pipeline():
    model = load_model()
    cascade = studio_app.get_face_cascade()
    proc = studio_app.ColorizationProcessor(model, cascade, resolution=128, wipe=50, face=True, clamp=True)
    bgr = np.full((120, 160, 3), 128, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
    out_frame = proc.recv(frame)
    out_bgr = out_frame.to_ndarray(format="bgr24")
    assert out_bgr.shape == (120, 160, 3)
    assert proc.latest_color is not None


def test_batch_zip_pipeline():
    model = load_model()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(2):
            gray = np.full((64, 64), 90 + i * 40, dtype=np.uint8)
            rgb = studio_app.model_colorize(model, gray)
            ok, enc = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            assert ok
            zf.writestr(f"img_{i}-colorized.png", enc.tobytes())
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf:
        assert zf.namelist() == ["img_0-colorized.png", "img_1-colorized.png"]
