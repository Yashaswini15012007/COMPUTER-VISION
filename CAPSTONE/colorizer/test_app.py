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

    final_rgb = studio_app.apply_postprocessing(
        raw_rgb, restored, preset="Balanced", saturation_clamp=True, face_protection=True, strength=100
    )
    assert final_rgb.shape == (96, 128, 3)
    assert final_rgb.dtype == np.uint8


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
