from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent / "app.py")


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
