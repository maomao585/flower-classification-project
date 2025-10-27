import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_image_bytes(size=(128, 128), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_predict_valid_small_image():
    buf = make_image_bytes((16, 16))
    files = {"file": ("small.png", buf, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    assert "label" in r.json() and "score" in r.json()


def test_predict_missing_file_field():
    r = client.post("/predict", files={})
    assert r.status_code in (400, 422)


def test_predict_non_image():
    buf = io.BytesIO(b"not an image content")
    files = {"file": ("text.txt", buf, "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code in (400, 422)


def test_predict_large_image():
    buf = make_image_bytes((1024, 768))
    files = {"file": ("large.png", buf, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
