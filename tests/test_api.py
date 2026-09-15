import os
import dotenv
loaded = dotenv.load_dotenv()

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used-for-health-check")

from fastapi.testclient import TestClient

from reanmath.api import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_frontend_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "រៀនគណិត" in res.text
