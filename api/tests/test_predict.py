import os
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app
import asyncio
import onnxruntime as ort
from api.main import app, ml_model, MODEL_PATH # Importe ml_model et le path

BASE_URL = "http://test"

@pytest.fixture(scope="session", autouse=True)
def load_model_for_tests():
    """
    Force le chargement du modèle dans le dictionnaire ml_model
    avant de lancer les tests.
    """
    if "session" not in ml_model:
        # On vérifie si le modèle existe par rapport à la racine
        if not os.path.exists(MODEL_PATH):
            pytest.fail(f"Modèle introuvable à : {os.path.abspath(MODEL_PATH)}")
        
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        ml_model["session"] = session
        ml_model["input_name"] = session.get_inputs()[0].name
        print("\n[TEST SETUP] Modèle chargé manuellement pour Pytest")

@pytest.fixture
async def client():
    # Utilisation du transport ASGI
    async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as ac:
        yield ac

# @pytest.fixture
# async def client():
#     async with AsyncClient(app=app, base_url=BASE_URL) as ac:
#         yield ac

@pytest.mark.anyio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.anyio
async def test_predict_good_image(client):
    # Load the good test image
    img_path = os.path.join(os.path.dirname(__file__), "good_test.png")
    with open(img_path, "rb") as f:
        files = {"file": ("good_test.png", f, "image/png")}
        response = await client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["defective"] == False
    assert data["confidence"] == 0.0
    assert data["defect_type"] is None
    assert data["bbox"] == []

@pytest.mark.anyio
async def test_predict_defective_image(client):
    img_path = os.path.join(os.path.dirname(__file__), "defect_test.png")
    with open(img_path, "rb") as f:
        files = {"file": ("defect_test.png", f, "image/png")}
        response = await client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["defective"] == True
    assert data["confidence"] > 0.5
    assert data["defect_type"] == "defect"
    assert len(data["bbox"]) == 4
