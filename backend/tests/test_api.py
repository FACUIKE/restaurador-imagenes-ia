from fastapi.testclient import TestClient
from app.main import app

# Creamos un cliente de pruebas que simula ser el frontend
client = TestClient(app)

def test_api_process_sin_filtros_devuelve_original():
    # Simulamos enviar un píxel blanco en Base64
    pixel_blanco_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
    
    payload = {
        "image_b64": pixel_blanco_b64,
        "categoria": "texto",
        "sub_modo": "", # Sin filtro
        "intensidad": 0
    }
    
    # Hacemos la petición POST a nuestra API
    response = client.post("/api/process", json=payload)
    
    # 1. Medimos que el servidor responda OK (200)
    assert response.status_code == 200
    
    data = response.json()
    
    # 2. Medimos que la respuesta contenga la imagen restaurada
    assert "restaurada_b64" in data
    
    # 3. Medimos que al no aplicar filtros, el Fidelity Score sea perfecto (1.0)
    assert data["metricas"]["fid"] == 1.0
    assert data["metricas"]["hal"] == 0.0