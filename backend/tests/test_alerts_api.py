import pytest
from fastapi.testclient import TestClient

# Importamos tu app de FastAPI
from src.api.server import app

client = TestClient(app)

def test_alert_complete_lifecycle():
    """
    Prueba exhaustiva del ciclo de vida de una alerta (Máquina de Estados).
    Cubre: Creación, Listado, Conteo, Lectura, y Cierre.
    """
    # ---------------------------------------------------------
    # 1. Crear una alerta (POST /api/alerts/)
    # ---------------------------------------------------------
    payload = {
        "incident_id": None, # Usamos null para no depender de incidentes por ahora
        "message": "Arma detectada en Lobby principal (Test Automatizado)"
    }
    res_create = client.post("/api/alerts/", json=payload)
    
    # En REST, una creación exitosa suele ser 201 Created (o 200)
    assert res_create.status_code in [200, 201], f"Fallo al crear: {res_create.text}"
    
    alert_data = res_create.json()
    assert "id" in alert_data
    assert alert_data["status"] == "UNREAD"
    assert alert_data["resolved_at"] is None
    
    alert_id = alert_data["id"]

    # ---------------------------------------------------------
    # 2. Contar alertas UNREAD (GET /api/alerts/count)
    # ---------------------------------------------------------
    res_count = client.get("/api/alerts/count?status=UNREAD")
    assert res_count.status_code == 200
    assert res_count.json()["count"] >= 1

    # ---------------------------------------------------------
    # 3. Obtener la alerta individual (GET /api/alerts/{id})
    # ---------------------------------------------------------
    res_get = client.get(f"/api/alerts/{alert_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == alert_id

    # ---------------------------------------------------------
    # 4. Reconocer la alerta (PATCH /api/alerts/{id}/status)
    # ---------------------------------------------------------
    res_ack = client.patch(f"/api/alerts/{alert_id}/status", json={"status": "ACKNOWLEDGED"})
    assert res_ack.status_code == 200
    assert res_ack.json()["status"] == "ACKNOWLEDGED"

    # ---------------------------------------------------------
    # 5. Resolver la alerta (PATCH /api/alerts/{id}/status)
    # ---------------------------------------------------------
    res_resolve = client.patch(f"/api/alerts/{alert_id}/status", json={"status": "RESOLVED"})
    assert res_resolve.status_code == 200
    
    resolved_data = res_resolve.json()
    assert resolved_data["status"] == "RESOLVED"
    # Verificar que el sistema estampó la fecha de resolución automáticamente
    assert resolved_data["resolved_at"] is not None 

    # ---------------------------------------------------------
    # 6. EDGE CASE: Intentar modificar una alerta ya resuelta
    # ---------------------------------------------------------
    res_conflict = client.patch(f"/api/alerts/{alert_id}/status", json={"status": "ACKNOWLEDGED"})
    # Verificamos que tu compañero implementó correctamente la protección (Error 409)
    assert res_conflict.status_code == 409, "El sistema permitió modificar una alerta cerrada!"


def test_alert_invalid_incident_id():
    """
    Prueba el caso de error que descubrimos en el Code Review.
    """
    payload = {
        "incident_id": "00000000-0000-0000-0000-000000000000", # UUID falso
        "message": "Esto debería fallar"
    }
    res = client.post("/api/alerts/", json=payload)
    
    # Debería devolver un 400 Bad Request o 404 Not Found, NO un 500
    assert res.status_code in [400, 404], f"Esperábamos un error controlado, pero recibimos: {res.status_code}"