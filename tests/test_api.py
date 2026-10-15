import pytest
from fastapi.testclient import TestClient
from maint_audit.api import app

client = TestClient(app)


def test_health_endpoint():
    with TestClient(app) as c:
        response = c.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["threshold"] == 0.515


def test_predict_single_normal():
    with TestClient(app) as c:
        payload = {
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 50.0,
            "Type": "M",
        }
        response = c.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["risk_flag"] is False
        assert data["rule_any"] is False
        assert 0.0 <= data["ml_risk"] <= 1.0


def test_predict_single_hdf_rule():
    with TestClient(app) as c:
        payload = {
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 308.0,  # diff = 8.0 < 8.6
            "Rotational speed [rpm]": 1300.0,  # < 1380
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 50.0,
            "Type": "M",
        }
        response = c.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["rule_hdf"] is True
        assert data["risk_flag"] is True
        assert "HDF" in data["likely_mode"]


def test_predict_batch():
    with TestClient(app) as c:
        records = [
            {
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 50.0,
                "Type": "M",
                "UDI": 1,
            },
            {
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1000.0,
                "Torque [Nm]": 5.0,  # power ~524 W -> PWF
                "Tool wear [min]": 50.0,
                "Type": "L",
                "UDI": 2,
            },
        ]
        response = c.post("/predict/batch", json=records)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["risk_flag"] is False
        assert data[1]["rule_pwf"] is True
        assert data[1]["risk_flag"] is True


def test_predict_invalid_input_schema():
    with TestClient(app) as c:
        payload = {
            "Air temperature [K]": 150.0,  # Below plausible min of 250
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 50.0,
            "Type": "M",
        }
        response = c.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity
