import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from src.web.app import app


def test_index_returns_browser_ui():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "HH + KCa Web Tuner" in response.text
    assert "/api/simulate" in response.text


def test_simulate_endpoint_returns_series():
    client = TestClient(app)

    response = client.post(
        "/api/simulate",
        json={
            "Kd_KCa": 3e-4,
            "v_rel": 0.01,
            "v_serca": 0.02,
            "t_end_ms": 1000,
            "dt_ms": 20,
            "max_points": 500,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["success"] is True
    assert data["summary"]["points"] > 0
    assert set(data["series"]) == {"t_s", "Vm_mV", "ci_mM"}
    assert len(data["series"]["t_s"]) == len(data["series"]["Vm_mV"])
    assert len(data["series"]["t_s"]) == len(data["series"]["ci_mM"])
