import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def client(predictor):  # predictor fixture skips if no model
    with TestClient(app) as c:
        yield c


def payload(df):
    return {"engine_id": "test-1", "records": df.drop(columns=["unit"]).to_dict("records")}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_matches_predictor(client, predictor, test_engine_history):
    r = client.post("/predict", json=payload(test_engine_history))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_rul"] == predictor.predict_history(test_engine_history)["predicted_rul"]
    assert body["health_status"] in {"HEALTHY", "WARNING", "CRITICAL"}


def test_predict_missing_sensor_is_400(client, test_engine_history):
    r = client.post("/predict", json=payload(test_engine_history.drop(columns=["s2"])))
    assert r.status_code == 400 and "Missing" in r.json()["detail"]


def test_predict_bad_schema_is_422(client):
    assert client.post("/predict", json={"records": []}).status_code == 422
    assert client.post("/predict", json={"records": [{"cycle": "x"}]}).status_code == 422


def test_feedback_and_metrics(client, test_engine_history):
    client.post("/feedback", json={"predicted_rul": 100, "actual_rul": 90})
    text = client.get("/metrics").text
    for name in ("rul_predictions_total", "rul_prediction_errors_total", "rul_prediction_latency_seconds_bucket",
                 "rul_http_requests_total", "rul_feedback_rolling_mae", "rul_model_loaded 1.0"):
        assert name in text
    assert 'rul_prediction_errors_total{type="validation"}' in text
