"""FastAPI prediction backend for aircraft-engine Remaining Useful Life.

Run locally:  uvicorn api.main:app --port 8000
Docs:         http://localhost:8000/docs     Metrics: http://localhost:8000/metrics
"""
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from pydantic import BaseModel, Field

from src.config import load_config
from src.drift import drift_report
from src.predict import InputError, RULPredictor

log = logging.getLogger("rul-api")
CFG = load_config()["serving"]

# ---- Operational metrics (is the service healthy?) ----
REQUESTS = Counter("rul_http_requests_total", "HTTP requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("rul_http_request_duration_seconds", "HTTP request latency", ["endpoint"])
PREDICTIONS = Counter("rul_predictions_total", "Successful RUL predictions")
ERRORS = Counter("rul_prediction_errors_total", "Failed prediction requests", ["type"])
PRED_LATENCY = Histogram("rul_prediction_latency_seconds", "Feature engineering + model inference time",
                         buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
MODEL_LOADED = Gauge("rul_model_loaded", "1 if a trained model is loaded")
MODEL_INFO = Info("rul_model", "Loaded model metadata")
# ---- Model output / input-data metrics (what is the model seeing and saying?) ----
PREDICTED_RUL = Histogram("rul_predicted_cycles", "Predicted RUL distribution", buckets=(10, 25, 50, 75, 100, 125))
INPUT_PSI = Gauge("rul_input_psi", "PSI of recent latest-cycle inputs vs training distribution", ["sensor"])
OUT_OF_RANGE = Counter("rul_input_out_of_range_total", "Latest-cycle values outside training range", ["sensor"])
# ---- Model-performance metrics (needs ground truth reported back after the fact) ----
FEEDBACK = Counter("rul_feedback_total", "Ground-truth RUL labels received")
FEEDBACK_ERR = Histogram("rul_feedback_abs_error_cycles", "|predicted - min(actual, cap)| RUL",
                         buckets=(5, 10, 15, 20, 30, 50, 75, 100))
FEEDBACK_MAE = Gauge("rul_feedback_rolling_mae", "MAE over the last 200 labelled predictions")

STATE: dict = {"predictor": None, "error": None, "inputs": deque(maxlen=CFG["drift_buffer_size"]),
               "abs_errors": deque(maxlen=200)}


def load_model():
    try:
        p = RULPredictor(os.getenv("RUL_MODEL_PATH") or None)
        STATE.update(predictor=p, error=None)
        MODEL_LOADED.set(1)
        MODEL_INFO.info({k: str(p.meta[k]) for k in ("model_name", "model_version", "subset")})
        log.info("Loaded model %s v%s", p.meta["model_name"], p.meta["model_version"])
    except Exception as e:  # keep serving /health and /metrics so the failure is visible
        STATE.update(predictor=None, error=str(e))
        MODEL_LOADED.set(0)
        log.error("Model not loaded: %s", e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Aircraft Engine RUL API", version="1.0.0", lifespan=lifespan,
              description="Predicts Remaining Useful Life (cycles) from NASA C-MAPSS style sensor histories.")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    endpoint = request.url.path
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if endpoint != "/metrics":
            REQUESTS.labels(endpoint, request.method, str(status)).inc()
            REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - t0)


@app.exception_handler(RequestValidationError)
async def schema_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/predict"):
        ERRORS.labels("schema").inc()
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


class PredictRequest(BaseModel):
    engine_id: str | None = Field(None, examples=["engine-42"])
    records: list[dict[str, float]] = Field(
        ..., min_length=1, max_length=CFG["max_records"],
        description="One dict per cycle: 'cycle' plus sensor columns s1..s21 (op settings optional).")


class PredictResponse(BaseModel):
    engine_id: str | None
    predicted_rul: float
    last_cycle: int
    cycles_observed: int
    model_name: str
    model_version: str
    health_status: str
    latency_ms: float


class FeedbackRequest(BaseModel):
    engine_id: str | None = None
    predicted_rul: float = Field(..., ge=0)
    actual_rul: float = Field(..., ge=0)


def require_model() -> RULPredictor:
    if STATE["predictor"] is None:
        ERRORS.labels("model_unavailable").inc()
        raise HTTPException(503, f"Model not loaded: {STATE['error']}")
    return STATE["predictor"]


def health_band(rul: float) -> str:
    return "CRITICAL" if rul <= 25 else "WARNING" if rul <= 75 else "HEALTHY"


def update_drift(p: RULPredictor, df: pd.DataFrame):
    # Only the latest cycle (the state being predicted) enters the buffer. Buffering whole
    # histories would over-weight repeated early-life cycles and look like drift.
    latest = df.sort_values("cycle").iloc[-1]
    for s, ref in p.drift_reference.items():
        if not ref["min"] <= latest[s] <= ref["max"]:
            OUT_OF_RANGE.labels(s).inc()
    STATE["inputs"].append(latest[list(p.drift_reference)].to_dict())
    if len(STATE["inputs"]) >= 50:  # PSI on tiny samples is meaningless
        for _, r in drift_report(p.drift_reference, pd.DataFrame(STATE["inputs"])).iterrows():
            INPUT_PSI.labels(r["sensor"]).set(r["psi"])


@app.get("/")
def root():
    return {"service": "aircraft-rul-api", "docs": "/docs", "health": "/health", "metrics": "/metrics"}


@app.get("/health")
def health():
    if STATE["predictor"] is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "model_loaded": False,
                                                      "error": STATE["error"]})
    return {"status": "ok", "model_loaded": True, "model_version": STATE["predictor"].meta["model_version"]}


@app.get("/model/info")
def model_info():
    p = require_model()
    return {**p.meta, "required_columns": ["cycle", *p.sensors]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p = require_model()
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame(req.records)
        out = p.predict_history(df)
    except InputError as e:
        ERRORS.labels("validation").inc()
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        ERRORS.labels("internal").inc()
        log.exception("Prediction failed")
        raise HTTPException(500, "Internal prediction error") from e
    finally:
        PRED_LATENCY.observe(time.perf_counter() - t0)
    PREDICTIONS.inc()
    PREDICTED_RUL.observe(out["predicted_rul"])
    update_drift(p, df)
    return {"engine_id": req.engine_id, **out, "health_status": health_band(out["predicted_rul"]),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Report the true RUL once it is known (e.g. after maintenance) -> live model-error metrics.

    The model is trained to saturate at rul_cap ("healthy"), so the error is measured against
    min(actual, cap); the raw error is returned too.
    """
    cap = STATE["predictor"].rul_cap if STATE["predictor"] else float("inf")
    err = abs(req.predicted_rul - min(req.actual_rul, cap))
    FEEDBACK.inc()
    FEEDBACK_ERR.observe(err)
    STATE["abs_errors"].append(err)
    mae = sum(STATE["abs_errors"]) / len(STATE["abs_errors"])
    FEEDBACK_MAE.set(mae)
    return {"abs_error": err, "raw_abs_error": abs(req.predicted_rul - req.actual_rul), "rolling_mae": mae,
            "n": len(STATE["abs_errors"])}


@app.get("/drift")
def drift():
    p = require_model()
    if len(STATE["inputs"]) < 50:
        return {"rows_buffered": len(STATE["inputs"]), "sensors": [], "note": "Need >= 50 predictions"}
    rep = drift_report(p.drift_reference, pd.DataFrame(STATE["inputs"]))
    rep["status"] = rep["status"].astype(str)
    return {"rows_buffered": len(STATE["inputs"]), "sensors": rep.to_dict("records")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
