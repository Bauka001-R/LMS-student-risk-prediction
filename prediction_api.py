"""FastAPI prediction microservice (LightGBM risk model).

Endpoints
    GET  /health                      liveness probe
    GET  /model/info                  model version, training date, test metrics
    POST /predict                     risk for a feature vector sent in the request body
    POST /predict/{student_id}        risk for a student whose features are in the data table
    POST /predict/batch               risk for a list of students or a whole group
    GET  /explain/{student_id}        per-feature contributions (SHAP-style, from LightGBM)

Run:
    uvicorn prediction_api:app --reload
    # interactive docs: http://127.0.0.1:8000/docs

Environment variables:
    LMS_MODEL_PATH  (default models/lightgbm_risk.joblib)
    LMS_DATA_PATH   (default data/students.csv)
"""
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import FEATURE_NAMES, FEATURES, risk_level

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("LMS_MODEL_PATH", "models/lightgbm_risk.joblib")
    data_path = os.getenv("LMS_DATA_PATH", "data/students.csv")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}. Run generate_dataset.py and train_model.py first.")
    STATE["bundle"] = joblib.load(model_path)
    STATE["students"] = (pd.read_csv(data_path).set_index("student_id")
                         if os.path.exists(data_path) else pd.DataFrame())
    yield
    STATE.clear()


app = FastAPI(title="LMS Student Risk Prediction", version="1.0.0", lifespan=lifespan)

# Request model with the 22 features (built dynamically from features.py)
FeatureVector = type(
    "FeatureVector", (BaseModel,),
    {"__annotations__": {name: float for name in FEATURE_NAMES},
     **{name: Field(..., description=desc) for name, desc in FEATURES}},
)


class Prediction(BaseModel):
    student_id: Optional[str] = None
    risk_probability: float
    risk_level: str = Field(description="low / medium / high")
    is_at_risk: bool


class BatchRequest(BaseModel):
    student_ids: Optional[List[str]] = None
    group: Optional[str] = None


def _predict(frame: pd.DataFrame) -> List[float]:
    model = STATE["bundle"]["model"]
    return [float(p) for p in model.predict_proba(frame[FEATURE_NAMES])[:, 1]]


def _to_prediction(student_id: Optional[str], p: float) -> Prediction:
    return Prediction(student_id=student_id, risk_probability=round(p, 4),
                      risk_level=risk_level(p), is_at_risk=p >= 0.5)


def _student_row(student_id: str) -> pd.DataFrame:
    students = STATE["students"]
    if student_id not in students.index:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    return students.loc[[student_id]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    b = STATE["bundle"]
    return {"model": "LightGBM", "trained_at": b["trained_at"], "n_features": len(b["features"]),
            "features": b["features"], "test_metrics": b["metrics"], "versions": b["versions"]}


@app.post("/predict", response_model=Prediction)
def predict(vector: FeatureVector):  # type: ignore[valid-type]
    frame = pd.DataFrame([vector.model_dump()])
    return _to_prediction(None, _predict(frame)[0])


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(req: BatchRequest):
    students = STATE["students"]
    if req.student_ids:
        missing = [s for s in req.student_ids if s not in students.index]
        if missing:
            raise HTTPException(status_code=404, detail=f"Unknown students: {missing}")
        frame = students.loc[req.student_ids]
    elif req.group:
        frame = students[students["group"] == req.group]
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"Group {req.group} not found")
    else:
        raise HTTPException(status_code=422, detail="Provide student_ids or group")
    return [_to_prediction(sid, p) for sid, p in zip(frame.index, _predict(frame))]


@app.post("/predict/{student_id}", response_model=Prediction)
def predict_student(student_id: str):
    row = _student_row(student_id)
    return _to_prediction(student_id, _predict(row)[0])


@app.get("/explain/{student_id}")
def explain(student_id: str, top: int = 5):
    """Feature contributions in log-odds (LightGBM native SHAP values via pred_contrib)."""
    row = _student_row(student_id)
    booster = STATE["bundle"]["model"].booster_
    contrib = booster.predict(row[FEATURE_NAMES], pred_contrib=True)[0]
    items = sorted(
        ({"feature": f, "value": float(row[f].iloc[0]), "contribution": float(c)}
         for f, c in zip(FEATURE_NAMES, contrib[:-1])),
        key=lambda x: abs(x["contribution"]), reverse=True)[:top]
    p = _predict(row)[0]
    return {"student_id": student_id, "risk_probability": round(p, 4),
            "base_value": float(contrib[-1]), "top_contributions": items}
