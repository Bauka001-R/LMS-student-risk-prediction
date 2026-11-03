import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import FEATURE_NAMES, TARGET, risk_level  # noqa: E402
from generate_dataset import generate  # noqa: E402


def test_dataset_shape_and_determinism():
    a, b = generate(seed=1), generate(seed=1)
    assert a.shape[0] == 247
    assert all(f in a.columns for f in FEATURE_NAMES) and len(FEATURE_NAMES) == 22
    assert a.equals(b)
    assert 0.25 < a[TARGET].mean() < 0.33
    assert a[FEATURE_NAMES].notna().all().all()


def test_risk_level_bands():
    assert risk_level(0.1) == "low"
    assert risk_level(0.5) == "medium"
    assert risk_level(0.9) == "high"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    data, model, reports = d / "students.csv", d / "model.joblib", d / "reports"
    subprocess.run([sys.executable, "generate_dataset.py", "--out", str(data)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "train_model.py", "--data", str(data), "--model-out", str(model),
                    "--reports", str(reports)], cwd=ROOT, check=True)
    return data, model, reports


def test_training_outputs(trained):
    data, model, reports = trained
    assert model.exists() and (reports / "metrics.json").exists()
    import json
    m = json.loads((reports / "metrics.json").read_text())
    assert set(m["models"]) == {"LogReg", "RandomForest", "GradBoost", "LightGBM", "MLP"}
    assert m["models"]["LightGBM"]["roc_auc"] > 0.85


def test_api(trained, monkeypatch):
    data, model, _ = trained
    monkeypatch.setenv("LMS_MODEL_PATH", str(model))
    monkeypatch.setenv("LMS_DATA_PATH", str(data))
    import prediction_api
    df = pd.read_csv(data)
    sid, grp = df.student_id.iloc[0], df.group.iloc[0]
    with TestClient(prediction_api.app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/model/info").json()["n_features"] == 22

        r = c.post(f"/predict/{sid}").json()
        assert 0 <= r["risk_probability"] <= 1 and r["risk_level"] in {"low", "medium", "high"}

        body = df[FEATURE_NAMES].iloc[0].to_dict()
        r2 = c.post("/predict", json=body).json()
        assert r2["risk_probability"] == pytest.approx(r["risk_probability"], abs=1e-3)

        batch = c.post("/predict/batch", json={"group": grp}).json()
        assert len(batch) == (df.group == grp).sum()

        ex = c.get(f"/explain/{sid}?top=3").json()
        assert len(ex["top_contributions"]) == 3

        assert c.post("/predict/NOPE").status_code == 404
        assert c.post("/predict/batch", json={}).status_code == 422
