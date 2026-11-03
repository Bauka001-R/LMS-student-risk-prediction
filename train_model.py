"""Train and compare the five algorithms from the thesis and save the production model.

Algorithms and hyper-parameters follow the thesis (section 2.3):
  Logistic Regression (C=1.0, L2), Random Forest (500 trees, depth 20),
  Gradient Boosting (300, lr 0.05, depth 6), LightGBM (300, lr 0.05, 31 leaves),
  MLP (128-64, ReLU, Adam). Class imbalance is handled with balanced class weights.
LightGBM is the production model (best speed/quality trade-off), as in the thesis.

Usage:
    python generate_dataset.py
    python train_model.py
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from features import FEATURE_NAMES, SEED, TARGET  # noqa: E402

TEST_SIZE = 75  # 75 test students, as in the thesis (247 = 172 train + 75 test)


class BalancedGradientBoosting(GradientBoostingClassifier):
    """GradientBoostingClassifier has no class_weight option, so use balanced sample weights."""

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def build_models() -> dict:
    return {
        "LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced",  # L2 (default)
                               max_iter=1000, random_state=SEED)),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=20, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "GradBoost": BalancedGradientBoosting(
            n_estimators=300, learning_rate=0.05, max_depth=6, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            colsample_bytree=0.9, subsample=0.8, subsample_freq=1,
            class_weight="balanced", random_state=SEED, verbose=-1),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                          max_iter=200, random_state=SEED)),
    }


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/students.csv")
    ap.add_argument("--model-out", default="models/lightgbm_risk.joblib")
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    X, y = df[FEATURE_NAMES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"train={len(X_train)} test={len(X_test)} at-risk share={y.mean():.1%}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in build_models().items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        m["cv_f1_mean"], m["cv_f1_std"] = cv_f1.mean(), cv_f1.std()
        results[name], fitted[name] = m, model
        print(f"{name:13s} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f} Acc={m['accuracy']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} | CV F1={m['cv_f1_mean']:.3f}")

    # confusion matrix of the production model
    lgbm = fitted["LightGBM"]
    tn, fp, fn, tp = confusion_matrix(y_test, lgbm.predict(X_test)).ravel()

    reports = Path(a.reports)
    reports.mkdir(parents=True, exist_ok=True)

    # --- plots ---
    names = list(results)
    f1s = [results[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["#e03131" if n == "LightGBM" else "#91c4f8" for n in names]
    ax.bar(names, f1s, color=colors)
    ax.set_ylim(min(f1s) - 0.05, 1.0)
    ax.set_ylabel("F1-score (test)")
    ax.set_title("Model comparison")
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(reports / "model_comparison.png", dpi=150)
    plt.close(fig)

    imp = pd.Series(lgbm.booster_.feature_importance("gain"), index=FEATURE_NAMES).nlargest(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp.index, imp.values, color="#2563eb")
    ax.set_title("Top-10 features (LightGBM, gain)")
    fig.tight_layout()
    fig.savefig(reports / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --- save production model + metadata ---
    metrics = {
        "dataset": {"n_students": len(df), "n_train": len(X_train), "n_test": len(X_test),
                    "at_risk_share": float(y.mean()), "n_features": len(FEATURE_NAMES)},
        "models": results,
        "lightgbm_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    bundle = {
        "model": lgbm,
        "features": FEATURE_NAMES,
        "metrics": results["LightGBM"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    out = Path(a.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"Saved model -> {out}; reports -> {reports}/")


if __name__ == "__main__":
    main()
