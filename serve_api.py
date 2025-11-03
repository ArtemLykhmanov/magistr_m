from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, Union, List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ------------------------------
# Шляхи до артефактів
# ------------------------------
ARTIFACTS_DIR = os.environ.get("IDS_ARTIFACTS", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features.json")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "manifest.json")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "tuned_threshold.json")

# ------------------------------
# Завантаження моделі та метаданих
# ------------------------------
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# meta/manifest (не обов’язково)
meta: Dict[str, Any] = {}
if os.path.exists(MANIFEST_PATH):
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read manifest.json: {e}")

# features.json
num_cols: List[str] = []
cat_cols: List[str] = []
features: List[str] = []

def _load_features() -> None:
    """Гнучко зчитує списки ознак із features.json або manifest.json."""
    global num_cols, cat_cols, features

    # Спроба 1: features.json
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feat_obj = json.load(f)
        # Підтримка декількох варіантів ключів
        for cand in ["num_cols", "num", "numeric", "numeric_cols"]:
            if cand in feat_obj and isinstance(feat_obj[cand], list):
                num_cols = feat_obj[cand]
                break
        for cand in ["cat_cols", "cat", "categorical", "categorical_cols"]:
            if cand in feat_obj and isinstance(feat_obj[cand], list):
                cat_cols = feat_obj[cand]
                break
        if not num_cols and not cat_cols and "features" in feat_obj:
            features = list(feat_obj["features"])
        else:
            features = list(num_cols) + list(cat_cols)
        if not features:
            raise ValueError("features.json не містить переліку ознак.")
        print(f"[INFO] Loaded features from features.json: num={len(num_cols)} cat={len(cat_cols)} total={len(features)}")
        return

    # Спроба 2: manifest.json
    if meta and "features" in meta and isinstance(meta["features"], list):
        features = list(meta["features"])
        print(f"[INFO] Loaded features from manifest.json: total={len(features)}")
        return

    raise RuntimeError("Не знайшов списку ознак у artifacts/features.json або artifacts/manifest.json")

_load_features()

# ------------------------------
# Завантаження порога
# ------------------------------
DEFAULT_THRESHOLD = 0.5
THRESHOLD = float(os.environ.get("IDS_THRESHOLD", DEFAULT_THRESHOLD))

if os.path.exists(THRESHOLD_PATH):
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            thr_obj = json.load(f)
        if isinstance(thr_obj, dict) and "threshold" in thr_obj:
            THRESHOLD = float(thr_obj["threshold"])
            print(f"[INFO] Loaded tuned THRESHOLD={THRESHOLD} from {THRESHOLD_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to read {THRESHOLD_PATH}: {e}")

# ------------------------------
# Pydantic-схеми запиту/відповіді
# ------------------------------
class PredictIn(BaseModel):
    values: Dict[str, Union[int, float, str]]  # довіряємо пайплайну привести типи

class PredictOut(BaseModel):
    prediction: int
    probability: float
    model: str
    threshold: float
    missing_features: Optional[List[str]] = None
    extra_features: Optional[List[str]] = None

# ------------------------------
# FastAPI
# ------------------------------
app = FastAPI(title="Traffic IDS API", version="1.0.0")

@app.get("/")
def root():
    return {
        "name": "Traffic IDS API",
        "status": "ok",
        "endpoints": ["/predict", "/meta", "/features", "/healthz"],
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/features")
def get_features():
    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "features": features,
        "count": len(features),
    }

@app.get("/meta")
def get_meta():
    return {
        "best_model": meta.get("best_model", "?"),
        "train_rows": meta.get("train_rows"),
        "val_rows": meta.get("val_rows"),
        "test_rows": meta.get("test_rows"),
        "cv": meta.get("cv"),
        "notes": meta.get("notes"),
        "threshold_default": THRESHOLD,
    }

def _coerce_value(v: Any) -> Any:
    """Спроба привести значення: числові рядки -> float, інакше залишаємо як є."""
    if v is None:
        return np.nan
    # якщо вже число — повертаємо
    if isinstance(v, (int, float, np.number)):
        return v
    # якщо рядок — пробуємо конвертувати в float
    if isinstance(v, str):
        vs = v.strip()
        if vs == "" or vs.lower() in {"nan", "none", "null"}:
            return np.nan
        try:
            return float(vs)
        except Exception:
            return v  # категоріальна ознака або несумісне значення
    return v

def _predict_proba_safe(X: pd.DataFrame) -> float:
    """Отримати ймовірність позитивного класу для одного запису."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0, 1])
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # логістична сигмоїда як наближення
        return float(1.0 / (1.0 + np.exp(-s[0])))
    # fallback: якщо модель не дає score/proba
    y = model.predict(X)
    return float(y[0])

@app.post("/predict", response_model=PredictOut)
async def predict(req: Request):
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict) or "values" not in body:
        raise HTTPException(status_code=400, detail="Expected JSON with 'values' field")

    data = body["values"]
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="'values' must be an object/map")

    # Порог: query-параметр має пріоритет
    thr = req.query_params.get("thr", None)
    try:
        threshold = float(thr) if thr is not None else THRESHOLD
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid 'thr' query param; must be a float")

    # Побудова одного рядка з колонами у правильному порядку
    row: Dict[str, Any] = {}
    missing: List[str] = []
    for c in features:
        if c in data:
            row[c] = _coerce_value(data[c])
        else:
            row[c] = np.nan  # SimpleImputer у пайплайні має це обробити
            missing.append(c)

    # Додаткові (невідомі) ключі у запиті — ігноруємо, але повернемо у відповіді
    extra = [k for k in data.keys() if k not in features]

    X = pd.DataFrame([row], columns=features)

    try:
        proba = _predict_proba_safe(X)
    except Exception as e:
        # Корисна діагностика, якщо раптом колонки/типи не зійшлись
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    pred = int(proba >= threshold)
    model_name = meta.get("best_model", getattr(model, "__class__", type("X", (), {})).__name__)

    return PredictOut(
        prediction=pred,
        probability=float(proba),
        model=model_name,
        threshold=float(threshold),
        missing_features=missing or None,
        extra_features=extra or None,
    )

if __name__ == "__main__":
    host = os.environ.get("IDS_HOST", "127.0.0.1")
    port = int(os.environ.get("IDS_PORT", "8000"))
    print(f"[INFO] Starting API on http://{host}:{port} | model={MODEL_PATH}")
    uvicorn.run("serve_api:app", host=host, port=port, reload=False)