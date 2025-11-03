# -*- coding: utf-8 -*-

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # рендер у файл без GUI
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import joblib

ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
DEMO_PRED = ART / "demo_predictions.csv"
MODEL_PATH = ART / "model.joblib"
FEAT_PATH  = ART / "features.json"

# 0) Перевірки
if not DEMO_PRED.exists():
    raise SystemExit("Немає artifacts/demo_predictions.csv. Спочатку запусти demo_batch.py")
if not MODEL_PATH.exists() or not FEAT_PATH.exists():
    raise SystemExit("Немає model.joblib або features.json. Спочатку запусти train_baseline.py")

# 1) Завантажимо результати демо та оригінальний CSV для розширених графіків
dfp = pd.read_csv(DEMO_PRED)  # очікує колонки: true, pred, proba_attack
if not {"true","pred","proba_attack"}.issubset(dfp.columns):
    raise SystemExit("demo_predictions.csv не містить очікуваних колонок (true, pred, proba_attack)")

y_true = dfp["true"].values.astype(int)
y_score = dfp["proba_attack"].values.astype(float)
y_pred  = dfp["pred"].values.astype(int)

# знайдемо джерело даних (для розподілів і топ-категорій)
CSV_PATH = Path(os.getenv("IDS_CSV", "data/UNSW_NB15_training-set.csv"))
raw_df = None
if CSV_PATH.exists():
    try:
        raw_df = pd.read_csv(CSV_PATH)
    except Exception:
        raw_df = None

# 2) Нормалізована матриця помилок
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
plt.figure()
plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
plt.title("Confusion Matrix (normalized)")
plt.xlabel("Predicted"); plt.ylabel("True")
for (i, j), v in np.ndenumerate(cm_norm):
    plt.text(j, i, f"{v:.2f}", ha="center", va="center")
plt.tight_layout()
plt.savefig(ART / "confusion_matrix_norm.png", dpi=150); plt.close()

# 3) ROC крива
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--', alpha=0.5)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ART / "roc_curve.png", dpi=150); plt.close()

# 4) PR крива (перегенеруємо з demo_predictions.csv)
prec, rec, thr = precision_recall_curve(y_true, y_score)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ART / "pr_curve.png", dpi=150); plt.close()

# 5) F1 vs Threshold
ths = np.linspace(0.01, 0.99, 99)
f1s, pres, recs = [], [], []
for t in ths:
    p = (y_score >= t).astype(int)
    f1s.append(f1_score(y_true, p))
    pres.append(precision_score(y_true, p, zero_division=0))
    recs.append(recall_score(y_true, p))
plt.figure()
plt.plot(ths, f1s, label="F1")
plt.plot(ths, pres, label="Precision", alpha=0.8)
plt.plot(ths, recs, label="Recall", alpha=0.8)
plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("F1 / Precision / Recall vs Threshold")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ART / "f1_vs_threshold.png", dpi=150); plt.close()

# 6) Calibration curve
prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=15, strategy="quantile")
plt.figure()
plt.plot([0,1],[0,1],'--', alpha=0.5, label="Perfect")
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.xlabel("Predicted probability"); plt.ylabel("True frequency"); plt.title("Calibration curve")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(ART / "calibration_curve.png", dpi=150); plt.close()

# 7) Histograms of predicted probabilities
plt.figure()
plt.hist(y_score[y_true==0], bins=30, alpha=0.7, label="Benign (0)")
plt.hist(y_score[y_true==1], bins=30, alpha=0.7, label="Attack (1)")
plt.xlabel("Predicted probability"); plt.ylabel("Count"); plt.title("Probability histograms")
plt.legend(); plt.tight_layout()
plt.savefig(ART / "proba_hist.png", dpi=150); plt.close()

# 8) Важливість ознак
model = joblib.load(MODEL_PATH)
meta = json.load(open(FEAT_PATH, "r", encoding="utf-8"))
features = meta.get("features", [])
num_cols  = meta.get("num_cols", [])
cat_cols  = meta.get("cat_cols", [])

def get_feature_names_after_preprocessor(pipeline):
    # намагаємося отримати імена після колон-трансформера
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        steps = dict(pipeline.named_steps) if hasattr(pipeline, "named_steps") else {}
        # шукаємо ColumnTransformer
        pre = None
        for name, obj in steps.items():
            if obj.__class__.__name__ in ("ColumnTransformer",):
                pre = obj
                break
            # або знайдемо по типу
            if "sklearn.compose._column_transformer.ColumnTransformer" in str(type(obj)):
                pre = obj
                break
        if pre is None:
            # можливо, перший крок — препроцесор
            for st_name, st_obj in pipeline.steps:
                if "ColumnTransformer" in str(type(st_obj)):
                    pre = st_obj
                    break
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            names = pre.get_feature_names_out()
            return list(map(str, names))
    except Exception:
        pass
    return None

feat_names_after = get_feature_names_after_preprocessor(model)

# (а) Спроба використати .feature_importances_ (для RF/XGB/ETC)
est = None
try:
    # останній крок пайплайна — класифікатор
    est = model.steps[-1][1] if hasattr(model, "steps") else None
except Exception:
    est = None

if est is not None and hasattr(est, "feature_importances_"):
    importances = np.array(est.feature_importances_, dtype=float)
    if feat_names_after is None or len(feat_names_after) != len(importances):
        # fallback: номери ознак
        feat_labels = [f"f{i}" for i in range(len(importances))]
    else:
        feat_labels = feat_names_after
    # візьмемо топ-30
    idx = np.argsort(importances)[::-1][:30]
    plt.figure(figsize=(8,8))
    plt.barh(range(len(idx)), importances[idx][::-1])
    plt.yticks(range(len(idx)), [feat_labels[i] for i in idx][::-1], fontsize=8)
    plt.title("Feature importance (estimator)")
    plt.tight_layout()
    plt.savefig(ART / "feature_importance.png", dpi=150); plt.close()
else:
    # (б) Пермутаційна важливість як запасний варіант
    # візьмемо невелику підвибірку з оригінального CSV і прогонимо пайплайн цілком
    if raw_df is not None and all(c in raw_df.columns for c in features):
        from sklearn.inspection import permutation_importance
        # приготуємо X та y на невеликій підвибірці
        sub = raw_df.dropna().sample(min(2000, len(raw_df)), random_state=42)
        # знайдемо колонку мітки (як у демо)
        LABEL_CANDIDATES = ["label","Label","class","target"]
        label_col = next((c for c in LABEL_CANDIDATES if c in sub.columns), None)
        y_sub = None
        if label_col is not None:
            yraw = sub[label_col]
            if pd.api.types.is_numeric_dtype(yraw):
                y_sub = (yraw.astype(float) > 0).astype(int).values
            else:
                POS = {"malicious","attack","anomaly","botnet","dos","ddos","malware","infiltration","portscan","bruteforce","brute force","1","true"}
                y_sub = (yraw.astype(str).str.strip().str.lower().isin(POS)).astype(int).values
        # сформуємо X у такому ж порядку фіч
        X_sub = sub[features].copy()
        # пермутаційна важливість (може зайняти 10–30 сек на 2k рядках)
        r = permutation_importance(model, X_sub, y_sub, scoring="f1", n_repeats=5, random_state=0, n_jobs=-1)
        imps = r.importances_mean
        idx = np.argsort(imps)[::-1][:30]
        labs = np.array(features)[idx]
        plt.figure(figsize=(8,8))
        plt.barh(range(len(idx)), imps[idx][::-1])
        plt.yticks(range(len(idx)), labs[::-1], fontsize=8)
        plt.title("Permutation importance (top-30)")
        plt.tight_layout()
        plt.savefig(ART / "perm_importance.png", dpi=150); plt.close()

# 9) Розподіли ключових числових фіч (якщо є сирі дані)
if raw_df is not None:
    key_nums = [c for c in ["dur","spkts","dpkts","sbytes","dbytes","sload","dload"] if c in raw_df.columns]
    # приведемо y_true для того ж набору індексів, якщо можливо співставити
    # простіше: зробимо невеликий семпл і подивимось на маргінальні розподіли по мітці з raw_df
    # обчислимо y_raw заново:
    LABEL_CANDIDATES = ["label","Label","class","target"]
    label_col = next((c for c in LABEL_CANDIDATES if c in raw_df.columns), None)
    y_raw = None
    if label_col is not None:
        yr = raw_df[label_col]
        if pd.api.types.is_numeric_dtype(yr):
            y_raw = (yr.astype(float) > 0).astype(int)
        else:
            POS = {"malicious","attack","anomaly","botnet","dos","ddos","malware","infiltration","portscan","bruteforce","brute force","1","true"}
            y_raw = (yr.astype(str).str.strip().str.lower().isin(POS)).astype(int)
    if y_raw is not None:
        small = raw_df[[*key_nums]].copy()
        small["__y__"] = y_raw
        small = small.dropna().sample(min(10000, len(small)), random_state=7)
        for col in key_nums:
            plt.figure()
            plt.hist(small.loc[small["__y__"]==0, col], bins=40, alpha=0.7, label="0 (benign)")
            plt.hist(small.loc[small["__y__"]==1, col], bins=40, alpha=0.7, label="1 (attack)")
            plt.title(f"Distribution: {col}")
            plt.xlabel(col); plt.ylabel("Count"); plt.legend()
            plt.tight_layout()
            plt.savefig(ART / f"num_dists_{col}.png", dpi=150); plt.close()

    # топ категорії
    for cat in ["proto","service","state"]:
        if cat in raw_df.columns:
            vc = raw_df[cat].astype(str).fillna("NA").value_counts().head(15)
            plt.figure(figsize=(8,4))
            plt.bar(range(len(vc)), vc.values)
            plt.xticks(range(len(vc)), vc.index, rotation=45, ha="right", fontsize=8)
            plt.title(f"Top categories: {cat}")
            plt.tight_layout()
            plt.savefig(ART / f"top_categories_{cat}.png", dpi=150); plt.close()

# 10) Топ-«важкі» приклади (висока впевненість, але помилкові)
dfp["err"] = (dfp["true"] != dfp["pred"]).astype(int)
hard_fp = dfp[(dfp["true"]==0) & (dfp["pred"]==1)].sort_values("proba_attack", ascending=False).head(50)
hard_fn = dfp[(dfp["true"]==1) & (dfp["pred"]==0)].sort_values("proba_attack", ascending=True).head(50)
pd.concat({"hard_false_positives": hard_fp, "hard_false_negatives": hard_fn}).to_csv(ART / "top_misclassified.csv")

print("Saved plots to artifacts/:")
print("- confusion_matrix_norm.png, roc_curve.png, pr_curve.png, f1_vs_threshold.png, calibration_curve.png, proba_hist.png")
print("- feature_importance.png and/or perm_importance.png (if available)")
print("- num_dists_*.png, top_categories_*.png (if raw CSV available)")
print("- top_misclassified.csv")