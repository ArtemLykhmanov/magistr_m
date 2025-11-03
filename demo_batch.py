# -*- coding: utf-8 -*-
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, average_precision_score, f1_score, confusion_matrix

MODEL_PATH = Path("artifacts/model.joblib")
FEAT_PATH  = Path("artifacts/features.json")
CSV_PATH   = Path("data/UNSW_NB15_training-set.csv")  
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Параметри демо
SAMPLE_ROWS = 20000     # скільки рядків взяти для швидкого показу 
THRESHOLD   = 0.5       # поріг для класу "атака" (можна змінювати для чутливості)

# 1) Завантаження артефактів
model = joblib.load(MODEL_PATH)
meta = json.load(open(FEAT_PATH, "r", encoding="utf-8"))
FEATURES = meta["features"]
NUM_COLS = set(meta.get("num_cols", []))
CAT_COLS = set(meta.get("cat_cols", []))

# 2) Дані
df = pd.read_csv(CSV_PATH)
# Визначаємо колонку мітки
LABEL_CANDIDATES = ["label", "Label", "class", "target"]
label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
if label_col is None:
    raise ValueError(f"Не знайшов колонку мітки серед {LABEL_CANDIDATES}")

# y (як у train_baseline.py)
y_raw = df[label_col]
if pd.api.types.is_bool_dtype(y_raw):
    y = y_raw.astype(int).values
elif pd.api.types.is_numeric_dtype(y_raw):
    y = (y_raw.astype(float) > 0).astype(int).values
else:
    POS_LABELS = {"malicious","attack","anomaly","botnet","dos","ddos","malware",
                  "infiltration","portscan","bruteforce","brute force","1","true"}
    y = (y_raw.astype(str).str.strip().str.lower().isin(POS_LABELS)).astype(int).values

# 3) Беремо підмножину (для швидкості показу) і формуємо X так само, як в API
if len(df) > SAMPLE_ROWS:
    df = df.sample(n=SAMPLE_ROWS, random_state=42)
    y = y[df.index.values]

def row_to_typed_dict(row: pd.Series) -> dict:
    out = {}
    for f in FEATURES:
        if f in NUM_COLS:
            val = row.get(f, 0)
            try: val = float(val)
            except Exception: val = 0.0
            out[f] = val
        elif f in CAT_COLS:
            val = row.get(f, "NA")
            out[f] = "NA" if pd.isna(val) else str(val)
        else:
            val = row.get(f, "NA")
            out[f] = "NA" if pd.isna(val) else str(val)
    return out

records = [row_to_typed_dict(r) for _, r in df.iterrows()]
X = pd.DataFrame.from_records(records)[FEATURES]  # важливо: DataFrame з іменами колонок

# 4) Інференс + метрики
proba = model.predict_proba(X)[:, 1]
pred  = (proba >= THRESHOLD).astype(int)

pr_auc = average_precision_score(y, proba)
f1     = f1_score(y, pred)
report = classification_report(y, pred, digits=4)
cm     = confusion_matrix(y, pred)

print("=== DEMO (batch) ===")
print(f"Rows: {len(X)} | Threshold: {THRESHOLD}")
print(f"PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")
print("Confusion matrix [ [TN FP] [FN TP] ]:")
print(cm)
print("\nClassification report:")
print(report)

# 5) Збереження результатів для показу/додатку
out_csv = OUTPUT_DIR / "demo_predictions.csv"
pd.DataFrame({
    "pred": pred,
    "proba_attack": proba,
    "true": y
}).to_csv(out_csv, index=False)

with open(OUTPUT_DIR / "demo_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Rows: {len(X)} | Threshold: {THRESHOLD}\n")
    f.write(f"PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}\n")
    f.write("Confusion matrix [[TN FP],[FN TP]]:\n")
    f.write(np.array2string(cm) + "\n\n")
    f.write(report)

# 6) Топ-кейси для демонстрації
df_show = df.copy()
df_show["proba_attack"] = proba
df_show["pred"] = pred
df_show["true"] = y

top_attack = df_show.sort_values("proba_attack", ascending=False).head(5)
top_benign = df_show.sort_values("proba_attack", ascending=True).head(5)

print("\n--- TOP 5 ATTACK-LIKE (highest probabilities) ---")
cols_preview = [c for c in ["proto","service","state","dur","spkts","dpkts","sbytes","dbytes","proba_attack","pred","true"] if c in df_show.columns]
print(top_attack[cols_preview].to_string(index=False))

print("\n--- TOP 5 BENIGN-LIKE (lowest probabilities) ---")
print(top_benign[cols_preview].to_string(index=False))

print(f"\nSaved: {out_csv}")
print("Also saved: artifacts/demo_report.txt")
