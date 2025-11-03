# -*- coding: utf-8 -*-
import json, time, statistics
from pathlib import Path
import pandas as pd
import requests
import numpy as np
from pathlib import Path
import os

CSV_PATH = Path(os.getenv("IDS_CSV", "data/UNSW_NB15_training-set.csv"))
API_URL      = "http://127.0.0.1:8000/predict"
FEAT_PATH    = Path("artifacts/features.json")
MAX_ROWS     = 500       # скільки рядків програти всього
CAL_ROWS     = 120       # скільки перших рядків використати для «калібрування» порогу
TARGET_ALERT = 0.10      # бажана частка алертів (10%), якщо немає істини
SLEEP        = 0.01      # пауза між «потоками», сек
LOG_PATH     = Path("artifacts/stream_alerts.csv")

# --- завантажимо метадані фіч від тренера ---
meta = json.load(open(FEAT_PATH, "r", encoding="utf-8"))
FEATURES = meta["features"]
NUM_COLS = set(meta.get("num_cols", []))
CAT_COLS = set(meta.get("cat_cols", []))

# --- зчитуємо CSV і обмежуємо рядки для демо ---
df = pd.read_csv(CSV_PATH)
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=123).reset_index(drop=True)

# знайдемо колонку істини (якщо є) для об'єктивних метрик
LABEL_CANDIDATES = ["label","Label","class","target"]
label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
y_true = None
if label_col is not None:
    yraw = df[label_col]
    if pd.api.types.is_numeric_dtype(yraw):
        y_true = (yraw.astype(float) > 0).astype(int).values
    else:
        POS = {"malicious","attack","anomaly","botnet","dos","ddos","malware","infiltration","portscan","bruteforce","brute force","1","true"}
        y_true = (yraw.astype(str).str.strip().str.lower().isin(POS)).astype(int).values

def make_payload(row: pd.Series) -> dict:
    """Побудувати JSON з усіма фічами, які чекала модель."""
    vals = {}
    for f in FEATURES:
        if f in NUM_COLS:
            v = row[f] if f in row.index else 0
            try: v = float(v)
            except Exception: v = 0.0
            vals[f] = v
        elif f in CAT_COLS:
            v = row[f] if f in row.index else "NA"
            vals[f] = "NA" if pd.isna(v) else str(v)
        else:
            v = row[f] if f in row.index else "NA"
            vals[f] = "NA" if pd.isna(v) else str(v)
    return {"values": vals}

def predict_proba_batch_via_api(sample_df: pd.DataFrame) -> np.ndarray:
    probs = []
    for _, r in sample_df.iterrows():
        payload = make_payload(r)
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        probs.append(float(resp.json().get("probability", 0.0)))
    return np.array(probs)

# --- 1) КАЛІБРУЄМО ПОРОГ ---
cal_df = df.iloc[:min(CAL_ROWS, len(df))].copy()
probs_cal = predict_proba_batch_via_api(cal_df)

if y_true is not None:
    y_cal = y_true[:len(probs_cal)]
    # підберемо поріг з кроком 0.01 за максимальним F1 (просто та наочно)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.2, 0.9, 71):  # 0.2..0.9
        pred = (probs_cal >= thr).astype(int)
        tp = int(((pred==1)&(y_cal==1)).sum())
        fp = int(((pred==1)&(y_cal==0)).sum())
        fn = int(((pred==0)&(y_cal==1)).sum())
        f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    THRESHOLD = best_thr
else:
    # немає істини — підженемо під бажану частку алертів
    q = 1.0 - float(TARGET_ALERT)
    THRESHOLD = float(np.quantile(probs_cal, q))

print(f"=== STREAM DEMO → {API_URL} | rows={len(df)} | threshold={THRESHOLD:.2f} ===")

# --- 2) СТРІМ З ОБРАНИМ ПОРОГОМ ---
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_PATH, "w", encoding="utf-8") as logf:
    logf.write("proba,pred,true,proto,service,state,dur\n")

latencies = []
alerts = tp = fp = fn = tn = sent = 0

for i, (_, row) in enumerate(df.iterrows()):
    payload = make_payload(row)
    t0 = time.time()
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        print(f"[ERR] request failed: {e}")
        continue
    dt_ms = (time.time() - t0) * 1000.0
    latencies.append(dt_ms)
    sent += 1

    proba = float(resp.get("probability", 0.0))
    pred  = int(proba >= THRESHOLD)
    true  = None if y_true is None else int(y_true[i])

    if pred == 1:
        alerts += 1

    if true is not None:
        if   pred==1 and true==1: tp += 1
        elif pred==1 and true==0: fp += 1
        elif pred==0 and true==1: fn += 1
        elif pred==0 and true==0: tn += 1

    proto   = str(payload["values"].get("proto","-"))
    service = str(payload["values"].get("service","-"))
    state   = str(payload["values"].get("state","-"))
    dur     = payload["values"].get("dur","-")

    if pred == 1:
        print(f"[ALERT] p={proba:.3f} proto={proto:<5} service={service:<10} state={state:<12} dur={dur}")

    with open(LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(f"{proba},{pred},{'' if true is None else true},{proto},{service},{state},{dur}\n")

    time.sleep(SLEEP)

# --- 3) ПІДСУМКИ ---
p50 = statistics.median(latencies) if latencies else float("nan")
p95 = sorted(latencies)[int(len(latencies)*0.95)-1] if latencies else float("nan")
print(f"\n=== SUMMARY ===")
print(f"sent={sent} | alerts={alerts} | latency_ms: p50={p50:.1f}, p95={p95:.1f}")

if y_true is not None:
    precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp / (tp+fn) if (tp+fn)>0 else 0.0
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"precision={precision:.3f} | recall={recall:.3f} (threshold={THRESHOLD:.2f})")

print(f"\nAlerts log saved to: {LOG_PATH.resolve()}")
