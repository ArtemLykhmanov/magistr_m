# -*- coding: utf-8 -*-
import json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

ART = Path("artifacts")
ART.mkdir(exist_ok=True, parents=True)


cm_path = ART / "confusion_matrix.csv"
if cm_path.exists():
    cm = np.loadtxt(cm_path, delimiter=";").astype(int)
else:
   
    demo_pred = ART / "demo_predictions.csv"
    if demo_pred.exists():
        df = pd.read_csv(demo_pred)
        cm = confusion_matrix(df["true"], df["pred"])
    else:
        raise SystemExit("Немає ні confusion_matrix.csv, ні demo_predictions.csv")

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
for (i,j),v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha='center', va='center')
plt.tight_layout()
plt.savefig(ART / "confusion_matrix.png", dpi=150)
plt.close()

# 2) PR-крива (з demo_predictions.csv)
demo_pred = ART / "demo_predictions.csv"
if demo_pred.exists():
    df = pd.read_csv(demo_pred)
    y_true = df["true"].values
    y_score = df["proba_attack"].values
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ART / "pr_curve.png", dpi=150)
    plt.close()

# 3) Latency гістограма (зі stream_alerts.csv)
stream_log = ART / "stream_alerts.csv"
if stream_log.exists():

    
    pass

print("Saved: artifacts/confusion_matrix.png, pr_curve.png (за наявності demo_predictions.csv)")
