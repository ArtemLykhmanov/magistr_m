# make_cv_summary_plot.py
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("artifacts/cv_metrics.csv")
fig = plt.figure(figsize=(6,4))
plt.bar(df["model"], df["PR_AUC_mean"])
plt.title("PR-AUC (mean, CV)")
plt.ylabel("PR-AUC")
plt.tight_layout()
Path("artifacts").mkdir(exist_ok=True, parents=True)
plt.savefig("artifacts/cv_pr_auc_bar.png", dpi=160)

fig = plt.figure(figsize=(6,4))
plt.bar(df["model"], df["F1_mean"])
plt.title("F1 (mean, CV)")
plt.ylabel("F1")
plt.tight_layout()
plt.savefig("artifacts/cv_f1_bar.png", dpi=160)
print("Saved: artifacts/cv_pr_auc_bar.png, cv_f1_bar.png")
