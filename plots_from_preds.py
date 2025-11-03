# plots_from_preds.py
# Робить графіки/звіт з файлу з ймовірностями (та, за можливості, з істинними мітками)
# Працює з різними назвами колонок, уміє сам знайти y_true з Label/attack_cat,
# або перейти в режим "усі benign" через --assume-all-benign.
#
# Приклади:
# 1) Є колонка Label у файлі:
#    python -u plots_from_preds.py --preds artifacts\preds_monday.csv --thr 0.173333 --label-col Label
# 2) Взагалі немає міток:
#    python -u plots_from_preds.py --preds artifacts\preds_monday.csv --thr 0.173333 --assume-all-benign 1
#
# Результати: artifacts/probs_hist.png, (за наявності y_true: pr_curve.png, confusion_matrix.png), artifacts/plots_report.txt

import os
import argparse
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

import matplotlib.pyplot as plt

POS_LABELS = {
    # UNSW / CICIDS / узагальнені позитивні:
    "malicious", "attack", "anomaly", "bot", "ddos", "dos", "infiltration",
    "bruteforce", "portscan", "webattack", "ftp-patator", "ssh-patator",
    "heartbleed", "xss", "sql-injection", "sqlinjection", "benign=0", "normal=0"
}
NEG_LABELS = {"benign", "normal", "clean"}

PROBA_CANDIDATES = ["proba", "prob", "p", "score", "y_pred_proba"]  # як називають імовірність
LABEL_CANDIDATES = ["y_true", "label", "Label", "class", "target", "attack_cat"]

def find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def map_labels_to_binary(series: pd.Series) -> np.ndarray:
    """Мапінг довільних текстових міток у 0/1.
       Все, що не 'benign/normal/clean' -> 1 (атака)."""
    vals = series.fillna("").astype(str).str.strip()
    def to_bin(v: str) -> int:
        lv = v.lower()
        if lv in NEG_LABELS:
            return 0
        # У CIC-IDS2017 'BENIGN' — окреме слово
        if lv == "benign" or lv == "normal" or lv == "clean":
            return 0
        # Якщо явно “0” як негатив
        if lv in {"0", "false", "neg"}:
            return 0
        # Якщо явно “1” як позитив
        if lv in {"1", "true", "pos"}:
            return 1
        # Багато атак мають власні назви — трактуємо як 1
        return 1
    return vals.apply(to_bin).astype(int).values

def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main(args):
    outdir = args.outdir
    ensure_outdir(outdir)

    df = pd.read_csv(args.preds)
    # 1) Знаходимо колонку з імовірністю
    proba_col = find_col(df, PROBA_CANDIDATES)
    if proba_col is None:
        raise ValueError(f"Не знайшов колонку з імовірністю серед {PROBA_CANDIDATES}. Колонки файлу: {list(df.columns)}")
    p = df[proba_col].astype(float).values

    # 2) Знаходимо істинні мітки (y_true) або формуємо їх
    y = None
    if args.label_col:
        if args.label_col not in df.columns:
            raise ValueError(f"--label-col='{args.label_col}' відсутня у файлі. Є колонки: {list(df.columns)}")
        y = map_labels_to_binary(df[args.label_col])
    else:
        # спроба знайти стандартні назви
        label_col = find_col(df, ["y_true"])
        if label_col is not None:
            y = df[label_col].astype(int).values
        else:
            label_col = find_col(df, ["Label", "label", "class", "target", "attack_cat"])
            if label_col is not None:
                y = map_labels_to_binary(df[label_col])

    if y is None and args.assume_all_benign:
        # немає міток — вважаємо, що все benign (0)
        y = np.zeros_like(p, dtype=int)
        print("[INFO] Лейбли відсутні — використовую режим --assume-all-benign: y_true=0 для всіх рядків.")

    # 3) Будуємо гістограму proba
    plt.figure()
    plt.hist(p, bins=50)
    plt.axvline(args.thr, linestyle="--")
    plt.title("Distribution of predicted probabilities")
    plt.xlabel("proba (attack)")
    plt.ylabel("count")
    hist_path = os.path.join(outdir, "probs_hist.png")
    plt.savefig(hist_path, bbox_inches="tight", dpi=140)
    plt.close()

    report_lines = []
    report_lines.append(f"File: {args.preds}")
    report_lines.append(f"Rows: {len(p)}")
    report_lines.append(f"Threshold: {args.thr}")
    report_lines.append(f"Proba column: {proba_col}")

    # 4) Якщо є y_true — робимо повний набір метрик/графіків
    if y is not None:
        # Чи є обидва класи?
        has_pos = (y == 1).any()
        has_neg = (y == 0).any()

        pred = (p >= args.thr).astype(int)

        if has_pos and has_neg:
            pr_auc = average_precision_score(y, p)
            f1 = f1_score(y, pred)

            precision, recall, thr = precision_recall_curve(y, p)
            plt.figure()
            plt.plot(recall, precision)
            plt.title(f"PR curve (AP={pr_auc:.4f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            pr_path = os.path.join(outdir, "pr_curve.png")
            plt.savefig(pr_path, bbox_inches="tight", dpi=140)
            plt.close()

            cm = confusion_matrix(y, pred, labels=[0,1])
            plt.figure()
            plt.imshow(cm, aspect="auto")
            plt.title("Confusion matrix [[TN FP],[FN TP]]")
            plt.colorbar()
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            cm_path = os.path.join(outdir, "confusion_matrix.png")
            plt.savefig(cm_path, bbox_inches="tight", dpi=140)
            plt.close()

            report_lines.append(f"PR-AUC: {pr_auc:.4f}")
            report_lines.append(f"F1@{args.thr:.2f}: {f1:.4f}")
            report_lines.append(f"Confusion: {cm.tolist()}")
            report_lines.append("Classification report:")
            report_lines.append(classification_report(y, pred, digits=4))
        else:
            # Тільки один клас — PR-AUC/F1 некоректні. Дамо FPR/TPR для кейсу all-benign або all-attack.
            report_lines.append("Only one class present in y_true → PR-AUC/F1 неінформативні.")
            if has_neg and not has_pos:
                # усі benign → FPR
                fpr = float((p >= args.thr).mean())
                report_lines.append(f"[Benign-only] FPR@{args.thr:.2f} = {fpr:.4f}")
            elif has_pos and not has_neg:
                # усі атаки → TPR
                tpr = float((p >= args.thr).mean())
                report_lines.append(f"[Attack-only] TPR@{args.thr:.2f} = {tpr:.4f}")
    else:
        # 5) Взагалі без міток — покажемо лише, скільки спрацювань
        rate = float((p >= args.thr).mean())
        report_lines.append("No ground-truth labels available.")
        report_lines.append(f"Alerts rate@{args.thr:.2f}: {rate:.4f} (частка записів з proba >= thr)")

    # 6) Збережемо короткий звіт
    rep_path = os.path.join(outdir, "plots_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {hist_path}")
    print(f"Saved report: {rep_path}")
    if os.path.exists(os.path.join(outdir, "pr_curve.png")):
        print(f"Saved: {os.path.join(outdir, 'pr_curve.png')}")
    if os.path.exists(os.path.join(outdir, "confusion_matrix.png")):
        print(f"Saved: {os.path.join(outdir, 'confusion_matrix.png')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV з імовірностями (та, бажано, мітками)")
    ap.add_argument("--thr", type=float, required=True, help="Поріг класифікації")
    ap.add_argument("--label-col", type=str, default=None, help="Якщо у файлі є оригінальна колонка мітки (наприклад 'Label')")
    ap.add_argument("--assume-all-benign", type=int, default=0, help="1 = вважати, що всі записи benign, якщо міток немає")
    ap.add_argument("--outdir", type=str, default="artifacts")
    args = ap.parse_args()
    main(args)
