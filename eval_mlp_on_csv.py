# eval_mlp_on_csv.py
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    classification_report,
    confusion_matrix,
)

import torch

FORBIDDEN_COLS = [
    # UNSW
    "id", "attack_cat",
    # CIC-IDS2017
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
]

def load_preprocessor(art_dir):
    pre = joblib.load(os.path.join(art_dir, "preprocessor.joblib"))
    # Витягуємо список колонок, на яких тренувався препроцесор
    if hasattr(pre, "feature_names_in_"):
        required = list(pre.feature_names_in_)
    else:
        # Бекап: спробуємо прочитати з маніфесту
        man_path = os.path.join(art_dir, "mlp_manifest.json")
        required = None
        if os.path.exists(man_path):
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            required = man.get("input_cols")
        if required is None:
            raise RuntimeError("Не вдалося визначити required columns для препроцесора.")
    return pre, required

def load_model(art_dir, device="cpu"):
    pt_path = os.path.join(art_dir, "mlp.pt")
    # Torch 2.6 попереджає про weights_only — тут це наш власний артефакт, вантажимо як TorchScript
    try:
        model = torch.jit.load(pt_path, map_location=device)
        as_logits = True  # ми тренували з BCEWithLogits — модель повертає логіти
    except Exception:
        # як fallback — звичайний state_dict
        state = torch.load(pt_path, map_location=device, weights_only=False)
        from train_mlp_torch import MLP  # клас має бути доступний
        input_dim = state["input_dim"] if "input_dim" in state else None
        hidden = state.get("hidden", [256, 128])
        drop = state.get("dropout", 0.1)
        if input_dim is None:
            raise RuntimeError("Не знайдено input_dim у checkpoint для звичайного state_dict.")
        model = MLP(input_dim=input_dim, hidden=hidden, dropout=drop)
        model.load_state_dict(state["model"])
        model.to(device)
        as_logits = True
    model.eval()
    return model, as_logits

def clean_and_align(df: pd.DataFrame, required_cols):
    # Унікальні назви
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Дропим заборонені/висококардинальні, якщо є
    to_drop = [c for c in FORBIDDEN_COLS if c in df.columns]
    if to_drop:
        print(f"[INFO] dropping cols: {to_drop}")
        df = df.drop(columns=to_drop)

    # Якщо є мітка — залишимо для метрик, але з ознак її приберемо нижче.
    label_candidates = ["label", "Label", "class", "target"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break

    # Формуємо Xdf: додаємо відсутні required колонки (нулями), прибираємо зайві
    Xdf = pd.DataFrame(index=df.index)
    for c in required_cols:
        if c in df.columns:
            Xdf[c] = df[c]
        else:
            # якщо колонки не вистачає — створюємо числову з нулями
            Xdf[c] = 0

    # Переконаємось у правильному порядку
    Xdf = Xdf[required_cols]

    # Спробуємо привести очевидні числові до float
    for c in Xdf.columns:
        if Xdf[c].dtype == object:
            # можливо це числове з "inf"/"NaN" як строки — спробуємо
            Xdf[c] = pd.to_numeric(Xdf[c], errors="ignore")
    return Xdf, label_col, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Шлях до CSV з вибіркою")
    ap.add_argument("--threshold", type=float, required=True, help="Поріг для класифікації")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Де зберігаються mlp.pt та preprocessor.joblib")
    ap.add_argument("--label-col", default=None, help="Назва колонки мітки, якщо хочеш задати явно")
    ap.add_argument("--save-probs", default=None, help="Куди зберегти ймовірності (CSV)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    device = "cpu"

    pre, required = load_preprocessor(args.artifacts_dir)
    model, as_logits = load_model(args.artifacts_dir, device=device)

    # Вирівнюємо колонки під препроцесор
    Xdf, auto_label, raw = clean_and_align(df, required)

    # Визначаємо назву колонки мітки
    y_col = args.label_col if args.label_col else auto_label
    y = None
    if y_col and y_col in raw.columns:
        # Ціль бінаризуємо: "BENIGN" / "ATTACK" → 0/1 або 0/1 з int
        if raw[y_col].dtype == object:
            y = (raw[y_col].astype(str).str.upper() != "BENIGN").astype(int).values
        else:
            y = raw[y_col].astype(int).values

    # Трансформуємо
    X = pre.transform(Xdf)

    # Прогноз
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X)).float())
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.flatten().cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits)) if as_logits else logits

    thr = float(args.threshold)
    pred = (proba >= thr).astype(int)

    print(f"[INFO] rows={len(proba)} | thr={thr:.3f}")
    if y is not None and len(np.unique(y)) > 1:
        ap_score = average_precision_score(y, proba)
        f1 = f1_score(y, pred)
        print(f"PR-AUC={ap_score:.4f} | F1@{thr:.2f}={f1:.4f}\n")
        print("Classification report:")
        print(classification_report(y, pred, digits=4))
        print("Confusion matrix [[TN FP],[FN TP]]:")
        print(confusion_matrix(y, pred))
    elif y is not None:
        # одна кляса в y_true — лише FPR для «0»-класу
        neg_mask = (y == 0)
        fpr = (pred[neg_mask] == 1).mean() if neg_mask.any() else np.nan
        print("Only one class present in y_true → PR-AUC/F1 неінформативні.")
        print(f"[Benign-only] FPR@{thr:.2f} = {fpr:.4f}")
    else:
        print("y_true відсутня — виводжу лише сумарну позитивну частку:")
        print(f"pos_rate@{thr:.2f} = {(pred==1).mean():.4f}")

    if args.save_probs:
        out = pd.DataFrame({"proba": proba, "pred": pred})
        if y is not None:
            out["y_true"] = y
        out.to_csv(args.save_probs, index=False)
        print(f"Saved probabilities to: {args.save_probs}")

if __name__ == "__main__":
    main()
