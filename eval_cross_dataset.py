# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# наше завантаження 
from src.data_loaders import load_dataset, split_num_cat

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", make_ohe())])
    return ColumnTransformer([
        ("num", num_pipe, list(num_cols)),
        ("cat", cat_pipe, list(cat_cols)),
    ], remainder="drop", sparse_threshold=0.0)

def make_model(name: str, seed: int):
    name = name.lower()
    if name in ("rf", "randomforest"):
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if name in ("mlp", "nn"):
        return MLPClassifier(hidden_layer_sizes=(128,64), activation="relu",
                             learning_rate_init=1e-3, batch_size=512,
                             max_iter=30, early_stopping=True, random_state=seed)
    if name in ("logreg","lr","logistic"):
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"Unknown model: {name}")

def drop_cols(df: pd.DataFrame, names):
    cols = [c.strip() for c in names.split(",") if c.strip()]
    keep = [c for c in cols if c in df.columns]
    if keep:
        print(f"[INFO] dropping cols: {keep}")
        return df.drop(columns=keep)
    return df

def main(a):
    # TRAIN
    Xtr, ytr, _, _ = load_dataset(Path(a.csv_train), label_col_hint=a.label_col_train)
    if a.drop_cols:
        Xtr = drop_cols(Xtr, a.drop_cols)
    if a.max_rows_train and len(Xtr) > a.max_rows_train:
        Xtr, ytr = Xtr.iloc[:a.max_rows_train].copy(), ytr[:a.max_rows_train]

    # TEST
    Xte, yte, _, _ = load_dataset(Path(a.csv_test), label_col_hint=a.label_col_test)
    if a.drop_cols:
        Xte = drop_cols(Xte, a.drop_cols)
    if a.max_rows_test and len(Xte) > a.max_rows_test:
        Xte, yte = Xte.iloc[:a.max_rows_test].copy(), yte[:a.max_rows_test]

    num_cols, cat_cols = split_num_cat(Xtr)
    pre = build_preprocessor(num_cols, cat_cols)
    clf = make_model(a.model, a.seed)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    print(f"[INFO] train_rows={len(Xtr)} test_rows={len(Xte)} | model={a.model}")
    pipe.fit(Xtr, ytr)

    # ймовірності
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        p = pipe.predict_proba(Xte)[:,1]
    else:
        if hasattr(pipe.named_steps["clf"], "decision_function"):
            s = pipe.decision_function(Xte)
            p = 1.0/(1.0+np.exp(-s))
        else:
            p = pipe.predict(Xte).astype(float)

    # метрики
    thr = a.threshold
    pr_auc = float(average_precision_score(yte, p))
    pred = (p >= thr).astype(int)

    print(f"\n=== CROSS-DATASET EVAL ===")
    print(f"PR-AUC={pr_auc:.4f} | F1@{thr:.2f}={f1_score(yte, pred, zero_division=0):.4f}")

    # якщо в тесті тільки 0 або тільки 1 — корисно надрукувати FPR/TPR
    labels = np.unique(yte)
    if len(labels) == 1:
        if labels[0] == 0:
            fpr = float((pred==1).mean())
            print(f"[Benign-only TEST] FPR@{thr:.2f} = {fpr:.4f}")
        else:
            tpr = float((pred==1).mean())
            print(f"[Attack-only TEST] TPR@{thr:.2f} = {tpr:.4f}")
    else:
        print("\nClassification report:")
        print(classification_report(yte, pred, digits=4))
        cm = confusion_matrix(yte, pred, labels=[0,1])
        print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

    # зберегти ймовірності
    if a.save_probs:
        out = Xte.copy()
        out.insert(0, "true", yte.astype(int))
        out["proba"] = p
        out["pred"] = pred
        Path(a.save_probs).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.save_probs, index=False, encoding="utf-8")
        print(f"\nSaved probabilities to: {a.save_probs}")

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--csv-train", required=True)
    P.add_argument("--csv-test", required=True)
    P.add_argument("--label-col-train", default=None)
    P.add_argument("--label-col-test", default=None)
    P.add_argument("--drop-cols", default="Flow ID,Source IP,Destination IP,Timestamp")
    P.add_argument("--model", default="rf")
    P.add_argument("--seed", type=int, default=42)
    P.add_argument("--threshold", type=float, default=0.50)
    P.add_argument("--max_rows_train", type=int, default=None)
    P.add_argument("--max_rows_test", type=int, default=None)
    P.add_argument("--save_probs", default=None)
    main(P.parse_args())
