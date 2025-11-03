# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import joblib

# наші утиліти завантаження даних
from src.data_loaders import load_dataset, split_num_cat

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- утиліти ----------

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # OneHotEncoder: у нових sklearn (>=1.2) параметр sparse_output, у старих — sparse
    def make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe())
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.0
    )
    return pre


def make_model(name: str, seed: int):
    name = name.lower()
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            n_jobs=None if "n_jobs" not in LogisticRegression().get_params() else -1,
            solver="lbfgs"
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=seed,
            n_jobs=-1,
            class_weight=None
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=30,
            random_state=seed,
            learning_rate_init=1e-3,
            verbose=False
        )
    if name == "rf_cal":
        base = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=seed,
            n_jobs=-1,
            class_weight=None
        )
        # Калібрування Platt (sigmoid) з внутрішнім CV=3 на трейні кожного фолда
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=3
        )
    raise ValueError(f"Unknown model: {name}")

def pr_auc_and_f1(y_true, proba, thr=0.5):
    pr_auc = float(average_precision_score(y_true, proba))
    pred = (proba >= thr).astype(int)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    return pr_auc, f1


# ---------- основна логіка ----------

def main(args):
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # завантаження даних
    Xdf, y, ts, raw = load_dataset(csv_path, label_col_hint=args.label_col)
    if args.max_rows and len(Xdf) > args.max_rows:
        Xdf = Xdf.iloc[:args.max_rows].copy()
        y = y[:args.max_rows]
        if ts is not None and len(ts) >= args.max_rows:
            ts = ts.iloc[:args.max_rows]
            
    # --- відкидаємо колонки з великою кардинальністю (ідентифікатори / час) ---
    DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp",
             "Src IP", "Dst IP", "Time"]  # на всяк випадок альтернативні назви
    to_drop = [c for c in DROP_COLS if c in Xdf.columns]
    if to_drop:
        print(f"[INFO] dropping high-cardinality cols: {to_drop}")
        Xdf = Xdf.drop(columns=to_drop)

    num_cols, cat_cols = split_num_cat(Xdf)
    models = [m.strip() for m in args.models.split(",")]
    folds = int(args.folds)
    seed = int(args.seed)

    print(f"[INFO] csv={csv_path} | rows={len(Xdf)} | folds={folds} (stratified) | models={models}")
    print(f"[INFO] num_cols={len(num_cols)} cat_cols={len(cat_cols)}\n")

    pre = build_preprocessor(num_cols, cat_cols)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    rows_summary = []
    best_model_name = None
    best_model_score = -1.0
    best_model_pipe = None

    # для збереження CV-метрик
    cv_metrics_all = []

    for model_name in models:
        print(f"=== Model: {model_name} ===")
        clf = make_model(model_name, seed)
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        fold_scores = []
        fold_f1s = []

        for i, (tr_idx, te_idx) in enumerate(cv.split(Xdf, y), start=1):
            X_tr, X_te = Xdf.iloc[tr_idx], Xdf.iloc[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            print(f"  [Fold {i}/{folds}] train={len(X_tr)} test={len(X_te)} ...")
            pipe_i = clone(pipe)
            pipe_i.fit(X_tr, y_tr)
            # predict_proba безпечніше через getattr
            if hasattr(pipe_i.named_steps["clf"], "predict_proba"):
                proba = pipe_i.predict_proba(X_te)[:, 1]
            else:
                # наприклад, якщо модель не має proba — використовуємо decision_function
                if hasattr(pipe_i.named_steps["clf"], "decision_function"):
                    s = pipe_i.decision_function(X_te)
                    # приведемо до [0,1] через сигмоїду
                    proba = 1.0 / (1.0 + np.exp(-s))
                else:
                    # fallback: бінарний предикт як "ймовірність"
                    proba = pipe_i.predict(X_te).astype(float)

            pr_auc, f1 = pr_auc_and_f1(y_te, proba, thr=0.5)
            fold_scores.append(pr_auc)
            fold_f1s.append(f1)
            print(f"    PR-AUC={pr_auc:.4f} | F1@0.5={f1:.4f}")

            cv_metrics_all.append({
                "model": model_name,
                "fold": i,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "pr_auc": pr_auc,
                "f1_at_0.5": f1
            })

        mean_auc = float(np.mean(fold_scores))
        mean_f1 = float(np.mean(fold_f1s))
        rows_summary.append({"model": model_name, "PR_AUC_mean": mean_auc, "F1_mean": mean_f1})

        print(f"  >> mean PR-AUC={mean_auc:.4f} | mean F1@0.5={mean_f1:.4f}\n")

        if mean_auc > best_model_score:
            best_model_score = mean_auc
            best_model_name = model_name
            best_model_pipe = clone(pipe)

    # підсумкова таблиця
    df_sum = pd.DataFrame(rows_summary).sort_values("PR_AUC_mean", ascending=False).reset_index(drop=True)
    print("\n=== CV SUMMARY ===")
    print(df_sum.to_string(index=False))

    selected = df_sum.iloc[0]["model"]
    print(f"Selected: {selected} (PR-AUC={df_sum.iloc[0]['PR_AUC_mean']:.4f})")

    # перевчимо вибрану модель на всіх даних
    best_clf = make_model(selected, seed)
    final_pipe = Pipeline([("pre", pre), ("clf", best_clf)])
    final_pipe.fit(Xdf, y)

    # --------- артефакти ----------
    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1) модель
    model_path = artifacts / "model.joblib"
    joblib.dump(final_pipe, model_path)

    # 2) фічі
    features_info = {
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
        "label_col": args.label_col or "auto",
        "csv_used": str(csv_path.resolve())
    }
    (artifacts / "features.json").write_text(json.dumps(features_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) детальні метрики по фолдах
    pd.DataFrame(cv_metrics_all).to_csv(artifacts / "cv_metrics.csv", index=False)

    # 4) demo_predictions.csv (на перших N рядках)
    N = min(2000, len(Xdf))
    X_demo = Xdf.iloc[:N].copy()
    y_demo = y[:N]
    if hasattr(final_pipe.named_steps["clf"], "predict_proba"):
        p_demo = final_pipe.predict_proba(X_demo)[:, 1]
    else:
        if hasattr(final_pipe.named_steps["clf"], "decision_function"):
            s = final_pipe.decision_function(X_demo)
            p_demo = 1.0 / (1.0 + np.exp(-s))
        else:
            p_demo = final_pipe.predict(X_demo).astype(float)

    demo = X_demo.copy()
    demo.insert(0, "true", y_demo.astype(int))
    demo["proba_attack"] = p_demo
    demo["pred"] = (p_demo >= 0.5).astype(int)
    demo.to_csv(artifacts / "demo_predictions.csv", index=False)

    # 5) manifest.json
    manifest = {
        "selected_model": selected,
        "cv_summary": df_sum.to_dict(orient="records"),
        "seed": seed,
        "folds": folds,
        "rows": int(len(Xdf)),
        "artifacts": [
            str(model_path),
            "artifacts/features.json",
            "artifacts/cv_metrics.csv",
            "artifacts/demo_predictions.csv",
        ]
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Артефакти → artifacts/model.joblib, features.json, cv_metrics.csv, demo_predictions.csv, manifest.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Шлях до CSV (UNSW/CIC тощо)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--models", type=str, default="logreg,rf,mlp",
                   help="Список моделей через кому: logreg,rf,mlp")
    p.add_argument("--max_rows", type=int, default=None, help="Опційно: обрізати до N рядків")
    p.add_argument("--label-col", type=str, default=None, help="Опційно: точна назва стовпця мітки")
    args = p.parse_args()
    main(args)
