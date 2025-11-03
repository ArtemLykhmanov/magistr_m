# train_mlp_torch.py
# -*- coding: utf-8 -*-
import os
import json
import re
import argparse
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.utils import check_random_state

from joblib import dump

# ---------------------------
# Утиліти
# ---------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # для повної відтворюваності (дещо повільніше)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


LABEL_CANDIDATES = ["label", "Label", "class", "target", "attack_cat", "Attack_cat", "Attack Category"]

FORBID_EXACT = {
    # очевидні лейбл/похідні
    "label", "Label", "class", "target",
    "attack_cat", "Attack_cat", "Attack Category",
    # ідентифікатори/рядкові унікальні
    "id", "ID", "Id", "flow_id", "Flow ID",
    # часто ламають узагальнення
    "Source IP", "Destination IP", "Timestamp",
    "srcip", "dstip", "src_ip", "dst_ip", "ts", "time", "Time"
}

FORBID_REGEX = [
    r".*ip$", r".*ip_addr$", r"^src.*", r"^dst.*", r".*mac$", r".*uuid.*", r".*guid.*",
    r".*timestamp.*", r".*time.*"
]


def find_label_col(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f'Вказана колонка мітки "{explicit}" відсутня у CSV')
        return explicit
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Не знайшов колонку мітки серед {LABEL_CANDIDATES}. "
                     f"Вкажи явно через --label-col")


def is_forbidden(name: str) -> bool:
    n = name.strip()
    if n in FORBID_EXACT:
        return True
    for pat in FORBID_REGEX:
        if re.fullmatch(pat, n, flags=re.IGNORECASE):
            return True
    return False


def read_csv_safely(path: str, nrows: int | None = None) -> pd.DataFrame:
    # CIC/UNSW інколи потребують engine="python"
    try_list = [
        dict(encoding="utf-8", engine="c"),
        dict(encoding="utf-8", engine="python"),
        dict(encoding="latin-1", engine="python"),
    ]
    last_err = None
    for kw in try_list:
        try:
            return pd.read_csv(path, nrows=nrows, **kw)
        except Exception as e:
            last_err = e
    raise last_err


def build_preprocessor(num_cols, cat_cols):
    # Імп’ютер + скейлер для числових, OHE для категоріальних
    num_pipe = [
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ]
    # Підтримка різних версій sklearn
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = [
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", ohe),
    ]

    from sklearn.pipeline import Pipeline
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_pipe), num_cols),
            ("cat", Pipeline(cat_pipe), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, int] = (256, 128), drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden[0]),
            nn.Dropout(drop),

            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden[1]),
            nn.Dropout(drop),

            nn.Linear(hidden[1], 1)  # логіти
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


def average_precision_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # average_precision_score = PR-AUC
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        # якщо одна мітка у валідації
        return 1.0 if len(np.unique(y_true)) == 1 else 0.0


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: int = 201) -> tuple[float, float]:
    thr_grid = np.linspace(0.01, 0.99, grid)
    best_f1 = -1.0
    best_thr = 0.5
    for t in thr_grid:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return float(best_f1), float(best_thr)


def leak_guard_and_split_Xy(df: pd.DataFrame, label_col: str):
    # y
    y = df[label_col].copy()

    # нормалізуємо y у {0,1} якщо треба
    if y.dtype == object:
        # типово для UNSW: label {0,1} як int; CIC: "BENIGN"/"ATTACK" або подібне
        y_norm = y.astype(str).str.lower().map(
            lambda v: 0 if v in {"0", "benign", "normal"} else 1
        )
        if y_norm.isna().any():
            # пробуємо як int
            try:
                y_norm = y.astype(int)
            except Exception:
                # останній шанс: бінаризація за більшістю категорії
                top = y.value_counts().idxmax()
                y_norm = (y != top).astype(int)
        y = y_norm
    else:
        # int/float → бінаризація (0/1)
        y = (y.astype(float) > 0).astype(int)

    X = df.drop(columns=[label_col]).copy()

    # drop заборонених
    drop_cols = [c for c in X.columns if is_forbidden(c)]
    if drop_cols:
        print(f"[INFO] dropping forbidden cols: {drop_cols}")
        X.drop(columns=drop_cols, inplace=True, errors="ignore")

    # константи
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        print(f"[INFO] dropping constant cols: {const_cols}")
        X.drop(columns=const_cols, inplace=True)

    # very high-cardinality object
    N = len(X)
    high_card_cols = []
    for c in X.columns:
        if X[c].dtype == object:
            uq = X[c].nunique(dropna=True)
            if uq > max(2000, int(0.05 * N)):
                high_card_cols.append(c)
    if high_card_cols:
        print(f"[INFO] dropping high-cardinality cols: {high_card_cols}")
        X.drop(columns=high_card_cols, inplace=True)

    # авто-детектор «майже лейбл» у категоріальних
    y_s = y.astype(int)
    y_s.index = X.index
    leaky_cols = []
    for c in X.columns:
        if X[c].dtype == object:
            purity_hits = 0
            total = 0
            for val, idx in X.groupby(c).groups.items():
                y_sub = y_s.loc[idx]
                if len(y_sub) == 0:
                    continue
                maj = 1 if (y_sub.mean() >= 0.5) else 0
                purity_hits += (y_sub == maj).sum()
                total += len(y_sub)
            if total > 0:
                purity = purity_hits / total
                if purity >= 0.98:
                    leaky_cols.append(c)
    if leaky_cols:
        print(f"[LEAK?] dropping high-purity label-like cols: {leaky_cols}")
        X.drop(columns=leaky_cols, inplace=True)

    # заміна inf → NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # остаточний поділ на типи
    num_cols = [c for c in X.columns if X[c].dtype != object]
    cat_cols = [c for c in X.columns if X[c].dtype == object]

    print(f"[INFO] num_cols={len(num_cols)} cat_cols={len(cat_cols)}")
    return X, y.astype(int).values, num_cols, cat_cols


def make_loaders(X_tr, y_tr, X_va, y_va, batch_size: int = 512, device: str = "cpu"):
    # у тензори
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    yva_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    tr_ds = TensorDataset(Xtr_t, ytr_t)
    va_ds = TensorDataset(Xva_t, yva_t)

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return tr_dl, va_dl


def train_one_fold(X_tr, y_tr, X_va, y_va, device="cpu",
                   epochs=30, batch_size=512, lr=1e-3, weight_decay=1e-4,
                   hidden=(256, 128), dropout=0.2, patience=3):
    in_dim = X_tr.shape[1]
    model = MLP(in_dim, hidden=hidden, drop=dropout).to(device)

    # балансування класів
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_dl, va_dl = make_loaders(X_tr, y_tr, X_va, y_va, batch_size=batch_size, device=device)

    best_ap = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            probs = []
            ys = []
            for xb, yb in va_dl:
                logits = model(xb)
                p = torch.sigmoid(logits)
                probs.append(p.detach().cpu().numpy())
                ys.append(yb.detach().cpu().numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys).astype(int)

        ap = average_precision_from_probs(ys, probs)

        if ap > best_ap + 1e-5:
            best_ap = ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 1 == 0:
            # підбір найкращого F1 лише для логів
            f1, thr = best_f1_threshold(ys, probs)
            print(f"    [ep {epoch:02d}] val PR-AUC={ap:.4f} | F1@best={f1:.4f} | thr*={thr:.3f}")

        if epochs_no_improve >= patience:
            break

    # відновлюємо найкращі ваги
    if best_state is not None:
        model.load_state_dict(best_state)

    # фінальні val метрики
    model.eval()
    with torch.no_grad():
        va_logits = []
        for i in range(0, X_va.shape[0], 4096):
            xb = torch.tensor(X_va[i:i+4096], dtype=torch.float32, device=device)
            va_logits.append(model(xb).cpu().numpy())
        va_logits = np.concatenate(va_logits)
        va_probs = 1.0 / (1.0 + np.exp(-va_logits))

    ap = average_precision_from_probs(y_va, va_probs)
    f1, thr = best_f1_threshold(y_va, va_probs)
    return model, ap, f1, thr, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str, help="Шлях до CSV")
    parser.add_argument("--label-col", default=None, type=str, help="Назва колонки мітки (якщо не знайду автоматично)")
    parser.add_argument("--folds", default=3, type=int)
    parser.add_argument("--max_rows", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--hidden", default="256,128", type=str, help="Напр. '256,128'")
    parser.add_argument("--dropout", default=0.2, type=float)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    # завантаження
    df = read_csv_safely(args.csv, nrows=args.max_rows)

    # знаходимо label колонку
    label_col = find_label_col(df, args.label_col)

    # попередньо друк для контролю
    # (UNSW: інколи є 'attack_cat' -> це і є витік)
    # підміняємо 'y' і чистимо X в leak_guard_and_split_Xy:
    X_raw, y, num_cols, cat_cols = leak_guard_and_split_Xy(df, label_col)

    # будую препроцесор і одразу роблю CV (фолди стратифіковані)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    rows = len(X_raw)
    print(f"[INFO] rows={rows}")

    cv_rows = []
    oof_prob = np.zeros(rows, dtype=np.float32)
    oof_idx_all = np.arange(rows)

    # щоб препроцесор не «підглядав», фітимо його лише на train кожного фолду
    # але для TorchScript збережемо один препроцесор, навчений на ВСІХ даних наприкінці

    fold_no = 0
    for tr_idx, va_idx in skf.split(X_raw, y):
        fold_no += 1
        X_tr_df = X_raw.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va_df = X_raw.iloc[va_idx].copy()
        y_va = y[va_idx]

        # препроцесор ТІЛЬКИ на train цього фолду
        pre = build_preprocessor(num_cols=[c for c in X_tr_df.columns if X_tr_df[c].dtype != object],
                                 cat_cols=[c for c in X_tr_df.columns if X_tr_df[c].dtype == object])
        X_tr = pre.fit_transform(X_tr_df)
        X_va = pre.transform(X_va_df)

        print(f"  [Fold {fold_no}/{args.folds}] PR-AUC/F1 валід. (early stopping увімкнено)")
        model, ap, f1, thr, best_epoch = train_one_fold(
            X_tr, y_tr, X_va, y_va, device=device,
            epochs=args.epochs, batch_size=args.batch_size,
            hidden=hidden, dropout=args.dropout
        )
        # оцінимо та збережемо OOF для цього фолду
        model.eval()
        with torch.no_grad():
            va_probs = []
            for i in range(0, X_va.shape[0], 4096):
                xb = torch.tensor(X_va[i:i+4096], dtype=torch.float32, device=device)
                va_probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            va_probs = np.concatenate(va_probs)
        oof_prob[va_idx] = va_probs

        print(f"  [Fold {fold_no}/{args.folds}] PR-AUC={ap:.4f} | F1@best={f1:.4f} | thr*={thr:.3f} | best_epoch={best_epoch}")
        cv_rows.append(dict(fold=fold_no, PR_AUC=ap, F1_best=f1, thr_star=thr, best_epoch=best_epoch))

    # Загальні OOF метрики
    oof_ap = average_precision_from_probs(y, oof_prob)
    oof_f1, oof_thr = best_f1_threshold(y, oof_prob)
    print("\n=== CV SUMMARY (OOF) ===")
    print(f"OOF PR-AUC={oof_ap:.6f} | OOF best F1={oof_f1:.6f} @thr*={oof_thr:.3f}")

    # Збереження артефактів
    os.makedirs("artifacts", exist_ok=True)

    # 1) фінальний препроцесор (на ВСІХ даних) для прод/оцінки на інших CSV
    pre_full = build_preprocessor(
        num_cols=[c for c in X_raw.columns if X_raw[c].dtype != object],
        cat_cols=[c for c in X_raw.columns if X_raw[c].dtype == object],
    )
    X_full = pre_full.fit_transform(X_raw)
    in_dim = X_full.shape[1]
    dump(pre_full, os.path.join("artifacts", "preprocessor.joblib"))

    # 2) фінальна модель — довчимо коротко на всіх даних (можна без ранньої зупинки, невелике число епох)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim, hidden=hidden, drop=args.dropout).to(device)

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ds = TensorDataset(torch.tensor(X_full, dtype=torch.float32, device=device),
                       torch.tensor(y, dtype=torch.float32, device=device))
    dl = DataLoader(ds, batch_size=1024, shuffle=True)

    model.train()
    fin_epochs = max(3, min(10, args.epochs // 3))
    for ep in range(fin_epochs):
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    # Збережемо як TorchScript (для твоїх eval_* скриптів)
    scripted = torch.jit.script(model.cpu())
    torch.jit.save(scripted, os.path.join("artifacts", "mlp.pt"))

    # 3) метрики CV
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join("artifacts", "cv_metrics.csv"), index=False)

    # 4) OOF прогнози
    oof_df = pd.DataFrame({"y_true": y.astype(int), "proba": oof_prob})
    oof_df.to_csv(os.path.join("artifacts", "mlp_oof.csv"), index=False)

    # 5) маніфест
    manifest = {
        "created_utc": int(time.time()),
        "csv": os.path.abspath(args.csv),
        "rows": int(len(df)),
        "used_rows": int(len(X_raw)),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden": list(hidden),
        "dropout": float(args.dropout),
        "oof_pr_auc": float(oof_ap),
        "oof_best_f1": float(oof_f1),
        "oof_thr_star": float(oof_thr),
        "forbidden_note": "Label-like, IP/ID/timestamp, constants, high-cardinality and auto-leaky categorical cols were dropped.",
        "label_column": label_col,
    }
    with open(os.path.join("artifacts", "mlp_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nАртефакти → artifacts/mlp.pt, artifacts/preprocessor.joblib, "
          "artifacts/cv_metrics.csv, artifacts/mlp_oof.csv, artifacts/mlp_manifest.json")


if __name__ == "__main__":
    main()
