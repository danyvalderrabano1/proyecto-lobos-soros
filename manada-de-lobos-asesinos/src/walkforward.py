# src/walkforward.py
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# Si tu proyecto usa torch, lo dejamos listo para integrar tu model.py
# pero por defecto esto hace evaluación temporal "agnóstica" al modelo.
# Luego conectamos tu train/evaluate real dentro de cada fold.

@dataclass
class WFConfig:
    date_col: str = "date"
    target_col: str = "target"
    min_train_size: int = 500
    test_size: int = 200
    step_size: int = 200


def load_processed(path: str) -> pd.DataFrame:
    # Soporta .csv o .parquet
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def find_processed_file(processed_dir: str) -> str:
    # Busca el primer archivo “razonable” dentro de data/processed
    candidates = []
    for root, _, files in os.walk(processed_dir):
        for f in files:
            if f.endswith(".csv") or f.endswith(".parquet"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"No encontré archivos .csv/.parquet en {processed_dir}")
    # Elige el más reciente
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def time_splits(n: int, min_train: int, test: int, step: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    train_end = min_train
    while train_end + test <= n:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test)
        splits.append((train_idx, test_idx))
        train_end += step
    return splits


def baseline_predict(y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    # Baseline simple: predice la clase mayoritaria (si es clasificación binaria)
    # o la media (si es regresión continua).
    # Detectamos si target parece binaria {0,1}.
    uniq = np.unique(y_train)
    if set(uniq).issubset({0, 1}):
        p = int(np.round(y_train.mean()))
        return np.full(len(x_test), p, dtype=int)
    else:
        m = float(np.mean(y_train))
        return np.full(len(x_test), m, dtype=float)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Métricas mínimas sin sklearn (para no depender)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    acc = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-file", default=None,
                    help="Ruta directa al CSV procesado (preferido sobre --processed-dir)")
    ap.add_argument("--processed-dir", default="data/processed",
                    help="Directorio con CSVs procesados (se usa si no se pasa --processed-file)")
    ap.add_argument("--out-metrics", default="reports/walkforward_metrics.json")
    ap.add_argument("--out-folds", default="reports/walkforward_folds.csv")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--target-col", default="target")
    ap.add_argument("--min-train-size", type=int, default=500)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--step-size", type=int, default=200)
    args = ap.parse_args()

    processed_file = args.processed_file if args.processed_file else find_processed_file(args.processed_dir)
    df = load_processed(processed_file)

    # Validaciones mínimas
    if args.target_col not in df.columns:
        raise ValueError(f"No existe target_col='{args.target_col}' en {processed_file}. Columnas: {list(df.columns)[:30]}...")

    # Orden temporal si hay columna fecha
    if args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
        df = df.sort_values(args.date_col).reset_index(drop=True)

    y = df[args.target_col].to_numpy()

    # X: todo menos date/target (baseline no lo usa, pero lo dejamos por estructura)
    drop_cols = [args.target_col]
    if args.date_col in df.columns:
        drop_cols.append(args.date_col)
    X = df.drop(columns=drop_cols, errors="ignore").to_numpy()

    splits = time_splits(len(df), args.min_train_size, args.test_size, args.step_size)
    if not splits:
        raise ValueError(
            f"No se pudieron crear splits. Dataset n={len(df)}. "
            f"Revisa min_train={args.min_train_size}, test={args.test_size}, step={args.step_size}"
        )

    fold_rows = []
    fold_metrics = []

    # Baseline temporal: mayoría/mean por fold
    for i, (tr, te) in enumerate(splits, start=1):
        y_tr, y_te = y[tr], y[te]
        X_te = X[te]

        y_pred = baseline_predict(y_tr, X_te)

        uniq = np.unique(y_tr)
        is_binary = set(np.unique(y)).issubset({0, 1})
        if is_binary:
            m = classification_metrics(y_te, y_pred)
        else:
            # regresión: MSE simple
            mse = float(np.mean((y_te - y_pred) ** 2))
            m = {"mse": mse}

        m["fold"] = i
        m["train_end"] = int(tr[-1])
        m["test_start"] = int(te[0])
        m["test_end"] = int(te[-1])
        fold_metrics.append(m)

        fold_rows.append({
            "fold": i,
            "train_size": len(tr),
            "test_size": len(te),
            **{k: v for k, v in m.items() if k not in {"fold"}}
        })

    # Agregados
    metrics_out = {
        "processed_file": processed_file,
        "n_rows": int(len(df)),
        "n_folds": int(len(splits)),
        "config": {
            "date_col": args.date_col,
            "target_col": args.target_col,
            "min_train_size": args.min_train_size,
            "test_size": args.test_size,
            "step_size": args.step_size,
        },
        "folds": fold_metrics,
    }

    # Promedios
    keys_numeric = [k for k in fold_metrics[0].keys() if k not in {"fold"} and isinstance(fold_metrics[0][k], (int, float))]
    avg = {}
    for k in keys_numeric:
        vals = [fm[k] for fm in fold_metrics if isinstance(fm.get(k), (int, float))]
        if vals:
            avg[f"avg_{k}"] = float(np.mean(vals))
    metrics_out["summary"] = avg

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    pd.DataFrame(fold_rows).to_csv(args.out_folds, index=False)

    print(f"[WALKFORWARD] metrics -> {args.out_metrics}")
    print(f"[WALKFORWARD] folds   -> {args.out_folds}")
    print(f"[WALKFORWARD] n_folds={len(splits)} processed={processed_file}")


if __name__ == "__main__":
    main()