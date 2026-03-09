from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import pickle
import time

import numpy as np
import pandas as pd

MODEL_PATH = Path("models") / "model.pkl"
DATA_PATH = Path("data") / "processed" / "processed.csv"
METRICS_PATH = Path("reports") / "metrics.json"
PRED_PATH = Path("reports") / "predictions.csv"

TARGET_COL = "target"
DATE_COL = "date"


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {MODEL_PATH.resolve()}\n"
            "Ejecuta primero: dvc repro train"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Datos no encontrados: {DATA_PATH.resolve()}\n"
            "Ejecuta primero: dvc repro synthetic"
        )
    df = pd.read_csv(DATA_PATH)
    drop_cols = [TARGET_COL] + ([DATE_COL] if DATE_COL in df.columns else [])
    X = df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[TARGET_COL].astype(float)
    dates = df[DATE_COL] if DATE_COL in df.columns else pd.Series(range(len(df)))
    return X, y, dates


def evaluate() -> int:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model()
    X, y, _ = _load_data()

    preds = model.predict(X)
    y_arr = y.to_numpy()

    mse = float(np.mean((y_arr - preds) ** 2))
    rmse = float(np.sqrt(mse))
    dir_acc = float(np.mean(np.sign(y_arr) == np.sign(preds)))

    metrics = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(MODEL_PATH.as_posix()),
        "n_samples": int(len(y_arr)),
        "mse": mse,
        "rmse": rmse,
        "direction_accuracy": dir_acc,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[EVAL] MSE={mse:.6f}  RMSE={rmse:.6f}  dir_acc={dir_acc:.4f}")
    print(f"[EVAL] Métricas guardadas en: {METRICS_PATH.resolve()}")
    return 0


def predict() -> int:
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model()
    X, y, dates = _load_data()

    preds = model.predict(X)

    rows = [
        {
            "date": date,
            "predicted_return": round(float(pred), 6),
            "actual_return": round(float(actual), 6),
            "signal": "BUY" if pred > 0 else "SELL",
        }
        for date, pred, actual in zip(dates, preds, y)
    ]

    with PRED_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "predicted_return", "actual_return", "signal"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[PREDICT] {len(rows)} predicciones guardadas en: {PRED_PATH.resolve()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--predict", action="store_true")
    args = ap.parse_args()

    if args.evaluate:
        return evaluate()
    if args.predict:
        return predict()

    print("Usa --evaluate o --predict")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())