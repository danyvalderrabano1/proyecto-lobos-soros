# src/train.py
import os
import argparse
import json

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# =========================
# CONFIGURACIÓN MLFLOW
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("lobo_ai_experiments")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento simple (Regresión) + MLflow logging")

    # ✅ default correcto (tu archivo existe: data/processed/processed.csv)
    parser.add_argument("--processed", type=str, default="data/processed/processed.csv",
                        help="Ruta al CSV procesado")

    # target y features
    parser.add_argument("--target", type=str, default="target",
                        help="Nombre de la columna target (numérica/continua)")
    parser.add_argument("--features", type=str, default="",
                        help="Lista de columnas features separadas por coma. Si vacío: usa todas menos target.")

    # split
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    # modelo
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)

    # outputs
    parser.add_argument("--out-model", type=str, default="models/model_rf.pkl")
    parser.add_argument("--out-metrics", type=str, default="reports/train_metrics_rf.json")

    return parser.parse_args()


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()

    # -------------------------
    # Cargar dataset
    # -------------------------
    if not os.path.exists(args.processed):
        raise FileNotFoundError(f"No existe el archivo: {args.processed}")

    df = pd.read_csv(args.processed)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' no existe en el CSV. Columnas: {list(df.columns)[:30]} ...")

    # -------------------------
    # --- LIMPIEZA / ENCODING AUTOMÁTICO ---
    # 1) Intentar convertir columnas tipo fecha (object) a datetime
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            # si al menos 90% no es NaT, lo consideramos fecha real
            if parsed.notna().mean() > 0.9:
                df[col] = parsed.view("int64") // 10**9  # segundos unix
        except Exception:
            pass

    # 2) One-hot para cualquier otra categórica que quede en object
    df = pd.get_dummies(df, drop_first=True)

    # -------------------------
    # Separar X / y
    # -------------------------
    y = df[args.target].astype(float)

    if args.features.strip():
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Features no encontradas: {missing}")
        X = df[feature_cols]
    else:
        X = df.drop(columns=[args.target])

    # Asegurar numéricos
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # -------------------------
    # Entrenamiento + MLflow
    # -------------------------
    with mlflow.start_run():
        # params
        mlflow.log_param("problem_type", "regression")
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("n_samples", int(len(X)))
        mlflow.log_param("processed_path", args.processed)

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)

        mlflow.log_metric("mse", float(mse))

        # log model en MLflow
        mlflow.sklearn.log_model(model, "model")

        # -------------------------
        # Guardar modelo + métricas
        # -------------------------
        ensure_dir(args.out_model)
        ensure_dir(args.out_metrics)

        # Guardado local (pickle)
        import pickle
        with open(args.out_model, "wb") as f:
            pickle.dump(model, f)

        metrics = {
            "mse": float(mse),
            "n_features": int(X.shape[1]),
            "n_samples": int(len(X)),
            "test_size": float(args.test_size)
        }
        with open(args.out_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"[TRAIN] model saved={args.out_model}")
        print(f"[TRAIN] metrics saved={args.out_metrics}")
        print(f"MSE: {mse}")


if __name__ == "__main__":
    main()