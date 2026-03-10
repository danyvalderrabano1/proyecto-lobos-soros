"""
tune.py — Proyecto Lobo Alfa
Optimización de hiperparámetros con Optuna.
Busca los mejores parámetros para RandomForest por par.
"""

import optuna
import pickle
import numpy as np
import pandas as pd
import yaml
import json
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_params(params_file: str = "params.yaml") -> dict:
    with open(params_file, "r") as f:
        return yaml.safe_load(f)


def load_data(pair: str, processed_dir: str = "data/processed") -> tuple:
    """Carga datos procesados para el par."""
    candidates = list(Path(processed_dir).glob(f"*{pair}*features*.csv")) + \
                 list(Path(processed_dir).glob(f"*{pair}*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No se encontró data procesada para {pair}")

    df = pd.read_csv(candidates[0], index_col=0, parse_dates=True)

    base_cols = {"open", "high", "low", "close", "tick_volume",
                 "spread", "real_volume", "volume", "target"}
    feature_cols = [c for c in df.columns if c not in base_cols]

    X = df[feature_cols].values
    y = df["target"].values

    return X, y, feature_cols


def objective_rf(trial, X, y, n_splits=5):
    """Función objetivo para RandomForest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "n_jobs": -1,
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=12)
    mse_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, preds))

    return np.mean(mse_scores)


def objective_gbm(trial, X, y, n_splits=5):
    """Función objetivo para GradientBoosting."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=12)
    mse_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, preds))

    return np.mean(mse_scores)


def tune_pair(pair: str, params: dict) -> dict:
    """Optimiza hiperparámetros para un par."""
    n_trials = params.get("tune", {}).get("n_trials", 50)
    processed_dir = params.get("data", {}).get("processed_dir", "data/processed")
    models_dir = Path(params.get("data", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*50}")
    print(f"🔍 Optimizando: {pair} | {n_trials} trials")

    X, y, feature_cols = load_data(pair, processed_dir)
    print(f"   Data: {X.shape[0]:,} barras × {X.shape[1]} features")

    results = {}

    # ── Optimizar RandomForest ──
    print(f"   [1/2] RandomForest...")
    study_rf = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_rf.optimize(lambda trial: objective_rf(trial, X, y),
                      n_trials=n_trials,
                      show_progress_bar=False)

    best_rf_params = study_rf.best_params
    best_rf_mse = study_rf.best_value
    print(f"   ✅ RF mejor MSE: {best_rf_mse:.4e} | params: {best_rf_params}")

    # ── Optimizar GBM ──
    print(f"   [2/2] GradientBoosting...")
    study_gbm = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_gbm.optimize(lambda trial: objective_gbm(trial, X, y),
                       n_trials=n_trials,
                       show_progress_bar=False)

    best_gbm_params = study_gbm.best_params
    best_gbm_mse = study_gbm.best_value
    print(f"   ✅ GBM mejor MSE: {best_gbm_mse:.4e} | params: {best_gbm_params}")

    # ── Seleccionar mejor modelo ──
    if best_rf_mse <= best_gbm_mse:
        winner = "RandomForest"
        best_params = best_rf_params
        best_mse = best_rf_mse
        best_model = RandomForestRegressor(**best_rf_params, n_jobs=-1, random_state=42)
    else:
        winner = "GradientBoosting"
        best_params = best_gbm_params
        best_mse = best_gbm_mse
        best_model = GradientBoostingRegressor(**best_gbm_params, random_state=42)

    print(f"\n   🏆 Ganador: {winner} | MSE: {best_mse:.4e}")

    # Entrenar modelo final con todos los datos
    best_model.fit(X, y)

    # Guardar modelo
    model_path = models_dir / f"model_{pair}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"   💾 Modelo guardado: {model_path}")

    results = {
        "pair": pair,
        "winner": winner,
        "best_mse": best_mse,
        "best_params": best_params,
        "rf_mse": best_rf_mse,
        "gbm_mse": best_gbm_mse,
        "n_features": X.shape[1],
        "n_bars": X.shape[0],
    }

    # Guardar métricas
    metrics_path = Path("reports") / f"tune_metrics_{pair}.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None,
                        help="Par específico (ej: EURUSD). Si no, optimiza todos.")
    args = parser.parse_args()

    params = load_params()
    all_pairs = params.get("data", {}).get("pairs", ["EURUSD"])

    pairs = [args.symbol] if args.symbol else all_pairs

    print(f"\n🐺 Lobo Alfa — Optimización con Optuna")
    print(f"   Pares: {pairs}")
    print(f"   Trials por par: {params.get('tune', {}).get('n_trials', 50)}")

    all_results = []
    for pair in pairs:
        try:
            result = tune_pair(pair, params)
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error en {pair}: {e}")
            all_results.append({"pair": pair, "error": str(e)})

    # Resumen final
    print(f"\n{'═'*50}")
    print("📊 RESUMEN OPTIMIZACIÓN")
    print(f"{'═'*50}")

    summary_df = pd.DataFrame([r for r in all_results if "error" not in r])
    if not summary_df.empty:
        cols = ["pair", "winner", "best_mse", "rf_mse", "gbm_mse"]
        available = [c for c in cols if c in summary_df.columns]
        print(summary_df[available].to_string(index=False))

    # Guardar resumen
    summary_path = Path("reports") / "tune_summary.csv"
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    print(f"\n✅ Resumen guardado: {summary_path}")


if __name__ == "__main__":
    main()