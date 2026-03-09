#!/usr/bin/env python3
"""
preprocess.py — Convierte datos MT5 crudos en features para el pipeline DVC.

Features generadas:
  ret1        — retorno 1 barra
  ret3_mean   — media de retornos 3 barras
  ret10_mean  — media de retornos 10 barras
  ret10_std   — volatilidad 10 barras
  hl_range    — rango high-low normalizado (proxy volatilidad intra-barra)
  spread_norm — spread normalizado por close
  target      — retorno de la siguiente barra (variable a predecir)

Uso:
  python src/data/preprocess.py --symbol EURUSD
  python src/data/preprocess.py --symbol XAUUSD --raw-dir data/raw --out data/processed/processed.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    ret1 = c.pct_change(1)

    features = pd.DataFrame({
        "date":        df["date"],
        "ret1":        ret1,
        "ret3_mean":   ret1.rolling(3).mean(),
        "ret10_mean":  ret1.rolling(10).mean(),
        "ret10_std":   ret1.rolling(10).std(),
        "hl_range":    (df["high"] - df["low"]) / c,
        "spread_norm": df["spread"] / (c * 10_000),   # spread en pips / precio
        "target":      ret1.shift(-1),                 # retorno de la siguiente barra
    })

    # Elimina filas con NaN (primeras 10 por rolling + última por shift)
    features = features.dropna().reset_index(drop=True)
    return features


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preprocesa datos MT5 crudos para el pipeline DVC."
    )
    ap.add_argument("--symbol",  type=str, default="EURUSD",
                    help="Par a procesar (default: EURUSD)")
    ap.add_argument("--source",  type=str, default="mt5",
                    choices=["mt5", "tv"],
                    help="Fuente del CSV crudo (default: mt5)")
    ap.add_argument("--raw-dir", type=str, default="data/raw",
                    help="Directorio con CSVs crudos (default: data/raw)")
    ap.add_argument("--out",     type=str, default=None,
                    help="Ruta del CSV procesado de salida (default: data/processed/mt5_{symbol}_M5.csv)")
    args = ap.parse_args()

    raw_path = Path(args.raw_dir) / f"{args.source}_{args.symbol}_M5.csv"
    out_path  = Path(args.out) if args.out else Path("data/processed") / f"{args.source}_{args.symbol}_M5.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"No existe {raw_path}. "
            f"Ejecuta primero: python src/data/fetch_data.py --sources {args.source}"
        )

    print(f"[PREPROCESS] Leyendo {raw_path} …")
    df = pd.read_csv(raw_path, parse_dates=["date"])

    features = build_features(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    print(f"[PREPROCESS] symbol={args.symbol}  source={args.source}")
    print(f"[PREPROCESS] filas={len(features):,}  cols={list(features.columns)}")
    print(f"[PREPROCESS] rango: {features['date'].iloc[0]} -> {features['date'].iloc[-1]}")
    print(f"[PREPROCESS] salida: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
