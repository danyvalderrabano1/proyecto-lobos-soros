import os
import glob
import argparse
import pandas as pd


def _read_csv_robusto(path: str) -> pd.DataFrame:
    """
    Lee CSV intentando varios encodings y separadores comunes.
    Evita el clásico: UnicodeDecodeError / CSV "garbage" con 2 cols raras.
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "utf-16"]
    seps = [",", ";", "\t"]

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)

                # Si salió vacío, no sirve
                if df.shape[0] == 0:
                    continue

                # Heurística: si sale basura típica tipo ["��d", "Unnamed: 1"] con muy pocas filas
                # lo consideramos mala lectura.
                cols = [str(c) for c in df.columns]
                if ("Unnamed: 0" in cols) and (len(cols) <= 2):
                    continue
                if any("��" in c for c in cols):
                    continue

                return df
            except Exception as e:
                last_err = e

    raise RuntimeError(f"No pude leer CSV de forma robusta: {path}\nÚltimo error: {last_err}")


def preprocess():
    """
    Mínimo viable:
    - toma el CSV más reciente en data/raw
    - limpieza mínima
    - intenta detectar/parsing de fecha
    - guarda data/processed/processed.csv
    """
    os.makedirs("data/processed", exist_ok=True)

    raw_candidates = glob.glob("data/raw/**/*.csv", recursive=True)
    if not raw_candidates:
        raise FileNotFoundError("No encontré CSV en data/raw. Mete al menos un archivo .csv ahí.")

    raw_candidates.sort(key=os.path.getmtime, reverse=True)
    raw_path = raw_candidates[0]

    df = _read_csv_robusto(raw_path)

    # limpieza mínima
    df = df.dropna().reset_index(drop=True)

    # intenta detectar fecha
    date_col = None
    for c in ["date", "time", "timestamp", "datetime"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})

    # Validación mínima para walkforward
    # Necesitas 'date' y 'target' para el walkforward actual.
    # Si tu CSV real no trae 'target' aún, está bien: luego lo generamos en el preprocess real.
    out_path = "data/processed/processed.csv"
    df.to_csv(out_path, index=False)

    with open("data/processed/preprocess_ok.txt", "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[PREPROCESS] raw: {raw_path}")
    print(f"[PREPROCESS] out: {out_path} rows={len(df)} cols={len(df.columns)}")

    if "date" in df.columns:
        print(f"[PREPROCESS] date range: {df['date'].min()} -> {df['date'].max()}")


def generate_synthetic(n=3000):
    """
    Genera dataset sintético con features de retornos rezagados:
    columnas: date, ret1, ret3_mean, ret10_mean, ret10_std, target
    """
    import numpy as np

    os.makedirs("data/processed", exist_ok=True)

    # Serie temporal tipo mercado (1H)
    dates = pd.date_range("2020-01-01", periods=n, freq="H")

    # Random walk controlado (simula precio)
    price = 100 + np.cumsum(np.random.normal(0, 0.2, n))
    ret = pd.Series(price).pct_change()

    df = pd.DataFrame({
        "date": dates,
        "ret1": ret,
        "ret3_mean": ret.rolling(3).mean(),
        "ret10_mean": ret.rolling(10).mean(),
        "ret10_std": ret.rolling(10).std(),
        "target": ret.shift(-1),
    }).dropna().reset_index(drop=True)

    out_path = "data/processed/processed.csv"
    df.to_csv(out_path, index=False)

    with open("data/processed/preprocess_ok.txt", "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[SYNTHETIC] rows={len(df)} cols={list(df.columns)} saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preprocess", "synthetic"], help="Stage to run")
    args = parser.parse_args()

    if args.stage == "preprocess":
        preprocess()
    elif args.stage == "synthetic":
        generate_synthetic()


if __name__ == "__main__":
    main()