#!/usr/bin/env python3
"""
fetch_data.py — Descarga datos de mercado desde MT5, tvdatafeed y FRED.

Uso:
    python src/data/fetch_data.py
    python src/data/fetch_data.py --sources mt5 fred
    python src/data/fetch_data.py --start 2024-01-01 --sources tv
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Configuración ─────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")

FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "USDCHF",
    "USDJPY", "EURJPY", "EURGBP", "USDCAD", "XAUUSD",
]

# Tickers FRED y nombre legible del archivo de salida
FRED_SERIES = {
    "DXY":   "DTWEXBGS",   # Trade Weighted Dollar Index (proxy DXY)
    "FED":   "FEDFUNDS",   # Fed Funds Rate efectivo
    "VIX":   "VIXCLS",     # CBOE Volatility Index
    "US10Y": "DGS10",      # Rendimiento Tesoro 10Y
}

# Exchange de TradingView por símbolo
_TV_EXCHANGE = {
    "EURUSD": "FX_IDC", "GBPUSD": "FX_IDC", "AUDUSD": "FX_IDC",
    "USDCHF": "FX_IDC", "USDJPY": "FX_IDC", "EURJPY": "FX_IDC",
    "EURGBP": "FX_IDC", "USDCAD": "FX_IDC",
    "XAUUSD": "OANDA",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── MetaTrader5 ────────────────────────────────────────────────────────────────

def fetch_mt5(start: datetime, out_dir: Path) -> list[str]:
    """
    Descarga barras M5 desde MetaTrader5.
    Requiere terminal MT5 abierta y cuenta conectada.
    Silencioso si el paquete no está instalado o el terminal no responde.
    """
    saved = []

    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.warning("[MT5] Paquete 'MetaTrader5' no instalado — omitiendo fuente.")
        log.warning("[MT5] Instala con: pip install MetaTrader5")
        return saved

    if not mt5.initialize():
        err = mt5.last_error()
        log.warning(f"[MT5] No se pudo inicializar: {err}")
        log.warning("[MT5] Verifica que el terminal esté abierto y una cuenta conectada.")
        return saved

    log.info("[MT5] Terminal inicializado.")
    start_utc = start.replace(tzinfo=timezone.utc)
    end_utc = datetime.now(timezone.utc)

    try:
        for symbol in FOREX_PAIRS:
            log.info(f"[MT5] {symbol} M5 desde {start_utc.date()} …")

            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_utc, end_utc)

            if rates is None or len(rates) == 0:
                log.warning(f"[MT5] Sin datos para {symbol}: {mt5.last_error()}")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"time": "date", "tick_volume": "volume"})
            df = df[["date", "open", "high", "low", "close", "volume", "spread"]]

            out_path = out_dir / f"mt5_{symbol}_M5.csv"
            df.to_csv(out_path, index=False)
            log.info(f"[MT5] {symbol}: {len(df):,} barras -> {out_path.name}")
            saved.append(str(out_path))

    finally:
        mt5.shutdown()
        log.info("[MT5] Terminal cerrado.")

    return saved


# ── tvdatafeed ─────────────────────────────────────────────────────────────────

def _estimate_bars(start: datetime) -> int:
    """Barras M5 aproximadas desde start (forex: 288 barras/día, 5 de 7 días)."""
    days = (datetime.now() - start.replace(tzinfo=None)).days
    return max(1, int(days * 288 * 5 / 7))


def fetch_tvdatafeed(start: datetime, out_dir: Path) -> list[str]:
    """
    Descarga barras M5 desde TradingView via tvdatafeed (sesión anónima).
    Usado como fuente de confirmación/alternativa a MT5.
    """
    saved = []

    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        log.warning("[TV] Paquete 'tvdatafeed' no instalado — omitiendo fuente.")
        log.warning("[TV] Instala con: pip install git+https://github.com/rongardF/tvdatafeed.git")
        return saved

    try:
        tv = TvDatafeed()
    except Exception as e:
        log.warning(f"[TV] No se pudo crear sesión: {e}")
        return saved

    n_bars = _estimate_bars(start)
    log.info(f"[TV] Solicitando ~{n_bars:,} barras por símbolo (desde {start.date()}).")
    start_ts = pd.Timestamp(start, tz="UTC")

    for symbol in FOREX_PAIRS:
        exchange = _TV_EXCHANGE[symbol]
        log.info(f"[TV] {symbol} ({exchange}) M5 …")

        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.in_5_minute,
                n_bars=n_bars,
            )
        except Exception as e:
            log.warning(f"[TV] Error en {symbol}: {e}")
            continue

        if df is None or df.empty:
            log.warning(f"[TV] Sin datos para {symbol}.")
            continue

        df = df.reset_index().rename(columns={"datetime": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df[df["date"] >= start_ts].reset_index(drop=True)
        df = df[["date", "open", "high", "low", "close", "volume"]]

        out_path = out_dir / f"tv_{symbol}_M5.csv"
        df.to_csv(out_path, index=False)
        log.info(f"[TV] {symbol}: {len(df):,} barras -> {out_path.name}")
        saved.append(str(out_path))

    return saved


# ── FRED ───────────────────────────────────────────────────────────────────────

def fetch_fred(start: datetime, out_dir: Path) -> list[str]:
    """
    Descarga series macro desde FRED via fredapi:
      - DXY   -> DTWEXBGS  (Trade Weighted Dollar Index, proxy DXY)
      - FED   -> FEDFUNDS  (Fed Funds Rate efectivo)
      - VIX   -> VIXCLS    (CBOE Volatility Index)
      - US10Y -> DGS10     (Rendimiento Tesoro 10Y)

    Requiere variable de entorno FRED_API_KEY.
    Clave gratuita en: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    import os
    saved = []

    try:
        from fredapi import Fred
    except ImportError:
        log.warning("[FRED] Paquete 'fredapi' no instalado — omitiendo fuente.")
        log.warning("[FRED] Instala con: pip install fredapi")
        return saved

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        log.warning("[FRED] Variable FRED_API_KEY no definida — omitiendo fuente.")
        log.warning("[FRED] Obtén una clave gratuita en: https://fred.stlouisfed.org/docs/api/api_key.html")
        log.warning("[FRED] Luego ejecuta: set FRED_API_KEY=tu_clave  (o export en Linux/Mac)")
        return saved

    fred = Fred(api_key=api_key)

    for name, ticker in FRED_SERIES.items():
        log.info(f"[FRED] {name} ({ticker}) …")
        try:
            series = fred.get_series(ticker, observation_start=start)
        except Exception as e:
            log.warning(f"[FRED] Error en {name} ({ticker}): {e}")
            continue

        df = (
            series.reset_index()
            .rename(columns={"index": "date", 0: name})
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna().reset_index(drop=True)

        out_path = out_dir / f"fred_{name}.csv"
        df.to_csv(out_path, index=False)
        log.info(f"[FRED] {name}: {len(df):,} observaciones -> {out_path.name}")
        saved.append(str(out_path))

    return saved


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Descarga datos de mercado desde MT5, tvdatafeed y FRED."
    )
    ap.add_argument(
        "--sources", nargs="+",
        choices=["mt5", "tv", "fred"], default=["mt5", "tv", "fred"],
        help="Fuentes a usar (default: todas)",
    )
    ap.add_argument(
        "--start", type=str, default="2023-01-01",
        help="Fecha inicio ISO (default: 2023-01-01)",
    )
    ap.add_argument(
        "--out-dir", type=str, default="data/raw",
        help="Directorio de salida (default: data/raw)",
    )
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = set(args.sources)
    all_saved: list[str] = []

    if "mt5" in sources:
        all_saved += fetch_mt5(start, out_dir)
    if "tv" in sources:
        all_saved += fetch_tvdatafeed(start, out_dir)
    if "fred" in sources:
        all_saved += fetch_fred(start, out_dir)

    log.info("─" * 50)
    if all_saved:
        log.info(f"Archivos guardados ({len(all_saved)}):")
        for p in all_saved:
            log.info(f"  {p}")
        return 0
    else:
        log.error("No se guardó ningún archivo. Revisa los warnings anteriores.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
