"""
features.py — Proyecto Lobo Alfa
Genera features técnicas sobre datos OHLCV de Forex/Gold.
Incluye: RSI, MACD, ATR, Bollinger Bands, EMA, Stochastic, CCI, ADX
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# UTILIDADES BASE
# ─────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


# ─────────────────────────────────────────────
# INDICADORES INDIVIDUALES
# ─────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    """RSI clásico de Wilder."""
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    """MACD línea, señal e histograma."""
    ema_fast = _ema(df[col], fast)
    ema_slow = _ema(df[col], slow)
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd_line"], signal)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df


def add_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> pd.DataFrame:
    """Bollinger Bands: upper, middle, lower y %B."""
    mid = _sma(df[col], period)
    std = df[col].rolling(window=period).std()
    df["bb_upper"] = mid + std_dev * std
    df["bb_mid"] = mid
    df["bb_lower"] = mid - std_dev * std
    # %B: posición dentro de las bandas (0=lower, 1=upper)
    band_width = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"] = (df[col] - df["bb_lower"]) / band_width.replace(0, np.nan)
    df["bb_width"] = band_width / mid  # Ancho normalizado
    return df


def add_ema_stack(
    df: pd.DataFrame, periods: list = [9, 21, 50, 200], col: str = "close"
) -> pd.DataFrame:
    """Stack de EMAs y sus posiciones relativas al precio."""
    for p in periods:
        df[f"ema_{p}"] = _ema(df[col], p)
    # Precio vs EMA rápida
    df["price_vs_ema9"] = (df[col] - df["ema_9"]) / df["ema_9"]
    df["price_vs_ema21"] = (df[col] - df["ema_21"]) / df["ema_21"]
    # Pendiente de EMAs (momentum)
    df["ema9_slope"] = df["ema_9"].diff(3) / df["ema_9"].shift(3)
    df["ema21_slope"] = df["ema_21"].diff(3) / df["ema_21"].shift(3)
    return df


def add_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator %K y %D."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_min) / denom
    df["stoch_d"] = _sma(df["stoch_k"], d_period)
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = _sma(tp, period)
    mean_dev = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    df["cci"] = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX — Average Directional Index (fuerza de tendencia)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de volumen (tick volume en Forex)."""
    if "tick_volume" not in df.columns:
        return df
    vol = df["tick_volume"].astype(float)
    df["vol_sma20"] = _sma(vol, 20)
    df["vol_ratio"] = vol / df["vol_sma20"].replace(0, np.nan)  # Spike de volumen
    df["vol_trend"] = vol.diff(5)  # Tendencia de volumen
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de velas (patrones básicos)."""
    body = (df["close"] - df["open"]).abs()
    total_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body_ratio"] = body / total_range  # Cuerpo vs total
    df["candle_direction"] = np.sign(df["close"] - df["open"])  # 1=alcista, -1=bajista
    df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / total_range
    df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / total_range
    return df


def add_return_features(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    """Returns y volatilidad realizada."""
    df["ret_1"] = df[col].pct_change(1)
    df["ret_3"] = df[col].pct_change(3)
    df["ret_12"] = df[col].pct_change(12)  # ~1 hora en M5
    df["ret_48"] = df[col].pct_change(48)  # ~4 horas en M5
    # Volatilidad realizada 20 periodos
    df["realized_vol"] = df["ret_1"].rolling(20).std()
    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de sesión de trading basadas en hora UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    hour = df.index.hour
    # Sesiones principales
    df["session_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)
    df["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    df["session_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    df["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)  # London+NY
    # Hora del día normalizada
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    # Día de semana
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    return df


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Aplica todos los features técnicos sobre un DataFrame OHLCV.

    Parámetros
    ----------
    df : DataFrame con columnas [open, high, low, close] y DatetimeIndex
    dropna : si True, elimina filas con NaN (necesario para ML)

    Retorna
    -------
    DataFrame enriquecido con todas las features
    """
    df = df.copy()

    # Renombrar columnas a minúsculas si vienen en mayúsculas
    df.columns = [c.lower() for c in df.columns]

    # Momentum / Tendencia
    df = add_rsi(df, period=14)
    df = add_rsi(df, period=7)   # RSI rápido adicional
    df = add_macd(df)
    df = add_ema_stack(df, periods=[9, 21, 50, 200])
    df = add_adx(df, period=14)

    # Volatilidad
    df = add_atr(df, period=14)
    df = add_bollinger_bands(df, period=20)

    # Osciladores
    df = add_stochastic(df)
    df = add_cci(df, period=20)

    # Volumen (si disponible)
    df = add_volume_features(df)

    # Velas
    df = add_candle_features(df)

    # Returns
    df = add_return_features(df)

    # Sesión
    df = add_session_features(df)

    if dropna:
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"  [features] Dropped {dropped} rows with NaN ({dropped/initial_len:.1%})")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Retorna lista de columnas que son features (excluye OHLCV base y target).
    """
    base_cols = {"open", "high", "low", "close", "tick_volume", "spread", "real_volume"}
    return [c for c in df.columns if c not in base_cols]


# ─────────────────────────────────────────────
# TEST RÁPIDO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, glob

    # Buscar primer CSV disponible
    files = glob.glob("data/processed/*.csv") + glob.glob("data/raw/**/*.csv", recursive=True)
    if not files:
        print("No CSV found. Generando datos sintéticos para test...")
        dates = pd.date_range("2024-01-01", periods=500, freq="5min")
        np.random.seed(42)
        price = 1.1 + np.random.randn(500).cumsum() * 0.0005
        df = pd.DataFrame({
            "open": price,
            "high": price + np.abs(np.random.randn(500)) * 0.0003,
            "low": price - np.abs(np.random.randn(500)) * 0.0003,
            "close": price + np.random.randn(500) * 0.0001,
            "tick_volume": np.random.randint(100, 1000, 500),
        }, index=dates)
    else:
        print(f"Cargando {files[0]}...")
        df = pd.read_csv(files[0], index_col=0, parse_dates=True)

    result = build_features(df)
    features = get_feature_columns(result)

    print(f"\n✅ Features generadas: {len(features)}")
    print(f"   Shape: {result.shape}")
    print(f"\n📋 Lista de features:")
    for i, f in enumerate(features, 1):
        print(f"   {i:2d}. {f}")

    print(f"\n📊 Estadísticas básicas (primeras 5 features):")
    print(result[features[:5]].describe().round(6))