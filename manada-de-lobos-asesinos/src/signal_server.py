"""
signal_server.py — Proyecto Lobo Alfa
Servidor ZeroMQ que recibe datos OHLCV de MT5 y devuelve señales del modelo.

Protocolo:
  MT5 (EA) → envía JSON con OHLCV → Python → responde con señal BUY/SELL/HOLD
"""

import zmq
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lobonet.features import build_features

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
PORT = 5555
MODELS_DIR = Path("models")
THRESHOLD_BUY  = 0.0002   # +0.02% para señal BUY
THRESHOLD_SELL = -0.0002  # -0.02% para señal SELL
MIN_BARS = 250            # mínimo de barras para calcular indicadores

# ─────────────────────────────────────────────
# CARGAR MODELOS
# ─────────────────────────────────────────────

def load_models() -> dict:
    """Carga todos los modelos entrenados."""
    models = {}
    for model_path in MODELS_DIR.glob("model_*.pkl"):
        pair = model_path.stem.replace("model_", "")
        with open(model_path, "rb") as f:
            models[pair] = pickle.load(f)
        print(f"  ✅ Modelo cargado: {pair}")
    return models


# ─────────────────────────────────────────────
# PROCESAR SEÑAL
# ─────────────────────────────────────────────

def process_request(data: dict, models: dict) -> dict:
    """
    Procesa una petición del EA y devuelve una señal.
    
    Entrada esperada:
    {
        "pair": "EURUSD",
        "bars": [
            {"time": "2024-01-01 00:00:00", "open": 1.1, "high": 1.101, 
             "low": 1.099, "close": 1.1005, "volume": 500},
            ...
        ]
    }
    
    Salida:
    {
        "signal": "BUY" | "SELL" | "HOLD",
        "prediction": 0.00025,
        "confidence": 0.73,
        "pair": "EURUSD",
        "timestamp": "2024-01-01 00:05:00"
    }
    """
    pair = data.get("pair", "EURUSD").upper()
    bars = data.get("bars", [])

    # Validaciones
    if pair not in models:
        return {"error": f"Modelo no disponible para {pair}", "signal": "HOLD"}
    
    if len(bars) < MIN_BARS:
        return {"error": f"Pocas barras: {len(bars)} < {MIN_BARS}", "signal": "HOLD"}

    try:
        # Convertir barras a DataFrame
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        df.columns = [c.lower() for c in df.columns]

        # Renombrar volume → tick_volume si necesario
        if "volume" in df.columns and "tick_volume" not in df.columns:
            df["tick_volume"] = df["volume"]

        # Aplicar features
        df_features = build_features(df, dropna=True)
        
        if len(df_features) == 0:
            return {"error": "Sin datos tras features", "signal": "HOLD"}

        # Tomar la última barra para predecir
        feature_cols = [c for c in df_features.columns 
                       if c not in {"open", "high", "low", "close", "tick_volume", 
                                    "spread", "real_volume", "volume", "target"}]
        
        X_last = df_features[feature_cols].iloc[[-1]]
        
        # Predecir
        model = models[pair]
        prediction = float(model.predict(X_last)[0])
        
        # Confianza (si el modelo soporta predict_proba o feature_importances)
        confidence = min(abs(prediction) / THRESHOLD_BUY, 1.0)

        # Señal
        if prediction > THRESHOLD_BUY:
            signal = "BUY"
        elif prediction < THRESHOLD_SELL:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "prediction": round(prediction, 8),
            "confidence": round(confidence, 4),
            "pair": pair,
            "timestamp": datetime.utcnow().isoformat(),
            "last_close": float(df["close"].iloc[-1]),
            "bars_used": len(df_features)
        }

    except Exception as e:
        return {"error": str(e), "signal": "HOLD"}


# ─────────────────────────────────────────────
# SERVIDOR PRINCIPAL
# ─────────────────────────────────────────────

def run_server():
    print(f"\n🐺 Lobo Alfa Signal Server")
    print(f"{'─'*40}")
    
    # Cargar modelos
    print("Cargando modelos...")
    models = load_models()
    
    if not models:
        print("❌ No se encontraron modelos en ./models/")
        print("   Corre primero: dvc repro")
        return

    print(f"\n✅ {len(models)} modelos cargados: {list(models.keys())}")
    
    # Iniciar servidor ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    
    print(f"\n🚀 Servidor escuchando en puerto {PORT}")
    print(f"   Thresholds: BUY > {THRESHOLD_BUY} | SELL < {THRESHOLD_SELL}")
    print(f"   Ctrl+C para detener\n")

    stats = {"total": 0, "buy": 0, "sell": 0, "hold": 0, "errors": 0}

    try:
        while True:
            # Esperar mensaje del EA
            message = socket.recv_string()
            
            try:
                data = json.loads(message)
                result = process_request(data, models)
            except json.JSONDecodeError:
                result = {"error": "JSON inválido", "signal": "HOLD"}
            except Exception as e:
                result = {"error": str(e), "signal": "HOLD"}

            # Enviar respuesta
            socket.send_string(json.dumps(result))
            
            # Estadísticas
            stats["total"] += 1
            signal = result.get("signal", "HOLD").lower()
            if signal in stats:
                stats[signal] += 1
            if "error" in result:
                stats["errors"] += 1
            
            # Log
            pair = data.get("pair", "?") if isinstance(data, dict) else "?"
            print(f"  [{datetime.utcnow().strftime('%H:%M:%S')}] {pair:8s} → "
                  f"{result['signal']:4s} | pred={result.get('prediction', 'N/A')} | "
                  f"conf={result.get('confidence', 'N/A')} | "
                  f"total={stats['total']} (B:{stats['buy']} S:{stats['sell']} H:{stats['hold']})")

    except KeyboardInterrupt:
        print(f"\n\n🛑 Servidor detenido.")
        print(f"📊 Estadísticas finales: {stats}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    run_server()