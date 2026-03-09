# lobonet/predict.py

from pathlib import Path
import torch

from lobonet.model import LoboNet
from lobonet.config import INPUT_SIZE

CKPT_PATH = Path(__file__).parent.parent.parent / "models" / "lobonet_model.pth"


def predict_demo():

    # 1️⃣ Crear modelo
    model = LoboNet()

    # 2️⃣ Verificar que exista el checkpoint
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No existe '{CKPT_PATH}'. Ejecuta primero:\n"
            f"python -m lobonet.train"
        )

    # 3️⃣ Cargar modelo entrenado
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")

    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"El checkpoint no contiene 'model_state_dict'. "
            f"Revisa cómo se guarda en train.py."
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4️⃣ Input reproducible (debug estable)
    torch.manual_seed(42)
    x = torch.randn(5, INPUT_SIZE)

    # 5️⃣ Inferencia
    with torch.no_grad():
        y_pred = model(x)
        y_pred = y_pred.view(-1)

        probs = y_pred
        preds01 = (probs > 0.5).int()

    # 6️⃣ Output
    print("Input:")
    print(x)

    print("\nPredicciones (probabilidades):")
    print(probs)

    print("\nPredicciones (0/1):")
    print(preds01)


if __name__ == "__main__":
    predict_demo()