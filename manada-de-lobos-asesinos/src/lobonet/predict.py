import torch
import os

from lobonet.model import LoboNet
from lobonet.config import INPUT_SIZE


def predict_demo():

    # Crear modelo
    model = LoboNet()

    # Cargar checkpoint
    ckpt = torch.load("lobonet_model.pth", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Input reproducible
    torch.manual_seed(42)
    x = torch.randn(5, INPUT_SIZE)

    # Inferencia
    with torch.no_grad():
        y_pred = model(x)
        y_pred = y_pred.view(-1)

        probs = y_pred
        preds01 = (probs > 0.5).int()

    print("Input:")
    print(x)
    print("\nPredicciones (probabilidades):")
    print(probs)
    print("\nPredicciones (0/1):")
    print(preds01)


if __name__ == "__main__":
    predict_demo()