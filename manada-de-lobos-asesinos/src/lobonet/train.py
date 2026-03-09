# lobonet/train.py

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from lobonet.model import LoboNet
from lobonet.config import INPUT_SIZE, EPOCHS, LEARNING_RATE

CKPT_PATH = Path(__file__).parent.parent.parent / "models" / "lobonet_model.pth"


def train():
    torch.manual_seed(42)

    # =========================
    # 1) Dataset sintético
    # =========================
    N = 1200
    X = torch.randn(N, INPUT_SIZE)

    # Etiqueta: 1 si suma de features > 0, si no 0 (binaria)
    y = (torch.sum(X, dim=1) > 0).float().unsqueeze(1)

    # Split simple train/test
    split = int(N * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # =========================
    # 2) Modelo + loss + optim
    # =========================
    model = LoboNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =========================
    # 3) Entrenamiento
    # =========================
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # =========================
    # 4) Accuracy rápido (TRAIN)
    # =========================
    model.eval()
    with torch.no_grad():
        probs_train = model(X_train)
        preds_train = (probs_train > 0.5).float()
        train_acc = (preds_train == y_train).float().mean().item()

    print(f"Train Accuracy: {train_acc*100:.2f}%")

    # =========================
    # 5) Evaluación en TEST
    # =========================
    with torch.no_grad():
        probs_test = model(X_test)
        preds_test = (probs_test > 0.5).float()
        test_acc = (preds_test == y_test).float().mean().item()

    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # =========================
    # 6) Guardar checkpoint
    # =========================
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": INPUT_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }, CKPT_PATH)

    print(f"Modelo guardado: {CKPT_PATH}")
    print("Entrenamiento finalizado.")


if __name__ == "__main__":
    train()