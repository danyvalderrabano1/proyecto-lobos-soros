# src/lobonet/tune.py

import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
import mlflow

from lobonet.model import LoboNet
from lobonet.config import INPUT_SIZE
from lobonet.dataset import load_data


# =========================
# Config base
# =========================
DEVICE = "cpu"
EXPERIMENT_NAME = "LoboNet_Optuna_Tune"

SEED = 42
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# MLflow: fija tracking a TU proyecto (src/mlruns)
# tune.py está en: src/lobonet/tune.py
# dirname(dirname(__file__)) => src/
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../src
TRACKING_DIR = os.path.join(PROJECT_ROOT, "mlruns")

# File URI windows-safe
tracking_uri = "file:///" + TRACKING_DIR.replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)


# =========================
# Helpers
# =========================
def _to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)

def make_loader(X, y, batch_size: int, shuffle: bool):
    X_t = _to_tensor(X, torch.float32)
    y_t = _to_tensor(y, torch.float32).view(-1, 1)  # (N,1)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        pred = model(xb)  # salida ~ [0,1] por sigmoid en tu model
        loss = loss_fn(pred, yb)

        total_loss += float(loss.item()) * xb.size(0)
        preds_bin = (pred >= 0.5).float()
        correct += int((preds_bin == yb).sum().item())
        total += xb.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# =========================
# Optuna objective: 1 trial = 1 run (nested)
# =========================
def train_one_trial(trial: optuna.Trial):
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):

        # --- hiperparámetros
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        epochs = trial.suggest_int("epochs", 5, 30)

        mlflow.log_params({
            "lr": lr,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "device": DEVICE,
            "input_size": int(INPUT_SIZE) if isinstance(INPUT_SIZE, (int, np.integer)) else str(INPUT_SIZE),
        })

        # --- data
        X_train, y_train, X_val, y_val = load_data()
        train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

        # --- model
        model = LoboNet(INPUT_SIZE, hidden_size).to(DEVICE)

        # --- loss/opt
        loss_fn = nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_acc = -1.0
        best_val_loss = 1e9

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            n = 0

            for xb, yb in train_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

                running += float(loss.item()) * xb.size(0)
                n += xb.size(0)

            train_loss = running / max(n, 1)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn)

            # log por epoch (opcional pero útil)
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)
            mlflow.log_metric("val_acc", float(val_acc), step=epoch)

            # Optuna pruning (si quieres)
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # log final del trial
        mlflow.log_metric("best_val_acc", float(best_val_acc))
        mlflow.log_metric("best_val_loss", float(best_val_loss))

        return best_val_acc


# =========================
# Main: run padre (estudio)
# =========================
def main():
    seed_everything(SEED)

    n_trials = 30  # cámbialo a 50/100 cuando quieras
    sampler = optuna.samplers.TPESampler(seed=SEED)

    with mlflow.start_run(run_name="optuna_study"):
        mlflow.log_params({
            "n_trials": n_trials,
            "sampler": "TPE",
            "seed": SEED,
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment": EXPERIMENT_NAME,
        })

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(train_one_trial, n_trials=n_trials)

        # Best summary
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_value", float(study.best_value))

        print("\n=== DONE ===")
        print("Best value:", study.best_value)
        print("Best params:", study.best_params)
        print("MLflow tracking uri:", mlflow.get_tracking_uri())
        print("Experiment:", EXPERIMENT_NAME)


if __name__ == "__main__":
    main()