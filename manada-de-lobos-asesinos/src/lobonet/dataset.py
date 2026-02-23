# lobonet/dataset.py

import torch
from typing import Tuple


def load_data(
    test_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Devuelve:
    X_train, y_train, X_test, y_test

    Dataset sintético temporal para que todo el pipeline funcione.
    Luego lo reemplazamos por datos reales de mercado.
    """

    torch.manual_seed(seed)

    from lobonet.config import INPUT_SIZE

    N = 2000

    # Features aleatorias
    X = torch.randn(N, INPUT_SIZE, dtype=torch.float32)

    # Target artificial (regla no lineal)
    y = (X[:, 0] * 0.7 - X[:, 1] * 0.4 + torch.sin(X[:, 2]) > 0.2).float()

    # Split train/test
    n_test = int(N * test_ratio)
    idx = torch.randperm(N)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test