# src/lobonet/model.py

import torch
import torch.nn as nn

from lobonet.config import INPUT_SIZE


class LoboNet(nn.Module):
    """
    MLP simple para clasificación binaria.
    Salida en rango [0,1] usando sigmoid.

    Parámetros:
      - input_size: número de features de entrada
      - hidden_size: tamaño de la capa oculta base
    """

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 32):
        super().__init__()

        # Capa 1
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Capa 2 (reducimos a la mitad, mínimo 8 para no colapsar)
        hidden2 = max(8, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size, hidden2)

        # Capa 3 (salida binaria)
        self.fc3 = nn.Linear(hidden2, 1)

        self.act = nn.ReLU()
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(self.fc3(x))
        return x