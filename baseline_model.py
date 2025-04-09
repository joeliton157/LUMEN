# baseline_model.py
import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, input_dim=24):
        super(BaselineModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
