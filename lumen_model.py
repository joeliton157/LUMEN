# lumen_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LUMENModel(nn.Module):
    def __init__(self, input_dims=(10, 8, 6)):
        super(LUMENModel, self).__init__()

        # Sub-rede genética
        self.gen_fc1 = nn.Linear(input_dims[0], 32)
        self.gen_fc2 = nn.Linear(32, 16)
        self.gen_softmax = nn.Softmax(dim=1)

        # Sub-rede ambiental
        self.amb_fc1 = nn.Linear(input_dims[1], 32)
        self.amb_fc2 = nn.Linear(32, 16)
        self.amb_softmax = nn.Softmax(dim=1)

        # Sub-rede relativística
        self.rel_fc1 = nn.Linear(input_dims[2], 32)
        self.rel_fc2 = nn.Linear(32, 16)
        self.rel_softmax = nn.Softmax(dim=1)

        # Fusão dos vetores de risco intermediários
        self.fusion_fc1 = nn.Linear(16 * 3, 64)
        self.fusion_fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x_gen, x_amb, x_rel):
        # Genético
        g = F.relu(self.gen_fc1(x_gen))
        g = F.relu(self.gen_fc2(g))
        g = self.gen_softmax(g)

        # Ambiental
        a = F.relu(self.amb_fc1(x_amb))
        a = F.relu(self.amb_fc2(a))
        a = self.amb_softmax(a)

        # Relativístico
        r = F.relu(self.rel_fc1(x_rel))
        r = F.relu(self.rel_fc2(r))
        r = self.rel_softmax(r)

        # Concatenar os vetores
        combined = torch.cat([g, a, r], dim=1)

        # Fusão e saída
        x = F.relu(self.fusion_fc1(combined))
        x = F.relu(self.fusion_fc2(x))
        x = torch.sigmoid(self.output(x))
        return x
