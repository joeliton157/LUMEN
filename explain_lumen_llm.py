import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from datetime import datetime
import os

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableSequence

# ====== LUMEN Model Definition ======
class LUMENModel(nn.Module):
    def __init__(self, input_dims=(10, 8, 6)):
        super(LUMENModel, self).__init__()
        self.gen_fc1 = nn.Linear(input_dims[0], 32)
        self.gen_fc2 = nn.Linear(32, 16)
        self.gen_softmax = nn.Softmax(dim=1)

        self.amb_fc1 = nn.Linear(input_dims[1], 32)
        self.amb_fc2 = nn.Linear(32, 16)
        self.amb_softmax = nn.Softmax(dim=1)

        self.rel_fc1 = nn.Linear(input_dims[2], 32)
        self.rel_fc2 = nn.Linear(32, 16)
        self.rel_softmax = nn.Softmax(dim=1)

        self.fusion_fc1 = nn.Linear(16 * 3, 64)
        self.fusion_fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x_gen, x_amb, x_rel):
        g = F.relu(self.gen_fc1(x_gen))
        g = F.relu(self.gen_fc2(g))
        g = self.gen_softmax(g)

        a = F.relu(self.amb_fc1(x_amb))
        a = F.relu(self.amb_fc2(a))
        a = self.amb_softmax(a)

        r = F.relu(self.rel_fc1(x_rel))
        r = F.relu(self.rel_fc2(r))
        r = self.rel_softmax(r)

        combined = torch.cat([g, a, r], dim=1)
        x = F.relu(self.fusion_fc1(combined))
        x = F.relu(self.fusion_fc2(x))
        x = torch.sigmoid(self.output(x))
        return x, g, a, r

# ====== Dados e modelo ======
df = pd.read_csv("data/data.csv")
X = df.drop("TARGET", axis=1).values
X_gen = torch.tensor(X[:, :10], dtype=torch.float32)
X_amb = torch.tensor(X[:, 10:18], dtype=torch.float32)
X_rel = torch.tensor(X[:, 18:24], dtype=torch.float32)

model = LUMENModel()
model.load_state_dict(torch.load("models/lumen.pt"))
model.eval()

# ====== Selecionar paciente ======
i = 0
xg = X_gen[i:i+1]
xa = X_amb[i:i+1]
xr = X_rel[i:i+1]

with torch.no_grad():
    y_pred, g_vec, a_vec, r_vec = model(xg, xa, xr)

# ====== Gerar prompt ======
def format_vector(name, vec):
    return ", ".join([f"{name}{i+1}: {v:.3f}" for i, v in enumerate(vec)])

full_prompt = f"""
Paciente ID: {i:03d}
Risco estimado: {y_pred.item()*100:.2f}%

Domínio Genético:
{format_vector("G", g_vec[0])}

Domínio Ambiental:
{format_vector("A", a_vec[0])}

Domínio Fisiológico:
{format_vector("R", r_vec[0])}

Com base nisso, responda:
1. Quais vetores mais influenciaram o risco final?
2. Qual domínio teve maior impacto?
3. Gere uma explicação médica clara e objetiva com base nesses fatores.
"""

# ====== LLM com LangChain ======
llm = Ollama(model="gemma3")

prompt = PromptTemplate.from_template("{question}")
chain: RunnableSequence = prompt | llm

print("\n🧠 Gerando explicação com Ollama local...")
response = chain.invoke({"question": full_prompt})

print("\n📋 Explicação gerada:")
print(response)

# Criar diretório se não existir
output_dir = "diagnosticos"
os.makedirs(output_dir, exist_ok=True)

# Nome do arquivo com timestamp e ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{output_dir}/paciente_{i:03d}_{timestamp}.txt"

# Salvar resposta
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"Paciente ID: {i:03d}\n")
    f.write(f"Risco estimado: {y_pred.item()*100:.2f}%\n\n")
    f.write("Domínio Genético:\n" + format_vector("G", g_vec[0]) + "\n\n")
    f.write("Domínio Ambiental:\n" + format_vector("A", a_vec[0]) + "\n\n")
    f.write("Domínio Fisiológico:\n" + format_vector("R", r_vec[0]) + "\n\n")
    f.write("Explicação:\n" + response)

print(f"\n💾 Diagnóstico salvo em: {filename}")

