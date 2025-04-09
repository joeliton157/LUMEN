import os
import numpy as np
import pandas as pd

# Gerar dataset sintético caso não exista
if not os.path.exists("data/data.csv"):
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    data = np.random.rand(500000, 24)
    target = np.random.randint(0, 2, 500000)
    df = pd.DataFrame(data, columns=[f"F{i}" for i in range(1, 25)])
    df["TARGET"] = target
    df.to_csv("data/data.csv", index=False)
    print("✅ Arquivo data.csv gerado automaticamente.")
else:
    print("✅ Arquivo data.csv já existe.")