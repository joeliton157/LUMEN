# 🧠 LUMEN – Light-based Understanding Model for Explainable Networks

LUMEN é uma arquitetura de rede neural explicável desenvolvida para avaliação de risco médico com base em três domínios principais:

- **Genético**
- **Ambiental**
- **Fisiológico** (antigo "relativístico")

O diferencial está no uso de **softmax intermediários** em cada domínio, permitindo que cada vetor de risco seja interpretável separadamente, e **explicado com linguagem natural via LLM local** (como Ollama).

---

## 📁 Estrutura do Projeto

'''
├── data/                 # Dados de entrada (data.csv)
│   └── diagnosticos/     # Explicações salvas automaticamente
├── models/               # Modelos treinados (lumen.pt)
├── baseline_model.py     # Modelo convencional
├── lumen_model.py        # Arquitetura LUMEN
├── explain_lumen_llm.py  # Integração com LLM local para XAI
├── lumen_explain.ipynb   # Notebook visual para interpretação
├── generate_data.py      # Geração automática do data.csv
├── train.py              # Treinamento e avaliação dos modelos
├── requirements.txt      # Bibliotecas necessárias
└── README.md             # Documentação do projeto
'''

---

## 🚀 Como Executar

### 1. Instale as dependências

'''
pip install -r requirements.txt
'''

### 2. Gere os dados (caso não tenha 'data/data.csv')

'''
python generate_data.py
'''

### 3. Treine um modelo

'''
python train.py --model lumen
# ou
python train.py --model baseline
'''

### 4. Gere explicações automáticas

'''
python explain_lumen_llm.py
'''

---

## 📊 Resultados com Base Sintética (500.000 amostras)

| Modelo   | Acurácia | AUC    |
|----------|----------|--------|
| Baseline | 86.63%   | 0.9472 |
| **LUMEN**    | 86.51%   | **0.9474** |

---

## 📈 Explicabilidade (XAI)

Os vetores de risco são calculados com softmax por domínio:

- Vetor **genético** → interpretação de predisposição
- Vetor **ambiental** → exposição externa
- Vetor **fisiológico** → reflexo de expressões biológicas

Esses vetores são interpretados via **LLM local** (como Gemma, LLaMA, Codellama) para explicar a predição de forma humanizada.

---

## 📦 requirements.txt

'''txt
pandas
scikit-learn
torch
transformers
langchain
langchain-community
langchain-core
langchain-ollama
matplotlib
'''

---

## ⚠️ Aviso

> Este projeto foi realizado para fins de estudos. Não utilizar para diagnóstico real.

---

## ✨ Autor

Desenvolvido por Joeliton Victor
📍 Projeto LUMEN – IA explicável a serviço da saúde.
