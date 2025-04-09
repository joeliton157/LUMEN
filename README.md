# LUMEN – Local Unified Model for Explainable Neurodiagnosis

LUMEN é uma arquitetura de rede neural explicável desenvolvida para avaliar riscos de doenças como o câncer a partir de três domínios principais: genético, ambiental e fisiológico (anteriormente denominado "relativístico"). A proposta central é integrar esses domínios por meio de sub-redes especializadas com ativações 'softmax', permitindo uma leitura interpretável dos vetores de risco.

---

## 🔬 Visão Geral do Projeto

- Modelagem baseada em sub-redes autônomas por domínio  
- Vetores intermediários com 'softmax' para interpretação direta  
- Integração com LLMs locais (Mistral, OpenChat, Zephyr, Deepseek) para explicação textual  
- Comparativo com modelo convencional (baseline) para validação de performance  

---

## 📊 Resultados Atuais (Base Sintética - 500k amostras)

| Modelo   | Acurácia | AUC    |
|----------|----------|--------|
| Baseline | 86.63%   | 0.9472 |
| LUMEN    | 86.51%   | 0.9474 |

---

## 📈 Explicabilidade (XAI)

O modelo LUMEN gera vetores intermediários que podem ser visualizados por domínio:

- Domínio Genético: identifica predisposições hereditárias  
- Domínio Ambiental: avalia exposições e influências do ambiente  
- Domínio Fisiológico: representa alterações metabólicas e expressões gênicas  

Esses vetores são usados como entrada para um LLM, que gera explicações automáticas em linguagem natural.

---

## 📚 Exemplo de Saída Explicada

> "O risco estimado é de 38.6%. Fatores fisiológicos como R2 e R12 mostraram forte influência, enquanto o domínio genético apresentou predisposição significativa em G2. O padrão sugere risco moderado com base em interações bioquímicas e herança genética."

---

## 📁 Estrutura do Projeto

'''
├── data/                 # Dados de entrada (data.csv)
├── models/               # Modelos treinados (lumen.pt)
├── baseline_model.py     # Modelo convencional
├── lumen_model.py        # Arquitetura LUMEN
├── train.py              # Treinamento e validação
├── explain_lumen_llm.py  # Integração com LLM para XAI
├── lumen_explain.ipynb   # Notebook visual para interpretação
└── README.md             # Documentação do projeto
'''

---

## 🚀 Como Rodar

### 1. Instale os requisitos
'''
pip install -r requirements.txt
'''

### 2. Treine o modelo
'''
python train.py --model lumen
'''

### 3. Execute a explicação com LLM local (necessário instalar o Ollama e ou rodar via Huggingface Transformers)
'''
python explain_lumen_llm.py
'''

---

## 📦 requirements.txt

'''
pandas
scikit-learn
torch
transformers
matplotlib
'''

---

## 📚 Licença

Projeto experimental desenvolvido para fins de pesquisa e avaliação. Não utilizar para diagnóstico real. Direitos reservados ao autor.

---

Desenvolvido por [Joeliton Victor] — projeto LUMEN: explicabilidade como aliada da decisão médica.
