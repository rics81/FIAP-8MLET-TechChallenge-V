# src/train_policy.py
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
arm_train = pd.read_csv(DATA_DIR / "arm_train.csv").squeeze("columns")

# Mesma lógica das Etapas 3/4: um modelo por braço, treinado só com as
# linhas históricas daquele braço.
model_arm0 = LogisticRegression(max_iter=1000)
model_arm0.fit(X_train[arm_train == 0], y_train[arm_train == 0])

model_arm1 = LogisticRegression(max_iter=1000)
model_arm1.fit(X_train[arm_train == 1], y_train[arm_train == 1])

joblib.dump(model_arm0, MODELS_DIR / "model_arm0.joblib")
joblib.dump(model_arm1, MODELS_DIR / "model_arm1.joblib")

# Salva a lista de colunas de X_train para a API saber montar o vetor de
# features na mesma ordem/formato usado no treino.
with open(MODELS_DIR / "columns.json", "w") as f:
    json.dump(list(X_train.columns), f)

print("Modelos salvos em models/")
print(f"model_arm0 (cellular): treinado com {(arm_train == 0).sum()} linhas")
print(f"model_arm1 (telephone): treinado com {(arm_train == 1).sum()} linhas")
print(f"colunas salvas em models/columns.json ({len(X_train.columns)} colunas)")