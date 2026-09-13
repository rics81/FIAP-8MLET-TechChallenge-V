# src/api.py
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODELS_DIR = Path("models")

app = FastAPI(title="TC5 - Recomendador de Oferta (Bandit)")

model_arm0 = joblib.load(MODELS_DIR / "model_arm0.joblib")   # cellular
model_arm1 = joblib.load(MODELS_DIR / "model_arm1.joblib")   # telephone
with open(MODELS_DIR / "columns.json") as f:
    TRAIN_COLUMNS = json.load(f)

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "month", "day_of_week", "poutcome",
]

ARM_NAMES = {0: "cellular", 1: "telephone"}


class Cliente(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float


def montar_features(cliente: Cliente) -> pd.DataFrame:
    raw = cliente.model_dump()
    # Pydantic não aceita "." no nome do campo; devolve aos nomes originais.
    raw["emp.var.rate"] = raw.pop("emp_var_rate")
    raw["cons.price.idx"] = raw.pop("cons_price_idx")
    raw["cons.conf.idx"] = raw.pop("cons_conf_idx")
    raw["nr.employed"] = raw.pop("nr_employed")

    df = pd.DataFrame([raw])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0)
    return df


@app.post("/recommend")
def recommend(cliente: Cliente):
    X = montar_features(cliente)

    prob_arm0 = model_arm0.predict_proba(X)[0][1]
    prob_arm1 = model_arm1.predict_proba(X)[0][1]

    braco_recomendado = 0 if prob_arm0 >= prob_arm1 else 1

    return {
        "oferta_recomendada": ARM_NAMES[braco_recomendado],
        "probabilidade_cellular": round(float(prob_arm0), 4),
        "probabilidade_telephone": round(float(prob_arm1), 4),
    }