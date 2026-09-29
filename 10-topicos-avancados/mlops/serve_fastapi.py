"""Servindo um modelo como API REST com FastAPI: validação de entrada (Pydantic), versão do modelo e health check.

pip install fastapi uvicorn scikit-learn joblib
Antes: python mlops_pipeline.py     (gera o artefato)
Rodar: uvicorn serve_fastapi:app --reload   -> http://127.0.0.1:8000/docs
Teste sem subir o servidor: python serve_fastapi.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTEFATOS = Path(__file__).parent / "artefatos"
versoes = sorted(p for p in ARTEFATOS.glob("*") if (p / "modelo.joblib").exists()) if ARTEFATOS.exists() else []
if not versoes:
    raise SystemExit("Nenhum artefato encontrado — rode primeiro: python mlops_pipeline.py")
pasta = versoes[-1]  # versão mais recente (em produção: um registry decide qual está "em produção")
modelo = joblib.load(pasta / "modelo.joblib")
meta = json.loads((pasta / "metadata.json").read_text())

app = FastAPI(title="Classificador de tumores", version=meta["versao"])


class Entrada(BaseModel):
    features: list[float] = Field(..., min_length=len(meta["features"]), max_length=len(meta["features"]))


@app.get("/saude")
def saude():
    return {"status": "ok", "versao_modelo": meta["versao"], "metricas": meta["metricas"]}


@app.post("/prever")
def prever(x: Entrada):
    if not np.all(np.isfinite(x.features)):
        raise HTTPException(422, "valores não finitos")
    p = float(modelo.predict_proba([x.features])[0, 1])
    return {"probabilidade_benigno": round(p, 4), "classe": int(p >= 0.5), "versao_modelo": meta["versao"]}


if __name__ == "__main__":  # teste rápido sem servidor
    from fastapi.testclient import TestClient
    from sklearn.datasets import load_breast_cancer

    cli = TestClient(app)
    print(cli.get("/saude").json())
    X, _ = load_breast_cancer(return_X_y=True)
    print(cli.post("/prever", json={"features": X[0].tolist()}).json())
    print("entrada inválida ->", cli.post("/prever", json={"features": [1.0, 2.0]}).status_code, "(422 = rejeitada pela validação)")
