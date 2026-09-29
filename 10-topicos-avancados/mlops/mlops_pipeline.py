"""Mini pipeline de MLOps de ponta a ponta: treino reprodutível -> portão de qualidade -> versionamento do artefato
com metadados -> carregar e servir -> monitorar deriva de dados -> decidir retreino.

pip install scikit-learn scipy joblib pandas numpy
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
ARTEFATOS = Path(__file__).parent / "artefatos"  # ignorado pelo git (ver .gitignore da raiz)
LIMITE_AUC = 0.95                                 # portão de qualidade para "promover" o modelo


def hash_dados(df: pd.DataFrame) -> str:
    """Impressão digital dos dados de treino: garante rastreabilidade (qual dado gerou qual modelo)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]


def treina_e_registra():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    params = dict(n_estimators=200, max_depth=6, random_state=SEED)
    modelo = make_pipeline(StandardScaler(), RandomForestClassifier(**params)).fit(X_tr, y_tr)  # pré-processamento DENTRO do artefato
    auc = roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1])
    print(f"AUC no teste = {auc:.4f}  (portão: >= {LIMITE_AUC})")
    if auc < LIMITE_AUC:
        raise SystemExit("modelo reprovado no portão de qualidade — não será promovido")

    versao = time.strftime("%Y%m%d-%H%M%S")
    pasta = ARTEFATOS / versao
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, pasta / "modelo.joblib")
    X_tr.to_csv(pasta / "referencia.csv", index=False)  # dados de referência p/ monitorar deriva depois
    meta = {
        "versao": versao, "metricas": {"auc_teste": round(auc, 4)}, "hiperparametros": params,
        "hash_dados_treino": hash_dados(X_tr), "n_treino": len(X_tr), "features": list(X.columns),
        "ambiente": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    (pasta / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"artefato registrado em {pasta.relative_to(Path(__file__).parent)}: modelo + metadata + dados de referência")
    return pasta


def carrega(pasta: Path):
    return joblib.load(pasta / "modelo.joblib"), pd.read_csv(pasta / "referencia.csv"), json.loads((pasta / "metadata.json").read_text())


def psi(ref, prod, bins=10):
    cortes = np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1])
    p = np.clip(np.bincount(np.searchsorted(cortes, ref), minlength=bins) / len(ref), 1e-4, None)
    q = np.clip(np.bincount(np.searchsorted(cortes, prod), minlength=bins) / len(prod), 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def monitora(ref: pd.DataFrame, prod: pd.DataFrame, nome: str):
    psis = {c: psi(ref[c].values, prod[c].values) for c in ref.columns}
    pior = sorted(psis.items(), key=lambda kv: -kv[1])[:3]
    frac_ks = np.mean([ks_2samp(ref[c], prod[c]).pvalue < 0.01 for c in ref.columns])
    alerta = max(psis.values()) > 0.25
    print(f"[{nome}] maior PSI: {[(c, round(v, 2)) for c, v in pior]} | {frac_ks:.0%} das features com KS p<0.01 -> "
          f"{'DERIVA: acionar retreino/investigação' if alerta else 'estável'}")


if __name__ == "__main__":
    pasta = treina_e_registra()
    modelo, ref, meta = carrega(pasta)
    print("metadata:", {k: meta[k] for k in ("versao", "metricas", "hash_dados_treino")})

    # produção simulada: (a) mesma distribuição; (b) sensor descalibrado (uma família de features desloca +30%)
    prod_ok = ref.sample(300, random_state=1).reset_index(drop=True)
    prod_drift = prod_ok.copy()
    prod_drift[[c for c in prod_drift.columns if c.startswith("mean")]] *= 1.3
    monitora(ref, prod_ok, "produção normal ")
    monitora(ref, prod_drift, "produção c/ deriva")
    print("previsões (fração de 'benigno'): normal=%.2f  com deriva=%.2f" % (modelo.predict(prod_ok).mean(), modelo.predict(prod_drift).mean()))
