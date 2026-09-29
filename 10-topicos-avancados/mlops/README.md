# MLOps

Engenharia para levar modelos de ML **do notebook à produção** e mantê-los funcionando: DevOps + dados + modelos. Um modelo em produção é um sistema que **degrada com o tempo** (o mundo muda), não um artefato estático.

## Ciclo de vida

1. **Dados**: coleta, validação de esquema/qualidade, versionamento (DVC, lakeFS, Delta), *feature store* (Feast) para reutilizar features idênticas em treino e serviço.
2. **Experimentos**: rastrear código, dados, hiperparâmetros, métricas e artefatos (MLflow, Weights & Biases) — **reprodutibilidade** (sementes, ambientes fixos, contêineres).
3. **Treino** automatizado em *pipelines* (Airflow, Kubeflow, Prefect, Dagster).
4. **Validação e portões de qualidade** antes de promover: métricas mínimas, testes em fatias (subgrupos), testes de comportamento/robustez, comparação com o modelo atual, análise de vieses.
5. **Registro de modelos** (*model registry*): versões, estágio (staging/produção), metadados, linhagem.
6. **Implantação**: API online (REST/gRPC), *batch*, *streaming*, borda/dispositivo. Estratégias seguras: **shadow**, **canário**, **A/B**, *rollback*.
7. **Monitoramento** e **retreino** (gatilhos por tempo, deriva ou queda de desempenho) → volta ao passo 1.

## Riscos específicos de ML

- **Training–serving skew**: pré-processamento diferente no treino e na produção → empacote tudo no artefato (`Pipeline`) ou use uma *feature store*.
- **Vazamento de dados** e testes que não refletem produção (ver [validação](../../01-fundamentos-ml/validacao-cruzada/)).
- **Deriva** (*drift*):
  - *Data/covariate drift*: $P(X)$ muda (novo perfil de usuário, sensor descalibrado);
  - *Concept drift*: $P(Y\mid X)$ muda (fraude evolui, pandemia);
  - *Label/prior drift*: proporção das classes muda.
  Detecção: **PSI**, teste **KS**, divergências (KL/JS), monitoramento das previsões e, quando chegam os rótulos (com atraso), do desempenho real.
- **Ciclos de retroalimentação** (o modelo influencia os dados que o treinam), viés, dependências frágeis, dívida técnica ("Hidden Technical Debt in ML Systems").

## Boas práticas

- Código de pesquisa → pacote testado (testes unitários de transformações, testes de dados/esquema, testes de modelo), CI/CD (GitHub Actions), contêineres (Docker), infraestrutura como código.
- Observabilidade: latência, vazão, erros, uso de recursos, distribuição de entradas/saídas, *logging* de predições para análise posterior.
- Governança: *model cards*, documentação de dados, auditoria, privacidade ([federado](../aprendizado-federado/), DP), explicabilidade ([XAI](../interpretabilidade-xai/)).
- LLMOps: versionamento de *prompts*, avaliação contínua (evals), guardrails, custo/latência por token, observabilidade de cadeias/agentes.

## Implementação

- [`drift.cpp`](drift.cpp) (C++): **PSI, KS e Jensen–Shannon** do zero, aplicados a cenários com e sem deriva e regras de alerta.
- [`mlops_pipeline.py`](mlops_pipeline.py) (Python): mini-pipeline **treino → portão de qualidade → artefato versionado com metadados (hash dos dados, métricas, ambiente) → carga → monitoramento de deriva (PSI + KS)**. Note que a deriva simulada é detectada pelos dados de entrada **sem** alterar visivelmente a proporção prevista — por isso monitorar só as saídas não basta.
- [`serve_fastapi.py`](serve_fastapi.py) (Python): API REST com **FastAPI** + validação de entrada (Pydantic), health check e versão do modelo; inclui teste sem subir servidor.

```bash
g++ -std=c++17 -O2 drift.cpp -o drift && ./drift
pip install scikit-learn scipy joblib pandas fastapi uvicorn httpx
python mlops_pipeline.py        # cria ./artefatos/<versão>/ (ignorado pelo git)
python serve_fastapi.py         # teste rápido;  ou: uvicorn serve_fastapi:app --reload
```

## Referências

- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015).
- Huyen, *Designing Machine Learning Systems* (O'Reilly); Google, *Rules of Machine Learning*.
- Documentação MLflow, DVC, Evidently AI (monitoramento de deriva).
