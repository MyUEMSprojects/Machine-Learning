# Séries Temporais

Dados ordenados no tempo, onde **a ordem importa** e as observações são **autocorrelacionadas** (violando a suposição i.i.d. de quase todo o resto do curso).

## Componentes

$$y_t=\underbrace{T_t}_{\text{tendência}}+\underbrace{S_t}_{\text{sazonalidade}}+\underbrace{R_t}_{\text{resíduo}}$$

(ou multiplicativo). **Estacionariedade**: média/variância/autocovariância constantes no tempo — exigida por ARIMA; obtida por **diferenciação** ($y_t-y_{t-1}$), diferenciação sazonal ($y_t-y_{t-m}$) ou transformações (log). Teste ADF.

## Ferramentas de diagnóstico

- **ACF**: correlação da série com ela mesma defasada de $k$ passos (picos em múltiplos do período indicam sazonalidade).
- **PACF**: correlação parcial (ajuda a escolher a ordem $p$ do AR).

## Modelos de previsão

| Modelo | Ideia |
|---|---|
| **Ingênuo / sazonal ingênuo** | $\hat y_{t+1}=y_t$ / $\hat y_{t+1}=y_{t+1-m}$ — *baselines* obrigatórios |
| **Suavização exponencial** (SES, Holt, **Holt–Winters**) | nível/tendência/sazonalidade como médias móveis ponderadas exponencialmente ($\alpha,\beta,\gamma$) |
| **AR($p$)** | $y_t=c+\sum_{k=1}^p\phi_ky_{t-k}+\varepsilon_t$ (regressão nas próprias defasagens) |
| **ARIMA($p,d,q$) / SARIMA** | AR + diferenciação ($d$) + média móvel de erros ($q$), com parte sazonal |
| **ETS / state space / Prophet** | decomposições probabilísticas com incerteza |
| **ML com atributos de defasagem** | *lags*, médias móveis, calendário → GBM/Random Forest (não extrapolam tendências: detrend antes) |
| **Deep learning** | RNN/LSTM, TCN, Transformers (Informer, PatchTST), modelos de fundação (TimesFM, Chronos) |

**Previsão multi-passo**: recursiva (usa a própria previsão como entrada; acumula erro), direta (um modelo por horizonte) ou multi-saída.

## Avaliação — regras de ouro

- **Nunca embaralhe**: divida cronologicamente; use **validação walk-forward / origem rolante** (`TimeSeriesSplit`).
- Cuidado com **vazamento do futuro** (normalizar com estatísticas do conjunto completo; features que usam informação posterior).
- Métricas: MAE, RMSE, MAPE/sMAPE, MASE (escala relativa ao *baseline* ingênuo); avalie **intervalos de previsão** (cobertura).
- Sempre compare com o sazonal ingênuo — modelos complexos frequentemente não o superam.

## Implementação

- [`series_temporais.cpp`](series_temporais.cpp) (C++): ACF, ingênuo, sazonal ingênuo, **Holt–Winters** (busca em grade), **AR($p$) por mínimos quadrados com previsão recursiva** e validação walk-forward.
- [`series_statsmodels.py`](series_statsmodels.py) (Python): teste ADF, `ExponentialSmoothing`, **SARIMA** com intervalo de previsão, e GBM com atributos de defasagem + `TimeSeriesSplit`.

```bash
g++ -std=c++17 -O2 series_temporais.cpp -o series_temporais && ./series_temporais
pip install pandas scikit-learn statsmodels && python series_statsmodels.py
```

## Referências

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (otexts.com/fpp3, gratuito).
- Box, Jenkins et al., *Time Series Analysis: Forecasting and Control*.
- Ansari et al., *Chronos: Learning the Language of Time Series* (2024).
