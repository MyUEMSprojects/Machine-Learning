"""Séries temporais com bibliotecas: statsmodels (Holt-Winters, SARIMA) e ML com atributos de defasagem (TimeSeriesSplit).

pip install numpy pandas scikit-learn statsmodels
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n, H, M = 240, 24, 12
t = np.arange(n)
e = np.zeros(n)
for i in range(1, n):
    e[i] = 0.5 * e[i - 1] + 0.8 * rng.normal()
y = pd.Series(10 + 0.05 * t + 5 * np.sin(2 * np.pi * t / M) + e, index=pd.period_range("2005-01", periods=n, freq="M").to_timestamp())
treino, teste = y[:-H], y[-H:]
mae = lambda p: float(np.mean(np.abs(np.asarray(p) - teste.values)))

# estacionariedade: teste ADF (H0: raiz unitária = não estacionária)
print(f"ADF p-valor: série original={adfuller(y)[1]:.3f} | diferenciada={adfuller(y.diff().dropna())[1]:.4f}")

hw = ExponentialSmoothing(treino, trend="add", seasonal="add", seasonal_periods=M).fit()
print(f"Holt-Winters (statsmodels)   MAE = {mae(hw.forecast(H)):.3f}")

sarima = SARIMAX(treino, order=(1, 1, 1), seasonal_order=(1, 1, 1, M)).fit(disp=False)
print(f"SARIMA(1,1,1)(1,1,1)[12]     MAE = {mae(sarima.forecast(H)):.3f}   AIC = {sarima.aic:.1f}")
ic = sarima.get_forecast(H).conf_int(alpha=0.05)
print(f"  intervalo 95% cobre {np.mean((teste.values >= ic.iloc[:, 0]) & (teste.values <= ic.iloc[:, 1])):.0%} dos pontos de teste")

# ML "tabular" com atributos de defasagem + sazonais; recursivo multi-passo
def atributos(serie, i):
    return [serie[i - k] for k in (1, 2, 3, 12, 13)] + [np.sin(2 * np.pi * i / M), np.cos(2 * np.pi * i / M), i]


vals = list(treino.values)
X = np.array([atributos(vals, i) for i in range(13, len(vals))])
alvo = np.array(vals[13:])
gb = GradientBoostingRegressor(random_state=0).fit(X, alvo)
prev = list(vals)
for _ in range(H):
    prev.append(gb.predict([atributos(prev, len(prev))])[0])
print(f"GradientBoosting c/ lags     MAE = {mae(prev[-H:]):.3f}  (árvores não extrapolam tendência — tende a ficar pior)")

# validação temporal: treino sempre ANTES do teste
erros = []
for tr_i, te_i in TimeSeriesSplit(5).split(X):
    m = GradientBoostingRegressor(random_state=0).fit(X[tr_i], alvo[tr_i])
    erros.append(np.mean(np.abs(m.predict(X[te_i]) - alvo[te_i])))
print("TimeSeriesSplit — MAE 1-passo por dobra:", np.round(erros, 2))
