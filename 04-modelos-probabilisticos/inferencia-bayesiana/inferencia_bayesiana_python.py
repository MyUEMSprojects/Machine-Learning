"""Inferência bayesiana com PyMC (MCMC/NUTS): regressão linear bayesiana com incerteza nos parâmetros.

    pip install pymc arviz
"""
import numpy as np

try:
    import arviz as az
    import pymc as pm
except ImportError:
    raise SystemExit("Instale as dependências: pip install pymc arviz")

rng = np.random.default_rng(0)
x = rng.uniform(-2, 2, 40)
y = 1.0 + 2.0 * x + rng.normal(0, 0.7, 40)  # verdadeiros: a=1, b=2, sigma=0.7

with pm.Model():
    a = pm.Normal("a", 0, 5)
    b = pm.Normal("b", 0, 5)
    sigma = pm.HalfNormal("sigma", 2)
    pm.Normal("y", mu=a + b * x, sigma=sigma, observed=y)
    trace = pm.sample(1000, tune=1000, chains=2, random_seed=0, progressbar=False)

print(az.summary(trace, var_names=["a", "b", "sigma"]))
