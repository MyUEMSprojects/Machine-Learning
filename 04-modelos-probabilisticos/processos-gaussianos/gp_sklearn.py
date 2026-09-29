"""Processo Gaussiano com scikit-learn: kernel RBF + ruído, hiperparâmetros por máxima verossimilhança marginal."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

rng = np.random.default_rng(0)
X = rng.uniform(-4, 4, 25)
X = X[(X < 0.5) | (X > 2)][:, None]  # buraco em [0.5, 2]
y = np.sin(X).ravel() + rng.normal(0, 0.15, len(X))

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=5, random_state=0).fit(X, y)
print("kernel otimizado:", gp.kernel_)

for xs in [-3.0, 1.25, 6.0]:
    m, s = gp.predict([[xs]], return_std=True)
    print(f"x={xs:5.2f}  média={m[0]:6.3f}  desvio={s[0]:.3f}  (sin(x)={np.sin(xs):6.3f})")
