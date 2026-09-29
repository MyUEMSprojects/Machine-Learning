"""HMM em Python: forward e Viterbi em NumPy (curto) — para uma biblioteca pronta use `hmmlearn`.

    pip install hmmlearn   # opcional: aprendizado (Baum-Welch) e HMM gaussiano
"""
import numpy as np

pi = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])  # estados: ensolarado/chuvoso; obs: caminhar/comprar/limpar
obs = [0, 1, 2, 2, 1, 0, 0]


def viterbi(obs):
    T, N = len(obs), len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = np.log(pi) + np.log(B[:, obs[0]])
    for t in range(1, T):
        cand = delta[t - 1][:, None] + np.log(A)  # cand[i, j]: vir de i para j
        psi[t] = cand.argmax(0)
        delta[t] = cand.max(0) + np.log(B[:, obs[t]])
    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t][path[-1]]))
    return path[::-1]


print("estados mais prováveis:", viterbi(obs), "(0 = ensolarado, 1 = chuvoso)")

try:
    from hmmlearn.hmm import CategoricalHMM

    rng = np.random.default_rng(0)
    seq = rng.integers(0, 3, size=(500, 1))
    m = CategoricalHMM(n_components=2, n_iter=100, random_state=0).fit(seq)
    print("hmmlearn — log-verossimilhança:", round(m.score(seq), 1))
except ImportError:
    print("(hmmlearn não instalado — pulando exemplo de aprendizado)")
