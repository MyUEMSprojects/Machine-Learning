"""Otimização de hiperparâmetros com Optuna (TPE, uma forma de otimização bayesiana) vs. busca aleatória.

Ajusta um Gradient Boosting; a função objetivo é a acurácia de validação cruzada (cara de avaliar).

pip install optuna scikit-learn
"""
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = load_breast_cancer(return_X_y=True)


def objetivo(trial):
    m = GradientBoostingClassifier(
        n_estimators=trial.suggest_int("n_estimators", 20, 300),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),  # escala logarítmica
        max_depth=trial.suggest_int("max_depth", 1, 6),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        random_state=0,
    )
    return cross_val_score(m, X, y, cv=3).mean()


for nome, sampler in [("Aleatória", optuna.samplers.RandomSampler(seed=0)), ("TPE (bayesiana)", optuna.samplers.TPESampler(seed=0))]:
    estudo = optuna.create_study(direction="maximize", sampler=sampler)
    estudo.optimize(objetivo, n_trials=30)
    melhor_ate = [max(t.value for t in estudo.trials[: i + 1]) for i in (4, 9, 19, 29)]
    print(f"{nome:16s} melhor acc após 5/10/20/30 avaliações: {[round(v, 4) for v in melhor_ate]}")
    print(f"{'':16s} melhores parâmetros: { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in estudo.best_params.items()} }")
