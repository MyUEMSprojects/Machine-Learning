"""Random Forest com scikit-learn: OOB score, importância de features e ExtraTrees."""
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X, y = data.data, data.target

print("Árvore única :", cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5).mean().round(3))
print("Random Forest:", cross_val_score(RandomForestClassifier(300, random_state=0), X, y, cv=5).mean().round(3))
print("ExtraTrees   :", cross_val_score(ExtraTreesClassifier(300, random_state=0), X, y, cv=5).mean().round(3))

rf = RandomForestClassifier(300, oob_score=True, random_state=0).fit(X, y)
print("OOB score    :", round(rf.oob_score_, 3))
top = sorted(zip(rf.feature_importances_, data.feature_names), reverse=True)[:5]
print("top-5 features:", [(str(n), round(float(v), 3)) for v, n in top])
