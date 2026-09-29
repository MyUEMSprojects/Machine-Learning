// Feature engineering: padronização (z-score), min-max, one-hot, features polinomiais,
// transformação log e codificação de dados faltantes (imputação pela média).
// Build: g++ -std=c++17 -O2 feature_engineering.cpp -o feature_engineering
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

// Escaladores: ajustar (fit) no TREINO, aplicar (transform) em treino e teste.
struct StandardScaler {
    Vec mu, sd;
    void fit(const Mat& X) {
        size_t d = X[0].size();
        mu.assign(d, 0); sd.assign(d, 0);
        for (auto& r : X) for (size_t j = 0; j < d; ++j) mu[j] += r[j] / X.size();
        for (auto& r : X) for (size_t j = 0; j < d; ++j) sd[j] += pow(r[j] - mu[j], 2) / X.size();
        for (auto& s : sd) s = sqrt(s) > 1e-12 ? sqrt(s) : 1.0;
    }
    Mat transform(Mat X) const {
        for (auto& r : X) for (size_t j = 0; j < r.size(); ++j) r[j] = (r[j] - mu[j]) / sd[j];
        return X;
    }
};

struct MinMaxScaler {
    Vec lo, hi;
    void fit(const Mat& X) {
        lo = hi = X[0];
        for (auto& r : X) for (size_t j = 0; j < r.size(); ++j) { lo[j] = min(lo[j], r[j]); hi[j] = max(hi[j], r[j]); }
    }
    Mat transform(Mat X) const {
        for (auto& r : X) for (size_t j = 0; j < r.size(); ++j) r[j] = (r[j] - lo[j]) / (hi[j] - lo[j]);
        return X;
    }
};

// One-hot para variável categórica
Mat one_hot(const vector<string>& col, vector<string>& categorias) {
    map<string, int> id;
    for (auto& c : col) if (!id.count(c)) { id[c] = id.size(); categorias.push_back(c); }
    Mat M(col.size(), Vec(id.size(), 0.0));
    for (size_t i = 0; i < col.size(); ++i) M[i][id[col[i]]] = 1;
    return M;
}

// Features polinomiais de grau 2 para 2 variáveis: [x1, x2, x1^2, x1*x2, x2^2]
Mat poly2(const Mat& X) {
    Mat P;
    for (auto& r : X) P.push_back({r[0], r[1], r[0] * r[0], r[0] * r[1], r[1] * r[1]});
    return P;
}

void print(const string& nome, const Mat& M) {
    cout << nome << ":\n";
    for (auto& r : M) { cout << "  "; for (double v : r) cout << v << "\t"; cout << "\n"; }
}

int main() {
    Mat treino = {{1.0, 1000}, {2.0, 2000}, {3.0, 3000}, {4.0, 10000}};
    Mat teste = {{2.5, 4000}};

    StandardScaler ss; ss.fit(treino);
    print("z-score (treino)", ss.transform(treino));
    print("z-score (teste, com estatisticas do treino)", ss.transform(teste));

    MinMaxScaler mm; mm.fit(treino);
    print("min-max (treino)", mm.transform(treino));

    vector<string> cats;
    print("one-hot de {gato,cao,gato,passaro}", one_hot({"gato", "cao", "gato", "passaro"}, cats));

    print("polinomial grau 2 de (2,3)", poly2({{2, 3}}));

    // cauda longa: log1p aproxima a distribuição de uma normal
    cout << "log1p de {1000,2000,3000,10000}: ";
    for (double v : {1000.0, 2000.0, 3000.0, 10000.0}) cout << log1p(v) << " ";
    cout << "\n";

    // imputação de faltantes (NaN) pela média da coluna
    Vec col = {1.0, NAN, 3.0, NAN, 5.0};
    double s = 0; int n = 0;
    for (double v : col) if (!isnan(v)) { s += v; ++n; }
    for (double& v : col) if (isnan(v)) v = s / n;
    cout << "imputacao pela media: ";
    for (double v : col) cout << v << " ";
    cout << "\n";
}
