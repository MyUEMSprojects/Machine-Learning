// AdaBoost (discreto) com stumps (árvores de profundidade 1) do zero.
// Build: g++ -std=c++17 -O2 adaboost.cpp -o adaboost
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

struct Stump {
    int feat = 0; double thr = 0; int sign = 1;  // predição: sign se x[feat] > thr, senão -sign
    int predict(const Vec& x) const { return x[feat] > thr ? sign : -sign; }
};

// Escolhe o stump com menor erro PONDERADO pelos pesos w. y em {-1,+1}.
Stump best_stump(const Mat& X, const vector<int>& y, const Vec& w, double& best_err) {
    int n = X.size(), d = X[0].size();
    Stump best; best_err = 1e18;
    for (int f = 0; f < d; ++f)
        for (int i = 0; i < n; ++i) {  // limiares candidatos = valores dos dados
            double thr = X[i][f];
            for (int sg : {1, -1}) {
                double err = 0;
                for (int j = 0; j < n; ++j) if ((X[j][f] > thr ? sg : -sg) != y[j]) err += w[j];
                if (err < best_err) { best_err = err; best = {f, thr, sg}; }
            }
        }
    return best;
}

struct AdaBoost {
    vector<Stump> stumps; Vec alphas;
    void fit(const Mat& X, const vector<int>& y, int T) {
        int n = X.size();
        Vec w(n, 1.0 / n);
        for (int t = 0; t < T; ++t) {
            double err;
            Stump s = best_stump(X, y, w, err);
            err = max(err, 1e-10);
            double alpha = 0.5 * log((1 - err) / err);  // stump melhor => voto maior
            double Z = 0;
            for (int i = 0; i < n; ++i) {
                w[i] *= exp(-alpha * y[i] * s.predict(X[i]));  // sobe peso dos erros, desce dos acertos
                Z += w[i];
            }
            for (double& v : w) v /= Z;
            stumps.push_back(s); alphas.push_back(alpha);
        }
    }
    double score(const Vec& x, int T = -1) const {
        double s = 0;
        int m = T < 0 ? stumps.size() : T;
        for (int t = 0; t < m; ++t) s += alphas[t] * stumps[t].predict(x);
        return s;
    }
    int predict(const Vec& x, int T = -1) const { return score(x, T) > 0 ? 1 : -1; }
};

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    uniform_real_distribution<double> U(0, 2 * M_PI);
    // círculos: nenhum stump sozinho separa, mas a combinação sim
    auto gera = [&](int n, Mat& X, vector<int>& y) {
        for (int i = 0; i < n; ++i) {
            int c = i % 2; double r = (c ? 2.0 : 0.8) + 0.25 * N(rng), t = U(rng);
            X.push_back({r * cos(t), r * sin(t)}); y.push_back(c ? 1 : -1);
        }
    };
    Mat Xtr, Xte; vector<int> ytr, yte;
    gera(300, Xtr, ytr); gera(300, Xte, yte);
    AdaBoost ab; ab.fit(Xtr, ytr, 100);
    cout << "  rodadas | acc treino | acc teste\n";
    for (int T : {1, 5, 10, 25, 50, 100}) {
        int a = 0, b = 0;
        for (size_t i = 0; i < Xtr.size(); ++i) a += ab.predict(Xtr[i], T) == ytr[i];
        for (size_t i = 0; i < Xte.size(); ++i) b += ab.predict(Xte[i], T) == yte[i];
        printf("   %4d   |   %.3f    |  %.3f\n", T, (double)a / Xtr.size(), (double)b / Xte.size());
    }
}
