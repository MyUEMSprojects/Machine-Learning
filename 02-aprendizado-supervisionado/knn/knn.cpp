// k-Vizinhos mais próximos (classificação e regressão) do zero, força bruta.
// Dataset "duas luas" gerado no código. Mostra o efeito de k.
// Build: g++ -std=c++17 -O2 knn.cpp -o knn
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double dist2(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

struct KNN {
    Mat X; vector<int> y; int k;
    void fit(const Mat& X_, const vector<int>& y_, int k_) { X = X_; y = y_; k = k_; }  // "treinar" = memorizar
    int predict(const Vec& q) const {
        vector<pair<double, int>> d;
        for (size_t i = 0; i < X.size(); ++i) d.push_back({dist2(q, X[i]), y[i]});
        partial_sort(d.begin(), d.begin() + k, d.end());
        map<int, int> votos;
        for (int i = 0; i < k; ++i) ++votos[d[i].second];
        int best = -1, bc = -1;
        for (auto& [c, v] : votos) if (v > bc) { bc = v; best = c; }
        return best;
    }
};

// Regressão: média dos alvos dos k vizinhos
double knn_regress(const Mat& X, const Vec& y, const Vec& q, int k) {
    vector<pair<double, double>> d;
    for (size_t i = 0; i < X.size(); ++i) d.push_back({dist2(q, X[i]), y[i]});
    partial_sort(d.begin(), d.begin() + k, d.end());
    double s = 0;
    for (int i = 0; i < k; ++i) s += d[i].second;
    return s / k;
}

void luas(int n, double ruido, mt19937& rng, Mat& X, vector<int>& y) {
    uniform_real_distribution<double> U(0, M_PI);
    normal_distribution<double> N(0, ruido);
    for (int i = 0; i < n; ++i) {
        double t = U(rng);
        if (i % 2 == 0) { X.push_back({cos(t) + N(rng), sin(t) + N(rng)}); y.push_back(0); }
        else { X.push_back({1 - cos(t) + N(rng), 0.5 - sin(t) + N(rng)}); y.push_back(1); }
    }
}

int main() {
    mt19937 rng(1);
    Mat Xtr, Xte; vector<int> ytr, yte;
    luas(300, 0.25, rng, Xtr, ytr);
    luas(200, 0.25, rng, Xte, yte);

    cout << "  k | acc treino | acc teste\n";
    for (int k : {1, 3, 5, 15, 51, 151}) {
        KNN m; m.fit(Xtr, ytr, k);
        int a = 0, b = 0;
        for (size_t i = 0; i < Xtr.size(); ++i) a += m.predict(Xtr[i]) == ytr[i];
        for (size_t i = 0; i < Xte.size(); ++i) b += m.predict(Xte[i]) == yte[i];
        printf("%3d |   %.3f    |  %.3f\n", k, (double)a / Xtr.size(), (double)b / Xte.size());
    }
    cout << "k=1 memoriza o treino (variancia alta); k enorme vira \"classe majoritaria\" (vies alto).\n";

    // regressão 1D: y = sin(x) + ruido
    Mat Xr; Vec yr;
    uniform_real_distribution<double> U(0, 2 * M_PI);
    normal_distribution<double> N(0, 0.2);
    for (int i = 0; i < 200; ++i) { double x = U(rng); Xr.push_back({x}); yr.push_back(sin(x) + N(rng)); }
    printf("\nkNN regressao (k=10): f(pi/2)=%.3f (esperado 1), f(3pi/2)=%.3f (esperado -1)\n",
           knn_regress(Xr, yr, {M_PI / 2}, 10), knn_regress(Xr, yr, {3 * M_PI / 2}, 10));
}
