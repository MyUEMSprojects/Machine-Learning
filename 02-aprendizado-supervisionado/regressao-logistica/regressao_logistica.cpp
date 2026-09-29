// Regressão logística binária e softmax (multiclasse) por gradiente descendente.
// Build: g++ -std=c++17 -O2 regressao_logistica.cpp -o regressao_logistica
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

// Binária: P(y=1|x) = sigmoid(w.x), perda = entropia cruzada (+ L2 opcional)
struct LogReg {
    Vec w;
    void fit(const Mat& X, const vector<int>& y, double lr = 0.1, int iters = 2000, double l2 = 0.0) {
        int n = X.size(), d = X[0].size() + 1;
        w.assign(d, 0.0);
        for (int it = 0; it < iters; ++it) {
            Vec g(d, 0.0);
            for (int i = 0; i < n; ++i) {
                double p = proba(X[i]);
                double err = p - y[i];  // gradiente da entropia cruzada wrt z
                g[0] += err;
                for (int j = 1; j < d; ++j) g[j] += err * X[i][j - 1];
            }
            for (int j = 0; j < d; ++j) w[j] -= lr * (g[j] / n + (j ? l2 * w[j] : 0));
        }
    }
    double proba(const Vec& x) const {
        double z = w[0];
        for (size_t j = 0; j < x.size(); ++j) z += w[j + 1] * x[j];
        return sigmoid(z);
    }
    int predict(const Vec& x) const { return proba(x) >= 0.5; }
};

// Multiclasse: softmax(W x). Gradiente: (p - onehot(y)) x^T
struct Softmax {
    Mat W;  // K x (d+1)
    int K;
    Vec proba(const Vec& x) const {
        Vec z(K);
        double mx = -1e300;
        for (int k = 0; k < K; ++k) {
            z[k] = W[k][0];
            for (size_t j = 0; j < x.size(); ++j) z[k] += W[k][j + 1] * x[j];
            mx = max(mx, z[k]);
        }
        double s = 0;
        for (double& v : z) { v = exp(v - mx); s += v; }  // estabilidade numérica
        for (double& v : z) v /= s;
        return z;
    }
    void fit(const Mat& X, const vector<int>& y, int K_, double lr = 0.1, int iters = 2000) {
        K = K_;
        int n = X.size(), d = X[0].size() + 1;
        W.assign(K, Vec(d, 0.0));
        for (int it = 0; it < iters; ++it) {
            Mat G(K, Vec(d, 0.0));
            for (int i = 0; i < n; ++i) {
                Vec p = proba(X[i]);
                for (int k = 0; k < K; ++k) {
                    double err = p[k] - (y[i] == k);
                    G[k][0] += err;
                    for (int j = 1; j < d; ++j) G[k][j] += err * X[i][j - 1];
                }
            }
            for (int k = 0; k < K; ++k) for (int j = 0; j < d; ++j) W[k][j] -= lr * G[k][j] / n;
        }
    }
    int predict(const Vec& x) const {
        Vec p = proba(x);
        return max_element(p.begin(), p.end()) - p.begin();
    }
};

void blobs(int n, const vector<Vec>& centros, mt19937& rng, Mat& X, vector<int>& y) {
    normal_distribution<double> N(0, 1);
    for (int i = 0; i < n; ++i) {
        int c = i % centros.size();
        X.push_back({centros[c][0] + N(rng), centros[c][1] + N(rng)});
        y.push_back(c);
    }
}

template <class M>
double acc(const M& m, const Mat& X, const vector<int>& y) {
    int ok = 0;
    for (size_t i = 0; i < X.size(); ++i) ok += m.predict(X[i]) == y[i];
    return (double)ok / X.size();
}

int main() {
    mt19937 rng(0);
    {
        Mat Xtr, Xte; vector<int> ytr, yte;
        blobs(300, {{-1.5, -1.5}, {1.5, 1.5}}, rng, Xtr, ytr);
        blobs(200, {{-1.5, -1.5}, {1.5, 1.5}}, rng, Xte, yte);
        LogReg m; m.fit(Xtr, ytr);
        printf("Binaria: w=[%.2f, %.2f, %.2f]  acc treino=%.3f teste=%.3f\n", m.w[0], m.w[1], m.w[2], acc(m, Xtr, ytr), acc(m, Xte, yte));
        printf("  P(y=1 | x=(0,0)) = %.3f   P(y=1 | x=(3,3)) = %.3f\n", m.proba({0, 0}), m.proba({3, 3}));
    }
    {
        Mat Xtr, Xte; vector<int> ytr, yte;
        vector<Vec> c = {{-3, 0}, {3, 0}, {0, 4}};
        blobs(300, c, rng, Xtr, ytr); blobs(300, c, rng, Xte, yte);
        Softmax m; m.fit(Xtr, ytr, 3);
        printf("Softmax 3 classes: acc treino=%.3f teste=%.3f\n", acc(m, Xtr, ytr), acc(m, Xte, yte));
    }
}
