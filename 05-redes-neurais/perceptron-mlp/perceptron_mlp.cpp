// Perceptron (regra de Rosenblatt) e MLP de 1 camada oculta treinado com backprop + SGD em mini-lotes.
// O perceptron falha no XOR; o MLP resolve.
// Build: g++ -std=c++17 -O2 perceptron_mlp.cpp -o perceptron_mlp
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

// ---------- Perceptron ----------
struct Perceptron {
    Vec w; double b = 0;
    void fit(const Mat& X, const vector<int>& y, int epochs) {  // y em {0,1}
        w.assign(X[0].size(), 0.0);
        for (int e = 0; e < epochs; ++e)
            for (size_t i = 0; i < X.size(); ++i) {
                int err = y[i] - predict(X[i]);  // -1, 0 ou +1: só atualiza quando erra
                for (size_t j = 0; j < w.size(); ++j) w[j] += err * X[i][j];
                b += err;
            }
    }
    int predict(const Vec& x) const {
        double s = b;
        for (size_t j = 0; j < w.size(); ++j) s += w[j] * x[j];
        return s > 0;
    }
};

// ---------- MLP: entrada -> H tanh -> 1 sigmoide ----------
struct MLP {
    int D, H;
    Mat W1; Vec b1, W2; double b2;
    MLP(int D_, int H_, mt19937& rng) : D(D_), H(H_), W1(H_, Vec(D_)), b1(H_, 0), W2(H_), b2(0) {
        normal_distribution<double> N(0, 1);
        double s1 = sqrt(1.0 / D), s2 = sqrt(1.0 / H);  // inicialização escalada
        for (auto& r : W1) for (double& w : r) w = N(rng) * s1;
        for (double& w : W2) w = N(rng) * s2;
    }
    double forward(const Vec& x, Vec& h) const {
        h.assign(H, 0);
        for (int i = 0; i < H; ++i) {
            double z = b1[i];
            for (int j = 0; j < D; ++j) z += W1[i][j] * x[j];
            h[i] = tanh(z);
        }
        double z = b2;
        for (int i = 0; i < H; ++i) z += W2[i] * h[i];
        return 1.0 / (1.0 + exp(-z));
    }
    double predict_proba(const Vec& x) const { Vec h; return forward(x, h); }

    void fit(const Mat& X, const vector<int>& y, int epochs, double lr, int batch, mt19937& rng) {
        int n = X.size();
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        for (int e = 0; e < epochs; ++e) {
            shuffle(idx.begin(), idx.end(), rng);
            for (int s = 0; s < n; s += batch) {
                int m = min(batch, n - s);
                Mat gW1(H, Vec(D, 0)); Vec gb1(H, 0), gW2(H, 0); double gb2 = 0;
                for (int t = 0; t < m; ++t) {
                    int i = idx[s + t];
                    Vec h; double p = forward(X[i], h);
                    double dz2 = p - y[i];                       // dL/dz2 (sigmoide + entropia cruzada)
                    for (int k = 0; k < H; ++k) {
                        gW2[k] += dz2 * h[k];
                        double dz1 = dz2 * W2[k] * (1 - h[k] * h[k]);  // regra da cadeia através do tanh
                        gb1[k] += dz1;
                        for (int j = 0; j < D; ++j) gW1[k][j] += dz1 * X[i][j];
                    }
                    gb2 += dz2;
                }
                for (int k = 0; k < H; ++k) {
                    W2[k] -= lr * gW2[k] / m; b1[k] -= lr * gb1[k] / m;
                    for (int j = 0; j < D; ++j) W1[k][j] -= lr * gW1[k][j] / m;
                }
                b2 -= lr * gb2 / m;
            }
        }
    }
};

template <class F>
double acc(F f, const Mat& X, const vector<int>& y) {
    int ok = 0;
    for (size_t i = 0; i < X.size(); ++i) ok += f(X[i]) == y[i];
    return (double)ok / X.size();
}

int main() {
    mt19937 rng(0);
    // 1) portas lógicas AND (separável) e XOR (não separável linearmente)
    Mat X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<int> yand = {0, 0, 0, 1}, yxor = {0, 1, 1, 0};
    Perceptron p1, p2; p1.fit(X, yand, 100); p2.fit(X, yxor, 100);
    printf("Perceptron AND: acc=%.2f | Perceptron XOR: acc=%.2f (impossivel: nao e linearmente separavel)\n",
           acc([&](const Vec& x) { return p1.predict(x); }, X, yand), acc([&](const Vec& x) { return p2.predict(x); }, X, yxor));
    MLP m(2, 4, rng);
    m.fit(X, yxor, 3000, 0.5, 4, rng);
    printf("MLP (4 neuronios ocultos) XOR: acc=%.2f  probs=[%.3f %.3f %.3f %.3f]\n",
           acc([&](const Vec& x) { return m.predict_proba(x) > 0.5; }, X, yxor),
           m.predict_proba(X[0]), m.predict_proba(X[1]), m.predict_proba(X[2]), m.predict_proba(X[3]));

    // 2) duas espirais: problema não linear mais difícil
    Mat Xs; vector<int> ys;
    normal_distribution<double> N(0, 0.05);
    for (int i = 0; i < 400; ++i) {
        int c = i % 2; double t = 0.5 + 3.0 * (i / 2) / 200.0 * M_PI, r = t / (1.5 * M_PI);
        double a = t + c * M_PI;
        Xs.push_back({r * cos(a) + N(rng), r * sin(a) + N(rng)}); ys.push_back(c);
    }
    vector<int> perm(Xs.size());
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);  // embaralha antes de separar treino/teste
    Mat Xtr, Xte; vector<int> ytr, yte;
    for (size_t k = 0; k < perm.size(); ++k) { (k < 300 ? Xtr : Xte).push_back(Xs[perm[k]]); (k < 300 ? ytr : yte).push_back(ys[perm[k]]); }
    MLP net(2, 32, rng);
    net.fit(Xtr, ytr, 1500, 0.1, 32, rng);
    auto f = [&](const Vec& x) { return net.predict_proba(x) > 0.5; };
    printf("Duas espirais, MLP (32 ocultos): acc treino=%.3f teste=%.3f\n", acc(f, Xtr, ytr), acc(f, Xte, yte));
}
