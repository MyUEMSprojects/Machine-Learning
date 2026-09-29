// SVM com Pegasos (SGD na perda hinge): versão linear e versão com kernel RBF.
// Build: g++ -std=c++17 -O2 svm.cpp -o svm
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

// ---------- SVM linear: min lambda/2 ||w||^2 + (1/n) sum max(0, 1 - y w.x) ----------
struct LinearSVM {
    Vec w;  // w[0] é o bias
    void fit(const Mat& X, const vector<int>& y01, double lambda, int T, mt19937& rng) {
        int n = X.size(), d = X[0].size();
        w.assign(d + 1, 0.0);
        uniform_int_distribution<int> pick(0, n - 1);
        for (int t = 1; t <= T; ++t) {
            int i = pick(rng);
            double yi = y01[i] ? 1 : -1, eta = 1.0 / (lambda * t);
            double m = w[0];
            for (int j = 0; j < d; ++j) m += w[j + 1] * X[i][j];
            for (int j = 1; j <= d; ++j) w[j] *= (1 - eta * lambda);  // regularização (encolhe w)
            if (yi * m < 1) {  // viola a margem: passo na direção de y x
                w[0] += eta * yi;
                for (int j = 0; j < d; ++j) w[j + 1] += eta * yi * X[i][j];
            }
        }
    }
    int predict(const Vec& x) const {
        double s = w[0];
        for (size_t j = 0; j < x.size(); ++j) s += w[j + 1] * x[j];
        return s > 0;
    }
};

// ---------- SVM com kernel RBF (Pegasos kernelizado) ----------
struct KernelSVM {
    Mat X; vector<int> y; vector<double> alpha;
    double gamma, lambda; int T;
    double K(const Vec& a, const Vec& b) const {
        double s = 0;
        for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
        return exp(-gamma * s);
    }
    void fit(const Mat& X_, const vector<int>& y01, double gamma_, double lambda_, int T_, mt19937& rng) {
        X = X_; gamma = gamma_; lambda = lambda_; T = T_;
        int n = X.size();
        y.resize(n); alpha.assign(n, 0.0);
        for (int i = 0; i < n; ++i) y[i] = y01[i] ? 1 : -1;
        uniform_int_distribution<int> pick(0, n - 1);
        for (int t = 1; t <= T; ++t) {
            int i = pick(rng);
            double s = 0;
            for (int j = 0; j < n; ++j) if (alpha[j] > 0) s += alpha[j] * y[j] * K(X[j], X[i]);
            if (y[i] * s / (lambda * t) < 1) alpha[i] += 1;
        }
    }
    int predict(const Vec& x) const {
        double s = 0;
        for (size_t j = 0; j < X.size(); ++j) if (alpha[j] > 0) s += alpha[j] * y[j] * K(X[j], x);
        return s > 0;
    }
    int n_support() const { int c = 0; for (double a : alpha) c += a > 0; return c; }
};

template <class M>
double acc(const M& m, const Mat& X, const vector<int>& y) {
    int ok = 0;
    for (size_t i = 0; i < X.size(); ++i) ok += m.predict(X[i]) == y[i];
    return (double)ok / X.size();
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    uniform_real_distribution<double> U(0, 2 * M_PI);

    // 1) linearmente separável
    Mat Xtr, Xte; vector<int> ytr, yte;
    auto blobs = [&](int n, Mat& X, vector<int>& y) {
        for (int i = 0; i < n; ++i) { int c = i % 2; X.push_back({(c ? 2 : -2) + N(rng), (c ? 2 : -2) + N(rng)}); y.push_back(c); }
    };
    blobs(300, Xtr, ytr); blobs(300, Xte, yte);
    LinearSVM lin; lin.fit(Xtr, ytr, 0.01, 20000, rng);
    printf("Blobs   | SVM linear: acc teste=%.3f\n", acc(lin, Xte, yte));

    // 2) círculos concêntricos: não linear
    Mat Ctr, Cte; vector<int> cytr, cyte;
    auto circulos = [&](int n, Mat& X, vector<int>& y) {
        for (int i = 0; i < n; ++i) {
            int c = i % 2; double r = (c ? 2.5 : 1.0) + 0.2 * N(rng), t = U(rng);
            X.push_back({r * cos(t), r * sin(t)}); y.push_back(c);
        }
    };
    circulos(300, Ctr, cytr); circulos(300, Cte, cyte);
    LinearSVM lin2; lin2.fit(Ctr, cytr, 0.01, 20000, rng);
    KernelSVM rbf; rbf.fit(Ctr, cytr, 0.5, 0.01, 5000, rng);
    printf("Circulos| SVM linear: acc teste=%.3f   (nao consegue separar)\n", acc(lin2, Cte, cyte));
    printf("Circulos| SVM RBF   : acc teste=%.3f   (vetores de suporte: %d/%zu)\n", acc(rbf, Cte, cyte), rbf.n_support(), Ctr.size());
}
