// Regressão linear múltipla do zero: equações normais (forma fechada) e gradiente descendente.
// Dados sintéticos: y = 2 + 3*x1 - 1.5*x2 + ruído.
// Build: g++ -std=c++17 -O2 regressao_linear.cpp -o regressao_linear
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

Vec solve(Mat A, Vec b) {
    int n = A.size();
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
        swap(A[p], A[c]); swap(b[p], b[c]);
        for (int r = c + 1; r < n; ++r) {
            double f = A[r][c] / A[c][c];
            for (int k = c; k < n; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    Vec x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

// X já deve conter a coluna de 1s (intercepto) na posição 0.
Vec fit_normal_equations(const Mat& X, const Vec& y) {  // w = (X^T X)^-1 X^T y
    int n = X.size(), d = X[0].size();
    Mat A(d, Vec(d, 0)); Vec b(d, 0);
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < d; ++p) {
            b[p] += X[i][p] * y[i];
            for (int q = 0; q < d; ++q) A[p][q] += X[i][p] * X[i][q];
        }
    return solve(A, b);
}

Vec fit_gradient_descent(const Mat& X, const Vec& y, double lr, int iters) {
    int n = X.size(), d = X[0].size();
    Vec w(d, 0.0);
    for (int it = 0; it < iters; ++it) {
        Vec g(d, 0.0);
        for (int i = 0; i < n; ++i) {
            double err = -y[i];
            for (int j = 0; j < d; ++j) err += w[j] * X[i][j];
            for (int j = 0; j < d; ++j) g[j] += 2.0 * err * X[i][j] / n;  // d(MSE)/dw
        }
        for (int j = 0; j < d; ++j) w[j] -= lr * g[j];
    }
    return w;
}

double predict(const Vec& w, const Vec& x) {
    double s = 0;
    for (size_t j = 0; j < w.size(); ++j) s += w[j] * x[j];
    return s;
}

double r2(const Vec& w, const Mat& X, const Vec& y) {
    double m = 0, res = 0, tot = 0;
    for (double v : y) m += v / y.size();
    for (size_t i = 0; i < y.size(); ++i) {
        res += pow(y[i] - predict(w, X[i]), 2);
        tot += pow(y[i] - m, 2);
    }
    return 1 - res / tot;
}

int main() {
    mt19937 rng(0);
    uniform_real_distribution<double> U(-1, 1);
    normal_distribution<double> ruido(0, 0.3);
    auto gera = [&](int n, Mat& X, Vec& y) {
        for (int i = 0; i < n; ++i) {
            double x1 = U(rng), x2 = U(rng);
            X.push_back({1.0, x1, x2});
            y.push_back(2 + 3 * x1 - 1.5 * x2 + ruido(rng));
        }
    };
    Mat Xtr, Xte; Vec ytr, yte;
    gera(200, Xtr, ytr); gera(100, Xte, yte);

    Vec w1 = fit_normal_equations(Xtr, ytr);
    Vec w2 = fit_gradient_descent(Xtr, ytr, 0.1, 2000);
    printf("verdadeiro         : b=2.000 w1=3.000 w2=-1.500\n");
    printf("equacoes normais   : b=%.3f w1=%.3f w2=%.3f  R2 teste=%.4f\n", w1[0], w1[1], w1[2], r2(w1, Xte, yte));
    printf("gradiente descend. : b=%.3f w1=%.3f w2=%.3f  R2 teste=%.4f\n", w2[0], w2[1], w2[2], r2(w2, Xte, yte));
}
