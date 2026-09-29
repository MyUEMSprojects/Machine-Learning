// Regularização: Ridge (L2, forma fechada) e Lasso (L1, descida por coordenadas).
// Dados: 100 amostras, 20 features, só 3 relevantes. O Lasso zera as irrelevantes.
// Build: g++ -std=c++17 -O2 regularizacao.cpp -o regularizacao
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

// Ridge: w = (X^T X + lambda I)^{-1} X^T y
Vec ridge(const Mat& X, const Vec& y, double lambda) {
    int n = X.size(), d = X[0].size();
    Mat A(d, Vec(d, 0)); Vec b(d, 0);
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < d; ++p) {
            b[p] += X[i][p] * y[i];
            for (int q = 0; q < d; ++q) A[p][q] += X[i][p] * X[i][q];
        }
    for (int p = 0; p < d; ++p) A[p][p] += lambda;
    return solve(A, b);
}

double soft(double z, double g) { return z > g ? z - g : (z < -g ? z + g : 0.0); }

// Lasso: min (1/2n)||y - Xw||^2 + lambda ||w||_1, por descida por coordenadas
Vec lasso(const Mat& X, const Vec& y, double lambda, int iters = 500) {
    int n = X.size(), d = X[0].size();
    Vec w(d, 0.0), r = y;  // r = resíduo y - Xw
    Vec norm2(d, 0);
    for (int j = 0; j < d; ++j) for (int i = 0; i < n; ++i) norm2[j] += X[i][j] * X[i][j] / n;
    for (int it = 0; it < iters; ++it)
        for (int j = 0; j < d; ++j) {
            double rho = 0;
            for (int i = 0; i < n; ++i) rho += X[i][j] * (r[i] + X[i][j] * w[j]) / n;
            double wn = soft(rho, lambda) / norm2[j];
            for (int i = 0; i < n; ++i) r[i] -= X[i][j] * (wn - w[j]);
            w[j] = wn;
        }
    return w;
}

void mostra(const char* nome, const Vec& w) {
    int zeros = 0;
    cout << nome << ": ";
    for (int j = 0; j < (int)w.size(); ++j) {
        if (fabs(w[j]) < 1e-6) ++zeros;
        if (j < 5) printf("w%d=%.2f ", j, w[j]);
    }
    printf("...  (coeficientes zerados: %d/%zu)\n", zeros, w.size());
}

int main() {
    mt19937 rng(3);
    normal_distribution<double> N(0, 1);
    int n = 100, d = 20;
    Vec w_true(d, 0.0); w_true[0] = 3; w_true[1] = -2; w_true[2] = 1.5;
    Mat X(n, Vec(d)); Vec y(n);
    for (int i = 0; i < n; ++i) {
        y[i] = N(rng) * 0.5;
        for (int j = 0; j < d; ++j) { X[i][j] = N(rng); y[i] += w_true[j] * X[i][j]; }
    }
    cout << "verdadeiro: w0=3 w1=-2 w2=1.5, demais = 0\n";
    mostra("OLS   (lambda=0)  ", ridge(X, y, 1e-9));
    mostra("Ridge (lambda=50) ", ridge(X, y, 50));
    mostra("Lasso (lambda=0.2)", lasso(X, y, 0.2));
}
