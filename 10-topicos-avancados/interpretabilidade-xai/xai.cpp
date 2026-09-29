// Interpretabilidade de modelos "caixa-preta", do zero: importância por permutação, dependência parcial (PDP),
// valores de Shapley EXATOS (enumeração de coalições) e um LIME simplificado (substituto linear local).
// Caixa-preta: f(x) = 3*x0 + 2*x1*x2 + sin(3*x4)   (x3 é irrelevante; x1 e x2 só importam em conjunto).
// Build: g++ -std=c++17 -O2 xai.cpp -o xai
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;
const int D = 5;

double f(const Vec& x) { return 3 * x[0] + 2 * x[1] * x[2] + sin(3 * x[4]); }

Vec solve(Mat A, Vec b) {
    int n = A.size();
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
        swap(A[p], A[c]); swap(b[p], b[c]);
        for (int r = c + 1; r < n; ++r) { double m = A[r][c] / A[c][c]; for (int k = c; k < n; ++k) A[r][k] -= m * A[c][k]; b[r] -= m * b[c]; }
    }
    Vec x(n);
    for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j]; x[i] = s / A[i][i]; }
    return x;
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    int n = 500;
    Mat X(n, Vec(D)); Vec y(n);
    for (int i = 0; i < n; ++i) { for (double& v : X[i]) v = N(rng); y[i] = f(X[i]) + 0.1 * N(rng); }
    const char* nome[D] = {"x0", "x1", "x2", "x3", "x4"};

    // ---- 1) importância por permutação: quanto o erro AUMENTA quando embaralhamos a feature (quebra sua relação com y)
    auto mse = [&](const Mat& Z) { double s = 0; for (int i = 0; i < n; ++i) s += pow(f(Z[i]) - y[i], 2); return s / n; };
    double base = mse(X);
    printf("1) Importancia por permutacao (MSE base = %.3f):\n", base);
    for (int j = 0; j < D; ++j) {
        double inc = 0; int R = 10;
        for (int r = 0; r < R; ++r) {
            Mat Z = X; vector<int> p(n); iota(p.begin(), p.end(), 0); shuffle(p.begin(), p.end(), rng);
            for (int i = 0; i < n; ++i) Z[i][j] = X[p[i]][j];
            inc += (mse(Z) - base) / R;
        }
        printf("   %s: +%.3f\n", nome[j], inc);
    }

    // ---- 2) dependência parcial de x0: média de f com x0 fixado em cada valor (marginaliza as demais)
    printf("\n2) PDP de x0 (efeito medio; f e linear em x0 com inclinacao 3):\n   ");
    for (double v : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        double s = 0;
        for (int i = 0; i < n; ++i) { Vec z = X[i]; z[0] = v; s += f(z) / n; }
        printf("x0=%.0f -> %.2f   ", v, s);
    }
    printf("\n");

    // ---- 3) Shapley exato para uma instância: v(S) = E_fundo[ f(x_S, X_fundo,~S) ]
    Vec x = {1.0, 1.5, 2.0, 0.7, 0.5};
    auto v = [&](int mask) {
        double s = 0;
        for (int i = 0; i < n; ++i) { Vec z = X[i]; for (int j = 0; j < D; ++j) if (mask >> j & 1) z[j] = x[j]; s += f(z) / n; }
        return s;
    };
    vector<double> V(1 << D);
    for (int m = 0; m < (1 << D); ++m) V[m] = v(m);
    auto fact = [](int k) { double r = 1; for (int i = 2; i <= k; ++i) r *= i; return r; };
    Vec phi(D, 0.0);
    for (int j = 0; j < D; ++j)
        for (int m = 0; m < (1 << D); ++m) {
            if (m >> j & 1) continue;
            int s = __builtin_popcount(m);
            double w = fact(s) * fact(D - s - 1) / fact(D);  // peso de Shapley da coalizão S
            phi[j] += w * (V[m | (1 << j)] - V[m]);          // contribuição marginal de j a S
        }
    printf("\n3) Shapley exato para x = (1.0, 1.5, 2.0, 0.7, 0.5):\n");
    double soma = 0;
    for (int j = 0; j < D; ++j) { printf("   phi(%s) = %+.3f\n", nome[j], phi[j]); soma += phi[j]; }
    printf("   soma dos phi = %.3f | f(x) - E[f(X)] = %.3f  (propriedade de eficiencia: devem coincidir)\n", soma, f(x) - V[0]);
    printf("   note: x1 e x2 dividem o credito da interacao 2*x1*x2; x3 recebe ~0 (irrelevante)\n");

    // ---- 4) LIME: perturba em torno de x, pondera por proximidade, ajusta regressão linear ponderada
    int Ns = 2000; double sigma = 0.5;
    Mat A(D + 1, Vec(D + 1, 0)); Vec b(D + 1, 0);
    for (int i = 0; i < Ns; ++i) {
        Vec z(D);
        for (int j = 0; j < D; ++j) z[j] = x[j] + sigma * N(rng);
        double d2 = 0; for (int j = 0; j < D; ++j) d2 += (z[j] - x[j]) * (z[j] - x[j]);
        double w = exp(-d2 / (2 * D * sigma * sigma));                   // kernel de proximidade
        Vec u = {1.0}; for (double zj : z) u.push_back(zj);
        double fz = f(z);
        for (int p = 0; p < D + 1; ++p) { b[p] += w * u[p] * fz; for (int q = 0; q < D + 1; ++q) A[p][q] += w * u[p] * u[q]; }
    }
    Vec c = solve(A, b);
    printf("\n4) LIME (coeficientes do modelo linear local ~ inclinacoes de f em x):\n   ");
    for (int j = 0; j < D; ++j) printf("%s: %+.2f  ", nome[j], c[j + 1]);
    printf("\n   verdadeiro (gradiente): x0: +3.00  x1: %+.2f  x2: %+.2f  x3: +0.00  x4: %+.2f\n", 2 * x[2], 2 * x[1], 3 * cos(3 * x[4]));
}
