// Trade-off viés-variância: ajusta polinômios de grau d a MUITOS conjuntos de treino
// sorteados de y = sin(2 pi x) + ruído e decompõe o erro em viés^2 + variância.
// Build: g++ -std=c++17 -O2 bias_variancia.cpp -o bias_variancia
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

Vec solve(Mat A, Vec b) {  // eliminação de Gauss com pivoteamento
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

// Mínimos quadrados polinomial via equações normais (com ridge minúsculo p/ estabilidade).
Vec polyfit(const Vec& x, const Vec& y, int d) {
    int m = d + 1;
    Mat A(m, Vec(m, 0)); Vec b(m, 0);
    for (size_t i = 0; i < x.size(); ++i)
        for (int p = 0; p < m; ++p) {
            b[p] += pow(x[i], p) * y[i];
            for (int q = 0; q < m; ++q) A[p][q] += pow(x[i], p + q);
        }
    for (int p = 0; p < m; ++p) A[p][p] += 1e-8;
    return solve(A, b);
}

double polyval(const Vec& w, double x) {
    double s = 0;
    for (size_t p = 0; p < w.size(); ++p) s += w[p] * pow(x, p);
    return s;
}

int main() {
    mt19937 rng(0);
    uniform_real_distribution<double> U(0, 1);
    normal_distribution<double> ruido(0, 0.3);
    auto verdade = [](double x) { return sin(2 * M_PI * x); };

    const int N = 25, DATASETS = 300, T = 50;
    Vec xt(T);
    for (int i = 0; i < T; ++i) xt[i] = (i + 0.5) / T;

    cout << "grau | vies^2  | variancia | vies^2+var | (ruido^2 = 0.09)\n";
    for (int d : {0, 1, 3, 5, 9}) {
        Mat preds(DATASETS, Vec(T));
        for (int s = 0; s < DATASETS; ++s) {
            Vec x(N), y(N);
            for (int i = 0; i < N; ++i) { x[i] = U(rng); y[i] = verdade(x[i]) + ruido(rng); }
            Vec w = polyfit(x, y, d);
            for (int i = 0; i < T; ++i) preds[s][i] = polyval(w, xt[i]);
        }
        double vies2 = 0, var = 0;
        for (int i = 0; i < T; ++i) {
            double m = 0;
            for (int s = 0; s < DATASETS; ++s) m += preds[s][i];
            m /= DATASETS;
            vies2 += (m - verdade(xt[i])) * (m - verdade(xt[i]));
            for (int s = 0; s < DATASETS; ++s) var += (preds[s][i] - m) * (preds[s][i] - m) / DATASETS;
        }
        vies2 /= T; var /= T;
        printf("  %d  | %.4f | %.4f    | %.4f\n", d, vies2, var, vies2 + var);
    }
    cout << "Grau baixo => vies alto (underfitting); grau alto => variancia alta (overfitting).\n";
}
