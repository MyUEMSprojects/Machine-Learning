// Séries temporais do zero: ACF, previsão ingênua/sazonal, suavização exponencial (SES, Holt, Holt-Winters aditivo)
// e autorregressão AR(p) por mínimos quadrados, comparadas por MAE numa previsão de 24 passos.
// Build: g++ -std=c++17 -O2 series_temporais.cpp -o series_temporais
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
const int M = 12;  // período sazonal (mensal)

double mae(const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) s += fabs(a[i] - b[i]); return s / a.size(); }

Vec acf(const Vec& y, int maxlag) {
    double m = accumulate(y.begin(), y.end(), 0.0) / y.size(), den = 0;
    for (double v : y) den += (v - m) * (v - m);
    Vec r;
    for (int k = 1; k <= maxlag; ++k) { double s = 0; for (size_t t = k; t < y.size(); ++t) s += (y[t] - m) * (y[t - k] - m); r.push_back(s / den); }
    return r;
}

Vec forecast_naive(const Vec& y, int h) { return Vec(h, y.back()); }
Vec forecast_snaive(const Vec& y, int h) { Vec f; for (int i = 0; i < h; ++i) f.push_back(y[y.size() - M + (i % M)]); return f; }

// Holt-Winters aditivo: nível l, tendência b, sazonalidade s[]. Parâmetros alpha, beta, gamma em [0,1].
struct HW {
    double a, b, g; double l, tr; Vec s; double sse = 0;
    void fit(const Vec& y) {
        l = accumulate(y.begin(), y.begin() + M, 0.0) / M;
        tr = (accumulate(y.begin() + M, y.begin() + 2 * M, 0.0) - accumulate(y.begin(), y.begin() + M, 0.0)) / (M * M);
        s.assign(M, 0);
        for (int i = 0; i < M; ++i) s[i] = y[i] - l;
        sse = 0;
        for (size_t t = M; t < y.size(); ++t) {
            double pred = l + tr + s[t % M], err = y[t] - pred;
            sse += err * err;
            double ln = a * (y[t] - s[t % M]) + (1 - a) * (l + tr);
            tr = b * (ln - l) + (1 - b) * tr;
            s[t % M] = g * (y[t] - ln) + (1 - g) * s[t % M];
            l = ln;
        }
    }
    Vec forecast(size_t n, int h) const { Vec f; for (int i = 1; i <= h; ++i) f.push_back(l + i * tr + s[(n + i - 1) % M]); return f; }
};

Vec solve(Mat A, Vec b) {
    int n = A.size();
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r) if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
        swap(A[p], A[c]); swap(b[p], b[c]);
        for (int r = c + 1; r < n; ++r) { double f = A[r][c] / A[c][c]; for (int k = c; k < n; ++k) A[r][k] -= f * A[c][k]; b[r] -= f * b[c]; }
    }
    Vec x(n);
    for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j]; x[i] = s / A[i][i]; }
    return x;
}

// AR(p) com tendência linear: y_t = c + d*t + sum phi_k y_{t-k}; previsão recursiva (cada previsão vira entrada da próxima)
Vec forecast_ar(const Vec& y, int p, int h) {
    int n = y.size(), d = p + 2;
    Mat A(d, Vec(d, 0)); Vec b(d, 0);
    for (int t = p; t < n; ++t) {
        Vec x = {1.0, (double)t};
        for (int k = 1; k <= p; ++k) x.push_back(y[t - k]);
        for (int i = 0; i < d; ++i) { b[i] += x[i] * y[t]; for (int j = 0; j < d; ++j) A[i][j] += x[i] * x[j]; }
    }
    for (int i = 0; i < d; ++i) A[i][i] += 1e-6;
    Vec w = solve(A, b), ext = y, f;
    for (int i = 0; i < h; ++i) {
        int t = ext.size();
        double v = w[0] + w[1] * t;
        for (int k = 1; k <= p; ++k) v += w[1 + k] * ext[t - k];
        ext.push_back(v); f.push_back(v);
    }
    return f;
}

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    int n = 240, H = 24;
    Vec y(n); double e = 0;
    for (int t = 0; t < n; ++t) { e = 0.5 * e + 0.8 * N(rng); y[t] = 10 + 0.05 * t + 5 * sin(2 * M_PI * t / M) + e; }  // tendência + sazonalidade + ruído AR(1)
    Vec treino(y.begin(), y.end() - H), teste(y.end() - H, y.end());

    // ACF de série sem tendência: pico em lag 12 revela sazonalidade
    Vec dt(y.size() - 1);
    for (size_t t = 1; t < y.size(); ++t) dt[t - 1] = y[t] - y[t - 1];
    Vec r = acf(dt, 14);
    printf("ACF da serie diferenciada (lags 1..14):");
    for (double v : r) printf(" %.2f", v);
    printf("\n  -> pico positivo perto do lag 12 = sazonalidade anual\n\n");

    // Holt-Winters: escolhe (alpha,beta,gamma) minimizando o erro de previsão 1-passo no treino (busca em grade)
    HW best; best.sse = 1e18;
    for (double a : {0.1, 0.3, 0.5}) for (double b : {0.01, 0.05, 0.1}) for (double g : {0.1, 0.3, 0.5}) {
        HW m; m.a = a; m.b = b; m.g = g; m.fit(treino);
        if (m.sse < best.sse) { HW cp = m; best = cp; }
    }
    printf("Holt-Winters escolhido: alpha=%.2f beta=%.2f gamma=%.2f\n\n", best.a, best.b, best.g);

    printf("MAE na previsao dos proximos %d meses (treino = %zu pontos):\n", H, treino.size());
    printf("  ingenuo (ultimo valor)      : %.3f\n", mae(teste, forecast_naive(treino, H)));
    printf("  sazonal ingenuo (y[t-12])   : %.3f\n", mae(teste, forecast_snaive(treino, H)));
    printf("  Holt-Winters aditivo        : %.3f\n", mae(teste, best.forecast(treino.size(), H)));
    for (int p : {2, 12, 14}) printf("  AR(%2d) + tendencia          : %.3f\n", p, mae(teste, forecast_ar(treino, p, H)));

    // validação walk-forward (origem rolante): nunca embaralhar séries temporais
    double soma = 0; int k = 0;
    for (int origem = 120; origem + 12 <= n; origem += 12) {
        Vec tr(y.begin(), y.begin() + origem), te(y.begin() + origem, y.begin() + origem + 12);
        soma += mae(te, forecast_ar(tr, 12, 12)); ++k;
    }
    printf("\nWalk-forward (AR(12), horizonte 12, %d origens): MAE medio = %.3f\n", k, soma / k);
}
