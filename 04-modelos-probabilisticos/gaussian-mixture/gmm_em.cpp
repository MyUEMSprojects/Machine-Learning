// Mistura de Gaussianas (GMM) em 2D com covariância completa, ajustada pelo algoritmo EM.
// Build: g++ -std=c++17 -O2 gmm_em.cpp -o gmm_em
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

struct Gauss2 {
    double mx = 0, my = 0, sxx = 1, sxy = 0, syy = 1;
    double pdf(double x, double y) const {
        double det = sxx * syy - sxy * sxy;
        double dx = x - mx, dy = y - my;
        double q = (syy * dx * dx - 2 * sxy * dx * dy + sxx * dy * dy) / det;  // forma quadrática (x-mu)^T S^-1 (x-mu)
        return exp(-0.5 * q) / (2 * M_PI * sqrt(det));
    }
};

struct GMM {
    int K; Vec pi; vector<Gauss2> g;

    // Retorna a log-verossimilhança final
    double fit(const Mat& X, int K_, mt19937& rng, int max_iter = 200, double tol = 1e-8) {
        K = K_;
        int n = X.size();
        pi.assign(K, 1.0 / K); g.assign(K, {});
        for (int k = 0; k < K; ++k) {  // inicializa médias em pontos aleatórios
            const Vec& p = X[uniform_int_distribution<int>(0, n - 1)(rng)];
            g[k].mx = p[0]; g[k].my = p[1]; g[k].sxx = g[k].syy = 2.0; g[k].sxy = 0;
        }
        Mat R(n, Vec(K));
        double prev = -1e300, ll = 0;
        for (int it = 0; it < max_iter; ++it) {
            // Passo E: responsabilidades r_ik = P(componente k | x_i)
            ll = 0;
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int k = 0; k < K; ++k) { R[i][k] = pi[k] * g[k].pdf(X[i][0], X[i][1]); s += R[i][k]; }
                for (int k = 0; k < K; ++k) R[i][k] /= s;
                ll += log(s);
            }
            // Passo M: re-estima parâmetros ponderando pelas responsabilidades
            for (int k = 0; k < K; ++k) {
                double Nk = 0, mx = 0, my = 0;
                for (int i = 0; i < n; ++i) { Nk += R[i][k]; mx += R[i][k] * X[i][0]; my += R[i][k] * X[i][1]; }
                mx /= Nk; my /= Nk;
                double sxx = 0, sxy = 0, syy = 0;
                for (int i = 0; i < n; ++i) {
                    double dx = X[i][0] - mx, dy = X[i][1] - my;
                    sxx += R[i][k] * dx * dx; sxy += R[i][k] * dx * dy; syy += R[i][k] * dy * dy;
                }
                pi[k] = Nk / n;
                g[k] = {mx, my, sxx / Nk + 1e-6, sxy / Nk, syy / Nk + 1e-6};  // 1e-6: evita covariância singular
            }
            if (fabs(ll - prev) < tol) break;  // EM nunca diminui a log-verossimilhança
            prev = ll;
        }
        return ll;
    }
    int predict(double x, double y) const {
        int best = 0; double bp = -1;
        for (int k = 0; k < K; ++k) { double p = pi[k] * g[k].pdf(x, y); if (p > bp) { bp = p; best = k; } }
        return best;
    }
};

int main() {
    mt19937 rng(1);
    normal_distribution<double> N(0, 1);
    Mat X;
    for (int i = 0; i < 900; ++i) {
        int c = i % 3;
        if (c == 0) X.push_back({-3 + 0.6 * N(rng), 0 + 0.6 * N(rng)});
        else if (c == 1) { double a = N(rng), b = N(rng); X.push_back({3 + 1.5 * a, 2 + 0.5 * b + 0.7 * a}); }  // elipse inclinada
        else X.push_back({0 + 0.8 * N(rng), -4 + 0.8 * N(rng)});
    }
    // várias inicializações; fica com a de maior log-verossimilhança
    GMM best; double bll = -1e300;
    for (int r = 0; r < 10; ++r) { GMM m; double ll = m.fit(X, 3, rng); if (ll > bll) { bll = ll; best = m; } }
    printf("log-verossimilhanca = %.1f\n", bll);
    for (int k = 0; k < 3; ++k)
        printf("componente %d: peso=%.3f  media=(%.2f, %.2f)  cov=[[%.2f, %.2f],[%.2f, %.2f]]\n", k, best.pi[k], best.g[k].mx, best.g[k].my,
               best.g[k].sxx, best.g[k].sxy, best.g[k].sxy, best.g[k].syy);

    // seleção de K por BIC = p ln n - 2 lnL (menor é melhor); p = K*(2+3) + (K-1)
    cout << "\n K | BIC\n";
    for (int K = 1; K <= 6; ++K) {
        double bl = -1e300;
        for (int r = 0; r < 8; ++r) { GMM m; bl = max(bl, m.fit(X, K, rng)); }
        int p = K * 5 + (K - 1);
        printf("%2d | %.1f\n", K, p * log((double)X.size()) - 2 * bl);
    }
}
