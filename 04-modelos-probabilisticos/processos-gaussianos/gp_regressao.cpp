// Regressão por Processo Gaussiano (kernel RBF): média e variância preditivas via Cholesky
// e escolha do comprimento de escala pela log-verossimilhança marginal.
// Build: g++ -std=c++17 -O2 gp_regressao.cpp -o gp_regressao
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double rbf(double a, double b, double ell, double sf2) { return sf2 * exp(-0.5 * (a - b) * (a - b) / (ell * ell)); }

// Cholesky: K = L L^T (L triangular inferior)
Mat cholesky(const Mat& K) {
    int n = K.size();
    Mat L(n, Vec(n, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = K[i][j];
            for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            L[i][j] = i == j ? sqrt(s) : s / L[j][j];
        }
    return L;
}
Vec solve_L(const Mat& L, Vec b) {  // L x = b
    for (size_t i = 0; i < b.size(); ++i) { for (size_t k = 0; k < i; ++k) b[i] -= L[i][k] * b[k]; b[i] /= L[i][i]; }
    return b;
}
Vec solve_LT(const Mat& L, Vec b) {  // L^T x = b
    for (int i = b.size() - 1; i >= 0; --i) { for (size_t k = i + 1; k < b.size(); ++k) b[i] -= L[k][i] * b[k]; b[i] /= L[i][i]; }
    return b;
}

struct GP {
    Vec x, y; double ell, sf2, sn2; Mat L; Vec alpha;
    void fit(const Vec& x_, const Vec& y_, double ell_, double sf2_, double sn2_) {
        x = x_; y = y_; ell = ell_; sf2 = sf2_; sn2 = sn2_;
        int n = x.size();
        Mat K(n, Vec(n));
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) K[i][j] = rbf(x[i], x[j], ell, sf2) + (i == j ? sn2 : 0);
        L = cholesky(K);
        alpha = solve_LT(L, solve_L(L, y));  // alpha = K^-1 y
    }
    // média = k*^T alpha;  variância = k(x*,x*) - v^T v, com v = L^-1 k*
    void predict(double xs, double& mean, double& var) const {
        int n = x.size();
        Vec ks(n);
        for (int i = 0; i < n; ++i) ks[i] = rbf(xs, x[i], ell, sf2);
        mean = 0;
        for (int i = 0; i < n; ++i) mean += ks[i] * alpha[i];
        Vec v = solve_L(L, ks);
        var = rbf(xs, xs, ell, sf2);
        for (double t : v) var -= t * t;
    }
    // log p(y|X) = -1/2 y^T K^-1 y - 1/2 log|K| - n/2 log 2pi   (log|K| = 2 sum log L_ii)
    double log_marginal() const {
        int n = x.size();
        double ll = 0;
        for (int i = 0; i < n; ++i) ll += -0.5 * y[i] * alpha[i] - log(L[i][i]);
        return ll - 0.5 * n * log(2 * M_PI);
    }
};

int main() {
    mt19937 rng(0);
    uniform_real_distribution<double> U(-4, 4);
    normal_distribution<double> N(0, 0.15);
    Vec x, y;
    for (int i = 0; i < 25; ++i) { double xi = U(rng); if (xi > 0.5 && xi < 2) continue; x.push_back(xi); y.push_back(sin(xi) + N(rng)); }  // "buraco" em (0.5, 2)
    printf("n = %zu pontos (sem dados em [0.5, 2])\n\n", x.size());

    printf("comprimento de escala | log p(y|X)\n");
    double best_ell = 1, best = -1e300;
    for (double ell : {0.1, 0.3, 0.6, 1.0, 1.5, 3.0, 6.0}) {
        GP g; g.fit(x, y, ell, 1.0, 0.15 * 0.15);
        double lm = g.log_marginal();
        printf("        %4.1f          | %8.2f\n", ell, lm);
        if (lm > best) { best = lm; best_ell = ell; }
    }
    GP g; g.fit(x, y, best_ell, 1.0, 0.15 * 0.15);
    printf("\nmelhor ell = %.1f\n   x   | media  | +-2 desvios | sin(x) verdadeiro\n", best_ell);
    for (double xs : {-3.0, -1.0, 1.25, 3.0, 6.0}) {
        double m, v;
        g.predict(xs, m, v);
        printf(" %5.2f | %6.3f |   %.3f     | %6.3f%s\n", xs, m, 2 * sqrt(max(v, 0.0)), sin(xs), (xs == 1.25 ? "   <- buraco: incerteza maior" : (xs == 6.0 ? "   <- extrapolacao: incerteza alta" : "")));
    }
}
