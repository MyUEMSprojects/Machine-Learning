// Otimização Bayesiana 1D: modelo substituto (Processo Gaussiano) + função de aquisição Expected Improvement,
// comparada com busca aleatória, na função de Forrester (cara de avaliar na vida real): f(x) = (6x-2)^2 sin(12x-4).
// Build: g++ -std=c++17 -O2 bayes_opt.cpp -o bayes_opt
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

double forrester(double x) { return pow(6 * x - 2, 2) * sin(12 * x - 4); }  // mínimo global ~ -6.02 em x ~ 0.757
double kern(double a, double b, double ell) { return exp(-0.5 * (a - b) * (a - b) / (ell * ell)); }

Mat cholesky(const Mat& K) {
    int n = K.size(); Mat L(n, Vec(n, 0));
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) {
        double s = K[i][j];
        for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
        L[i][j] = i == j ? sqrt(max(s, 1e-12)) : s / L[j][j];
    }
    return L;
}
Vec fwd(const Mat& L, Vec b) { for (size_t i = 0; i < b.size(); ++i) { for (size_t k = 0; k < i; ++k) b[i] -= L[i][k] * b[k]; b[i] /= L[i][i]; } return b; }
Vec bwd(const Mat& L, Vec b) { for (int i = b.size() - 1; i >= 0; --i) { for (size_t k = i + 1; k < b.size(); ++k) b[i] -= L[k][i] * b[k]; b[i] /= L[i][i]; } return b; }

struct GP {
    Vec x, y; double ell, noise = 1e-6; Mat L; Vec alpha; double ymean, ystd;
    void fit(const Vec& x_, const Vec& y_, double ell_) {
        x = x_; ell = ell_;
        ymean = 0; for (double v : y_) ymean += v / y_.size();
        ystd = 0; for (double v : y_) ystd += pow(v - ymean, 2) / y_.size();
        ystd = sqrt(ystd) + 1e-9;
        y.clear(); for (double v : y_) y.push_back((v - ymean) / ystd);   // padroniza o alvo
        int n = x.size(); Mat K(n, Vec(n));
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) K[i][j] = kern(x[i], x[j], ell) + (i == j ? noise : 0);
        L = cholesky(K); alpha = bwd(L, fwd(L, y));
    }
    double log_marg() const { double ll = 0; for (size_t i = 0; i < x.size(); ++i) ll += -0.5 * y[i] * alpha[i] - log(L[i][i]); return ll; }
    void predict(double xs, double& mu, double& sd) const {  // na escala original
        int n = x.size(); Vec ks(n);
        for (int i = 0; i < n; ++i) ks[i] = kern(xs, x[i], ell);
        double m = 0; for (int i = 0; i < n; ++i) m += ks[i] * alpha[i];
        Vec v = fwd(L, ks); double var = 1;
        for (double t : v) var -= t * t;
        mu = m * ystd + ymean; sd = sqrt(max(var, 1e-12)) * ystd;
    }
};

double Phi(double z) { return 0.5 * erfc(-z / sqrt(2.0)); }
double phi(double z) { return exp(-0.5 * z * z) / sqrt(2 * M_PI); }

// Expected Improvement (minimização): E[max(fmin - f(x), 0)] = (fmin-mu) Phi(z) + sd phi(z), z = (fmin-mu)/sd
double EI(double mu, double sd, double fmin) {
    if (sd < 1e-12) return 0;
    double z = (fmin - mu) / sd;
    return (fmin - mu) * Phi(z) + sd * phi(z);
}

int main() {
    const int BUDGET = 15, INIT = 3, RUNS = 50;
    mt19937 rng(0);
    uniform_real_distribution<double> U(0, 1);
    vector<double> grid; for (int i = 0; i <= 500; ++i) grid.push_back(i / 500.0);

    double bo_sum = 0, rs_sum = 0; int bo_ok = 0, rs_ok = 0;
    vector<double> traj_bo;
    for (int run = 0; run < RUNS; ++run) {
        Vec x, y;
        for (int i = 0; i < INIT; ++i) { x.push_back(U(rng)); y.push_back(forrester(x.back())); }
        Vec xr = x, yr = y;
        for (int it = INIT; it < BUDGET; ++it) {
            // 1) ajusta o GP (escolhendo o comprimento de escala por verossimilhança marginal)
            GP best; double bl = -1e18;
            for (double ell : {0.05, 0.1, 0.2, 0.4}) { GP g; g.fit(x, y, ell); if (g.log_marg() > bl) { bl = g.log_marg(); best = g; } }
            // 2) maximiza a aquisição EI sobre a grade e avalia a função verdadeira nesse ponto
            double fmin = *min_element(y.begin(), y.end()), bx = 0, be = -1;
            for (double g : grid) { double mu, sd; best.predict(g, mu, sd); double e = EI(mu, sd, fmin); if (e > be) { be = e; bx = g; } }
            x.push_back(bx); y.push_back(forrester(bx));
            xr.push_back(U(rng)); yr.push_back(forrester(xr.back()));  // busca aleatória: mesmo orçamento
        }
        double bo = *min_element(y.begin(), y.end()), rs = *min_element(yr.begin(), yr.end());
        bo_sum += bo / RUNS; rs_sum += rs / RUNS; bo_ok += bo < -5.9; rs_ok += rs < -5.9;
        if (run == 0) traj_bo = y;
    }
    printf("Forrester: minimo verdadeiro = -6.021. Orcamento = %d avaliacoes (%d iniciais), %d repeticoes\n", BUDGET, INIT, RUNS);
    printf("  Otimizacao Bayesiana (GP+EI): melhor valor medio = %.3f | achou o otimo (<-5.9) em %d/%d execucoes\n", bo_sum, bo_ok, RUNS);
    printf("  Busca aleatoria             : melhor valor medio = %.3f | achou o otimo (<-5.9) em %d/%d execucoes\n", rs_sum, rs_ok, RUNS);
    printf("\nPontos avaliados na 1a execucao (BO explora e depois concentra perto do otimo x~0.757):\n  ");
    for (double v : traj_bo) printf("%.2f ", v);
    printf("\n");
}
