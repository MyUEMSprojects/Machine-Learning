// Inferência bayesiana: (1) atualização conjugada Beta–Binomial (forma fechada) e
// (2) MCMC Metropolis–Hastings para o posterior de uma média com prior não conjugado (Laplace).
// Build: g++ -std=c++17 -O2 inferencia_bayesiana.cpp -o inferencia_bayesiana
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

int main() {
    mt19937 rng(0);

    // ---------- 1) Beta-Binomial ----------
    // Prior Beta(a,b); dados: k sucessos em n; posterior = Beta(a+k, b+n-k)
    double a = 2, b = 2;  // prior fraco, centrado em 0.5
    int n = 20, k = 14;
    double a2 = a + k, b2 = b + n - k;
    printf("Beta-Binomial: prior Beta(%.0f,%.0f) media=%.3f  |  dados %d/%d  |  posterior Beta(%.0f,%.0f) media=%.3f\n", a, b, a / (a + b), k, n, a2, b2, a2 / (a2 + b2));
    printf("  MLE = %.3f (ignora o prior); MAP = %.3f; media posterior = %.3f\n", (double)k / n, (a2 - 1) / (a2 + b2 - 2), a2 / (a2 + b2));

    // ---------- 2) Metropolis-Hastings ----------
    // Modelo: x_i ~ N(mu, 1); prior mu ~ Laplace(0, 1). Não há forma fechada => MCMC.
    normal_distribution<double> N(1.5, 1.0);  // mu verdadeiro = 1.5
    vector<double> x(30);
    for (double& v : x) v = N(rng);
    auto log_post = [&](double mu) {  // log prior + log verossimilhança (a menos de constante)
        double lp = -fabs(mu);
        for (double v : x) lp += -0.5 * (v - mu) * (v - mu);
        return lp;
    };
    normal_distribution<double> passo(0, 0.5);
    uniform_real_distribution<double> U(0, 1);
    double mu = 0, lp = log_post(mu);
    vector<double> amostras; int aceitos = 0;
    for (int it = 0; it < 60000; ++it) {
        double prop = mu + passo(rng), lpp = log_post(prop);
        if (log(U(rng)) < lpp - lp) { mu = prop; lp = lpp; ++aceitos; }  // aceita com prob min(1, p'/p)
        if (it >= 5000) amostras.push_back(mu);  // descarta burn-in
    }
    sort(amostras.begin(), amostras.end());
    double media = 0;
    for (double s : amostras) media += s / amostras.size();
    double xbar = 0;
    for (double v : x) xbar += v / x.size();
    printf("\nMetropolis-Hastings: taxa de aceitacao = %.2f\n", (double)aceitos / 60000);
    printf("  media amostral (MLE) = %.3f | media posterior = %.3f | IC 95%% credivel = [%.3f, %.3f]\n", xbar, media,
           amostras[amostras.size() * 0.025], amostras[amostras.size() * 0.975]);
    printf("  (o prior de Laplace puxa levemente a media posterior para 0)\n");
}
