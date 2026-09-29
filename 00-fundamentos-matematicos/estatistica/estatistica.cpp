// Estatística: média, variância, covariância, correlação, MLE gaussiano,
// intervalo de confiança e bootstrap.
// Build: g++ -std=c++17 -O2 estatistica.cpp -o estatistica
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;

double media(const Vec& x) { return accumulate(x.begin(), x.end(), 0.0) / x.size(); }

// Variância amostral (não viesada, divide por n-1).
double variancia(const Vec& x) {
    double m = media(x), s = 0;
    for (double v : x) s += (v - m) * (v - m);
    return s / (x.size() - 1);
}

double covariancia(const Vec& x, const Vec& y) {
    double mx = media(x), my = media(y), s = 0;
    for (size_t i = 0; i < x.size(); ++i) s += (x[i] - mx) * (y[i] - my);
    return s / (x.size() - 1);
}

double correlacao(const Vec& x, const Vec& y) {
    return covariancia(x, y) / sqrt(variancia(x) * variancia(y));
}

double mediana(Vec x) {
    sort(x.begin(), x.end());
    size_t n = x.size();
    return n % 2 ? x[n / 2] : (x[n / 2 - 1] + x[n / 2]) / 2;
}

int main() {
    mt19937 rng(1);
    normal_distribution<double> N(5.0, 2.0);  // verdadeiros: mu=5, sigma=2
    Vec x(500);
    for (auto& v : x) v = N(rng);

    // MLE gaussiano: mu = média amostral, sigma^2 = (1/n) * soma (x-mu)^2 (viesado!)
    double mu = media(x);
    double s2_mle = variancia(x) * (x.size() - 1) / x.size();
    cout << "MLE: mu=" << mu << " sigma=" << sqrt(s2_mle) << " (verdadeiros 5 e 2)\n";

    // IC 95% para a média (aprox. normal): mu +- 1.96 * s/sqrt(n)
    double se = sqrt(variancia(x) / x.size());
    cout << "IC95% da media: [" << mu - 1.96 * se << ", " << mu + 1.96 * se << "]\n";

    // Bootstrap: reamostrar com reposição para estimar a incerteza da mediana
    Vec meds;
    uniform_int_distribution<size_t> idx(0, x.size() - 1);
    for (int b = 0; b < 2000; ++b) {
        Vec r(x.size());
        for (auto& v : r) v = x[idx(rng)];
        meds.push_back(mediana(r));
    }
    sort(meds.begin(), meds.end());
    cout << "mediana=" << mediana(x) << " bootstrap IC95%=[" << meds[50] << ", " << meds[1949] << "]\n";

    // Correlação: y = 2x + ruído
    Vec y(x.size());
    normal_distribution<double> ruido(0, 3);
    for (size_t i = 0; i < x.size(); ++i) y[i] = 2 * x[i] + ruido(rng);
    cout << "cov(x,y)=" << covariancia(x, y) << " corr(x,y)=" << correlacao(x, y) << "\n";
}
