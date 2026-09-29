// Monitoramento de deriva de dados (data drift) do zero: PSI (Population Stability Index),
// estatística de Kolmogorov–Smirnov e divergência de Jensen–Shannon entre a distribuição de REFERÊNCIA (treino)
// e a de PRODUÇÃO, em vários cenários (sem deriva, deslocamento de média, mudança de variância).
// Build: g++ -std=c++17 -O2 drift.cpp -o drift
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;

// Bins definidos pelos QUANTIS da referência (cada bin tem ~10% dos dados de referência)
Vec proporcoes(const Vec& x, const Vec& cortes) {
    Vec p(cortes.size() + 1, 0.0);
    for (double v : x) p[upper_bound(cortes.begin(), cortes.end(), v) - cortes.begin()] += 1.0 / x.size();
    return p;
}

// PSI = sum (p_prod - p_ref) * ln(p_prod / p_ref).  Regra de bolso: <0.1 estável; 0.1-0.25 atenção; >0.25 deriva forte
double psi(const Vec& ref, const Vec& prod, int bins = 10) {
    Vec r = ref; sort(r.begin(), r.end());
    Vec cortes;
    for (int i = 1; i < bins; ++i) cortes.push_back(r[r.size() * i / bins]);
    Vec pr = proporcoes(ref, cortes), pp = proporcoes(prod, cortes);
    double s = 0;
    for (size_t i = 0; i < pr.size(); ++i) { double a = max(pr[i], 1e-4), b = max(pp[i], 1e-4); s += (b - a) * log(b / a); }
    return s;
}

// KS = sup |F_ref(x) - F_prod(x)| (maior distância entre as CDFs empíricas)
double ks(Vec a, Vec b) {
    sort(a.begin(), a.end()); sort(b.begin(), b.end());
    size_t i = 0, j = 0; double d = 0;
    while (i < a.size() && j < b.size()) {
        double x = min(a[i], b[j]);
        while (i < a.size() && a[i] <= x) ++i;
        while (j < b.size() && b[j] <= x) ++j;
        d = max(d, fabs((double)i / a.size() - (double)j / b.size()));
    }
    return d;
}

double js(const Vec& ref, const Vec& prod, int bins = 10) {
    Vec r = ref; sort(r.begin(), r.end());
    Vec cortes;
    for (int i = 1; i < bins; ++i) cortes.push_back(r[r.size() * i / bins]);
    Vec p = proporcoes(ref, cortes), q = proporcoes(prod, cortes);
    auto kl = [](const Vec& a, const Vec& b) { double s = 0; for (size_t i = 0; i < a.size(); ++i) if (a[i] > 0) s += a[i] * log2(a[i] / b[i]); return s; };
    Vec m(p.size());
    for (size_t i = 0; i < p.size(); ++i) m[i] = 0.5 * (p[i] + q[i]);
    return 0.5 * kl(p, m) + 0.5 * kl(q, m);
}

int main() {
    mt19937 rng(0);
    auto amostra = [&](int n, double mu, double sd) { normal_distribution<double> N(mu, sd); Vec x(n); for (double& v : x) v = N(rng); return x; };
    Vec ref = amostra(5000, 0, 1);
    struct Cen { string nome; Vec prod; };
    vector<Cen> cenarios = {
        {"sem deriva (mesma dist.)     ", amostra(2000, 0, 1)},
        {"media deslocada +0.2         ", amostra(2000, 0.2, 1)},
        {"media deslocada +0.5         ", amostra(2000, 0.5, 1)},
        {"variancia dobrada (sd=2)     ", amostra(2000, 0, 2)},
        {"deriva forte (media +1.5)    ", amostra(2000, 1.5, 1)},
    };
    printf("cenario                        |  PSI   |  KS   |  JS(bits) | alerta (PSI)\n");
    for (auto& c : cenarios) {
        double p = psi(ref, c.prod);
        printf("%s | %.3f  | %.3f |  %.4f   | %s\n", c.nome.c_str(), p, ks(ref, c.prod), js(ref, c.prod), p < 0.1 ? "ok" : (p < 0.25 ? "ATENCAO" : "DERIVA -> retreinar?"));
    }
    printf("\nDeriva de dados nem sempre derruba a acuracia (e a acuracia so pode ser medida quando os rotulos chegam, com atraso);\n"
           "por isso monitora-se tambem a distribuicao das entradas e das previsoes.\n");
}
