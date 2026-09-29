// Otimizadores de redes neurais: SGD, Momentum, Nesterov, RMSprop, Adam, AdamW — comparados em
// uma regressão logística mal condicionada (features em escalas muito diferentes).
// Build: g++ -std=c++17 -O2 otimizadores.cpp -o otimizadores
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

Mat X; vector<int> Y;

double loss_grad(const Vec& w, Vec& g) {  // entropia cruzada média + gradiente
    int n = X.size(), d = w.size();
    g.assign(d, 0.0);
    double L = 0;
    for (int i = 0; i < n; ++i) {
        double z = 0;
        for (int j = 0; j < d; ++j) z += w[j] * X[i][j];
        double p = 1 / (1 + exp(-z));
        L += -(Y[i] * log(p + 1e-12) + (1 - Y[i]) * log(1 - p + 1e-12)) / n;
        for (int j = 0; j < d; ++j) g[j] += (p - Y[i]) * X[i][j] / n;
    }
    return L;
}

struct Opt {
    virtual void step(Vec& w, const Vec& g) = 0;
    virtual ~Opt() = default;
};
struct SGD : Opt {
    double lr; SGD(double lr) : lr(lr) {}
    void step(Vec& w, const Vec& g) override { for (size_t i = 0; i < w.size(); ++i) w[i] -= lr * g[i]; }
};
struct Momentum : Opt {  // v <- b v + g ;  w <- w - lr v
    double lr, b; Vec v; Momentum(double lr, double b = 0.9) : lr(lr), b(b) {}
    void step(Vec& w, const Vec& g) override {
        if (v.empty()) v.assign(w.size(), 0);
        for (size_t i = 0; i < w.size(); ++i) { v[i] = b * v[i] + g[i]; w[i] -= lr * v[i]; }
    }
};
struct RMSprop : Opt {  // divide o passo pela raiz da média móvel de g^2
    double lr, rho, eps; Vec s; RMSprop(double lr, double rho = 0.9, double eps = 1e-8) : lr(lr), rho(rho), eps(eps) {}
    void step(Vec& w, const Vec& g) override {
        if (s.empty()) s.assign(w.size(), 0);
        for (size_t i = 0; i < w.size(); ++i) { s[i] = rho * s[i] + (1 - rho) * g[i] * g[i]; w[i] -= lr * g[i] / (sqrt(s[i]) + eps); }
    }
};
struct Adam : Opt {  // momentum + RMSprop + correção de viés; decoupled = AdamW
    double lr, b1, b2, eps, wd; bool decoupled; Vec m, v; int t = 0;
    Adam(double lr, double wd = 0, bool decoupled = false, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : lr(lr), b1(b1), b2(b2), eps(eps), wd(wd), decoupled(decoupled) {}
    void step(Vec& w, const Vec& g) override {
        if (m.empty()) { m.assign(w.size(), 0); v.assign(w.size(), 0); }
        ++t;
        for (size_t i = 0; i < w.size(); ++i) {
            double gi = g[i] + (decoupled ? 0 : wd * w[i]);
            m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
            double mh = m[i] / (1 - pow(b1, t)), vh = v[i] / (1 - pow(b2, t));
            w[i] -= lr * mh / (sqrt(vh) + eps);
            if (decoupled) w[i] -= lr * wd * w[i];  // AdamW: weight decay desacoplado do gradiente
        }
    }
};

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    // 3 features com escalas 1, 30, 0.03 (mal condicionado) + viés
    for (int i = 0; i < 400; ++i) {
        double a = N(rng), b = N(rng), c = N(rng);
        double z = 1.0 * a + 0.05 * b * 30 - 20.0 * c * 0.03 * 1.0;
        X.push_back({a, 30 * b, 0.03 * c, 1.0});
        Y.push_back(z + 0.5 * N(rng) > 0);
    }
    printf("%-14s | loss apos 50 | 200 | 1000 passos\n", "otimizador");
    struct Cfg { string nome; function<unique_ptr<Opt>()> make; };
    vector<Cfg> cfgs = {
        {"SGD lr=0.001", [] { return make_unique<SGD>(0.001); }},
        {"SGD lr=0.01", [] { return make_unique<SGD>(0.01); }},
        {"Momentum", [] { return make_unique<Momentum>(0.001); }},
        {"RMSprop", [] { return make_unique<RMSprop>(0.01); }},
        {"Adam", [] { return make_unique<Adam>(0.05); }},
        {"AdamW (wd=.01)", [] { return make_unique<Adam>(0.05, 0.01, true); }},
    };
    for (auto& c : cfgs) {
        auto opt = c.make();
        Vec w(4, 0.0), g;
        double l[3] = {0, 0, 0};
        for (int t = 1; t <= 1000; ++t) {
            double L = loss_grad(w, g);
            if (!std::isfinite(L)) { l[0] = l[1] = l[2] = NAN; break; }
            opt->step(w, g);
            if (t == 50) l[0] = L;
            if (t == 200) l[1] = L;
            if (t == 1000) l[2] = L;
        }
        printf("%-14s |   %.4f    | %.4f | %.4f\n", c.nome.c_str(), l[0], l[1], l[2]);
    }
    cout << "Adam/RMSprop adaptam o passo por parametro e lidam bem com escalas diferentes; SGD puro sofre.\n";
}
