// REINFORCE (gradiente de política) com baseline no CartPole, com física implementada do zero.
// Política: P(empurrar direita | s) = sigmoide(w . s + b) — o próprio gradiente da log-política é (a - p) * s.
// Build: g++ -std=c++17 -O2 reinforce.cpp -o reinforce
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;

// CartPole (mesmas equações do Gym): estado = (x, x_dot, theta, theta_dot)
struct CartPole {
    double x, xd, th, thd; int t;
    mt19937& rng;
    CartPole(mt19937& r) : rng(r) { reset(); }
    Vec reset() {
        uniform_real_distribution<double> U(-0.05, 0.05);
        x = U(rng); xd = U(rng); th = U(rng); thd = U(rng); t = 0;
        return state();
    }
    Vec state() const { return {x, xd, th, thd}; }
    // retorna true se o episódio terminou
    bool step(int action) {
        const double g = 9.8, mc = 1.0, mp = 0.1, l = 0.5, F = 10.0, tau = 0.02, mt = mc + mp, pml = mp * l;
        double force = action ? F : -F, ct = cos(th), st = sin(th);
        double temp = (force + pml * thd * thd * st) / mt;
        double tha = (g * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct * ct / mt));
        double xa = temp - pml * tha * ct / mt;
        x += tau * xd; xd += tau * xa; th += tau * thd; thd += tau * tha;
        ++t;
        return fabs(x) > 2.4 || fabs(th) > 12 * M_PI / 180 || t >= 500;
    }
};

Vec features(const Vec& s) { return {s[0] / 2.4, s[1] / 2.0, s[2] / 0.21, s[3] / 2.0, 1.0}; }  // normaliza + viés

int main() {
    mt19937 rng(0);
    CartPole env(rng);
    Vec w(5, 0.0);
    uniform_real_distribution<double> U(0, 1);
    const int BATCH = 10; const double gamma = 0.99, lr = 1.0;
    printf("iteracao | comprimento medio dos episodios (max 500)\n");
    for (int it = 1; it <= 300; ++it) {
        vector<vector<Vec>> S(BATCH); vector<vector<int>> Acts(BATCH); vector<vector<double>> G(BATCH);
        double len = 0;
        for (int e = 0; e < BATCH; ++e) {
            Vec s = env.reset(); bool done = false;
            while (!done) {
                Vec f = features(s);
                double z = 0;
                for (int i = 0; i < 5; ++i) z += w[i] * f[i];
                double p = 1 / (1 + exp(-z));
                int a = U(rng) < p;                       // amostra da política estocástica
                S[e].push_back(f); Acts[e].push_back(a);
                done = env.step(a);
                s = env.state();
            }
            int T = S[e].size(); len += T;
            G[e].assign(T, 0);
            double ret = 0;
            for (int t = T - 1; t >= 0; --t) { ret = 1.0 + gamma * ret; G[e][t] = ret; }  // retorno descontado (recompensa 1 por passo)
        }
        // baseline = média dos retornos do lote, normalizada por desvio padrão (reduz variância do gradiente)
        double mean = 0, var = 0; int n = 0;
        for (auto& g : G) for (double v : g) { mean += v; ++n; }
        mean /= n;
        for (auto& g : G) for (double v : g) var += (v - mean) * (v - mean);
        double sd = sqrt(var / n) + 1e-8;
        Vec grad(5, 0.0);
        for (int e = 0; e < BATCH; ++e)
            for (size_t t = 0; t < S[e].size(); ++t) {
                double z = 0;
                for (int i = 0; i < 5; ++i) z += w[i] * S[e][t][i];
                double p = 1 / (1 + exp(-z));
                double adv = (G[e][t] - mean) / sd;
                for (int i = 0; i < 5; ++i) grad[i] += (Acts[e][t] - p) * S[e][t][i] * adv;  // grad log pi * vantagem
            }
        for (int i = 0; i < 5; ++i) w[i] += lr * grad[i] / n;  // subida de gradiente
        if (it % 30 == 0 || it == 1) printf("  %4d   | %.1f\n", it, len / BATCH);
    }
    // avaliação da política final (estocástica)
    double total = 0; int E = 50;
    for (int e = 0; e < E; ++e) {
        Vec s = env.reset(); bool done = false;
        while (!done) {
            Vec f = features(s); double z = 0;
            for (int i = 0; i < 5; ++i) z += w[i] * f[i];
            done = env.step(U(rng) < 1 / (1 + exp(-z)));
            s = env.state();
        }
        total += env.t;
    }
    printf("avaliacao final: comprimento medio = %.1f em %d episodios (politica aleatoria ~22)\n", total / E, E);
}
