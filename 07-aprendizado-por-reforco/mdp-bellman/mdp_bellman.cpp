// MDP em um gridworld 4x4: Iteração de Valor e Iteração de Política resolvendo as equações de Bellman.
// Terminais nos cantos (0,0) e (3,3); recompensa -1 por passo; movimentos com ruído (80% certo, 10% para cada lado).
// Build: g++ -std=c++17 -O2 mdp_bellman.cpp -o mdp_bellman
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

const int N = 4, A = 4;                       // ações: 0=cima 1=direita 2=baixo 3=esquerda
const int DR[A] = {-1, 0, 1, 0}, DC[A] = {0, 1, 0, -1};
const char* SETA = "^>v<";

bool terminal(int s) { return s == 0 || s == N * N - 1; }

int move(int s, int a) {  // estado seguinte (bater na parede = ficar)
    int r = s / N + DR[a], c = s % N + DC[a];
    if (r < 0 || r >= N || c < 0 || c >= N) return s;
    return r * N + c;
}

// P(s'|s,a): a ação sai como planejada com 0.8; desliza para os lados com 0.1 cada
double q_value(const vector<double>& V, int s, int a, double gamma) {
    double q = 0;
    const double prob[3] = {0.8, 0.1, 0.1};
    const int dev[3] = {0, 1, 3};
    for (int k = 0; k < 3; ++k) q += prob[k] * (-1.0 + gamma * V[move(s, (a + dev[k]) % A)]);
    return q;
}

void print_V(const vector<double>& V, const vector<int>& pi) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int s = r * N + c;
            if (terminal(s)) printf("   T    ");
            else printf("%6.2f%c ", V[s], SETA[pi[s]]);
        }
        printf("\n");
    }
}

vector<int> greedy(const vector<double>& V, double gamma) {
    vector<int> pi(N * N, 0);
    for (int s = 0; s < N * N; ++s) {
        double best = -1e18;
        for (int a = 0; a < A; ++a) { double q = q_value(V, s, a, gamma); if (q > best + 1e-12) { best = q; pi[s] = a; } }
    }
    return pi;
}

int main() {
    double gamma = 1.0;

    // ---- Iteração de valor: V(s) <- max_a sum_s' P(s'|s,a) [r + gamma V(s')]  (equação de otimalidade de Bellman)
    vector<double> V(N * N, 0.0);
    int it = 0;
    for (;; ++it) {
        double delta = 0;
        for (int s = 0; s < N * N; ++s) {
            if (terminal(s)) continue;
            double best = -1e18;
            for (int a = 0; a < A; ++a) best = max(best, q_value(V, s, a, gamma));
            delta = max(delta, fabs(best - V[s]));
            V[s] = best;
        }
        if (delta < 1e-10) break;
    }
    printf("Iteracao de valor convergiu em %d varreduras. V* e politica gulosa:\n", it + 1);
    print_V(V, greedy(V, gamma));

    // ---- Iteração de política: avaliar a política (Bellman de expectativa) + melhorar (gulosa), até estabilizar
    vector<int> pi(N * N, 0);
    vector<double> Vp(N * N, 0.0);
    int rounds = 0;
    for (;; ++rounds) {
        for (int k = 0; k < 10000; ++k) {  // avaliação da política
            double delta = 0;
            for (int s = 0; s < N * N; ++s) {
                if (terminal(s)) continue;
                double v = q_value(Vp, s, pi[s], gamma);
                delta = max(delta, fabs(v - Vp[s])); Vp[s] = v;
            }
            if (delta < 1e-10) break;
        }
        vector<int> novo = greedy(Vp, gamma);  // melhoria
        if (novo == pi) break;
        pi = novo;
    }
    double dif = 0;
    for (int s = 0; s < N * N; ++s) dif = max(dif, fabs(V[s] - Vp[s]));
    printf("\nIteracao de politica convergiu em %d rodadas de melhoria; max|V_valor - V_politica| = %.1e (mesmos valores; politicas podem diferir so em empates)\n", rounds + 1, dif);

    // ---- efeito do fator de desconto
    printf("\nValor do estado (3,2) [1 passo do terminal] para diferentes gamma:\n");
    for (double g : {0.5, 0.9, 0.99, 1.0}) {
        vector<double> W(N * N, 0.0);
        for (int k = 0; k < 5000; ++k) for (int s = 0; s < N * N; ++s) { if (terminal(s)) continue; double b = -1e18; for (int a = 0; a < A; ++a) b = max(b, q_value(W, s, a, g)); W[s] = b; }
        printf("  gamma=%.2f  V(3,2)=%.3f  V(0,3)=%.3f\n", g, W[3 * N + 2], W[3]);
    }
}
