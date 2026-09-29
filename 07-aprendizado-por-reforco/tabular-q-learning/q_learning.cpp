// Q-learning (off-policy) vs SARSA (on-policy) no "Cliff Walking" 4x12 (Sutton & Barto, ex. 6.6).
// Q-learning aprende o caminho ótimo rente ao penhasco; SARSA, que leva a exploração em conta, um caminho mais seguro.
// Build: g++ -std=c++17 -O2 q_learning.cpp -o q_learning
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;

const int R = 4, C = 12, A = 4;                     // 0=cima 1=direita 2=baixo 3=esquerda
const int DR[A] = {-1, 0, 1, 0}, DC[A] = {0, 1, 0, -1};
const int START = 3 * C, GOAL = 3 * C + 11;

// Passo: penhasco = linha 3, colunas 1..10 -> recompensa -100 e volta ao início; demais passos -1
int step(int s, int a, double& r, bool& done) {
    int row = s / C + DR[a], col = s % C + DC[a];
    row = max(0, min(R - 1, row)); col = max(0, min(C - 1, col));
    done = false; r = -1;
    if (row == 3 && col >= 1 && col <= 10) { r = -100; return START; }
    int ns = row * C + col;
    if (ns == GOAL) done = true;
    return ns;
}

int eps_greedy(const vector<vector<double>>& Q, int s, double eps, mt19937& rng) {
    if (uniform_real_distribution<double>(0, 1)(rng) < eps) return uniform_int_distribution<int>(0, A - 1)(rng);
    int b = 0;
    for (int a = 1; a < A; ++a) if (Q[s][a] > Q[s][b]) b = a;
    return b;
}

// Roda um algoritmo por 500 episódios; retorna Q e a recompensa média dos episódios (desempenho ONLINE)
double treina(bool sarsa, vector<vector<double>>& Q, mt19937& rng, double alpha = 0.5, double gamma = 1.0, double eps = 0.1) {
    Q.assign(R * C, vector<double>(A, 0.0));
    double total = 0; int n = 500;
    for (int ep = 0; ep < n; ++ep) {
        int s = START, a = eps_greedy(Q, s, eps, rng);
        for (int t = 0; t < 1000; ++t) {
            double r; bool done;
            int ns = step(s, a, r, done);
            int na = eps_greedy(Q, ns, eps, rng);
            double alvo;
            if (sarsa) alvo = r + (done ? 0 : gamma * Q[ns][na]);  // SARSA: usa a ação que REALMENTE será tomada
            else alvo = r + (done ? 0 : gamma * *max_element(Q[ns].begin(), Q[ns].end()));  // Q-learning: usa o max
            Q[s][a] += alpha * (alvo - Q[s][a]);                   // atualização por diferença temporal
            total += r; s = ns; a = na;
            if (done) break;
        }
    }
    return total / n;
}

void mostra_caminho(const vector<vector<double>>& Q) {
    vector<string> g(R, string(C, '.'));
    for (int c = 1; c <= 10; ++c) g[3][c] = 'C';
    int s = START;
    for (int t = 0; t < 100 && s != GOAL; ++t) {
        int a = max_element(Q[s].begin(), Q[s].end()) - Q[s].begin();
        if (s != START) g[s / C][s % C] = "^>v<"[a];
        double r; bool d; s = step(s, a, r, d);
    }
    g[3][0] = 'S'; g[3][11] = 'G';
    for (auto& row : g) cout << "    " << row << "\n";
}

int main() {
    mt19937 rng(0);
    for (bool sarsa : {false, true}) {
        double media = 0; vector<vector<double>> Q;
        int runs = 50;
        for (int k = 0; k < runs; ++k) { vector<vector<double>> q; media += treina(sarsa, q, rng) / runs; if (k == 0) Q = q; }
        printf("%s: recompensa media por episodio durante o treino (eps=0.1) = %.1f\n", sarsa ? "SARSA     " : "Q-learning", media);
        printf("  caminho guloso aprendido (C = penhasco):\n");
        mostra_caminho(Q);
    }
    cout << "Q-learning acha o caminho otimo (13 passos) rente ao penhasco, mas cai de vez em quando por causa da exploracao;\n"
            "SARSA aprende que explorar perto do penhasco e perigoso e escolhe a rota mais segura (melhor desempenho online).\n";
}
