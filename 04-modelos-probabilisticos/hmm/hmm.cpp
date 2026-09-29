// Hidden Markov Model discreto: forward (verossimilhança), Viterbi (sequência mais provável)
// e Baum–Welch (aprendizado por EM), com escalonamento para estabilidade numérica.
// Build: g++ -std=c++17 -O2 hmm.cpp -o hmm
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

struct HMM {
    int N, M;      // nº de estados ocultos, nº de símbolos observáveis
    Vec pi;        // P(estado inicial)
    Mat A, B;      // A[i][j] = P(j | i) transição; B[i][o] = P(o | i) emissão

    // Forward com escalonamento: retorna log P(obs); alpha[t][i] normalizado a cada passo.
    double forward(const vector<int>& o, Mat& alpha, Vec& c) const {
        int T = o.size();
        alpha.assign(T, Vec(N)); c.assign(T, 0);
        for (int i = 0; i < N; ++i) { alpha[0][i] = pi[i] * B[i][o[0]]; c[0] += alpha[0][i]; }
        for (int i = 0; i < N; ++i) alpha[0][i] /= c[0];
        for (int t = 1; t < T; ++t) {
            for (int j = 0; j < N; ++j) {
                double s = 0;
                for (int i = 0; i < N; ++i) s += alpha[t - 1][i] * A[i][j];
                alpha[t][j] = s * B[j][o[t]]; c[t] += alpha[t][j];
            }
            for (int j = 0; j < N; ++j) alpha[t][j] /= c[t];
        }
        double ll = 0;
        for (double x : c) ll += log(x);
        return ll;
    }

    void backward(const vector<int>& o, const Vec& c, Mat& beta) const {
        int T = o.size();
        beta.assign(T, Vec(N, 0));
        for (int i = 0; i < N; ++i) beta[T - 1][i] = 1.0 / c[T - 1];
        for (int t = T - 2; t >= 0; --t)
            for (int i = 0; i < N; ++i) {
                double s = 0;
                for (int j = 0; j < N; ++j) s += A[i][j] * B[j][o[t + 1]] * beta[t + 1][j];
                beta[t][i] = s / c[t];
            }
    }

    // Viterbi em log-espaço: sequência de estados mais provável
    vector<int> viterbi(const vector<int>& o) const {
        int T = o.size();
        Mat d(T, Vec(N)); vector<vector<int>> bp(T, vector<int>(N, 0));
        for (int i = 0; i < N; ++i) d[0][i] = log(pi[i]) + log(B[i][o[0]]);
        for (int t = 1; t < T; ++t)
            for (int j = 0; j < N; ++j) {
                double best = -1e300; int bi = 0;
                for (int i = 0; i < N; ++i) { double v = d[t - 1][i] + log(A[i][j]); if (v > best) { best = v; bi = i; } }
                d[t][j] = best + log(B[j][o[t]]); bp[t][j] = bi;
            }
        vector<int> path(T);
        path[T - 1] = max_element(d[T - 1].begin(), d[T - 1].end()) - d[T - 1].begin();
        for (int t = T - 1; t > 0; --t) path[t - 1] = bp[t][path[t]];
        return path;
    }

    // Baum–Welch numa única sequência
    void baum_welch(const vector<int>& o, int iters) {
        int T = o.size();
        for (int it = 0; it < iters; ++it) {
            Mat alpha, beta; Vec c;
            forward(o, alpha, c); backward(o, c, beta);
            Mat gamma(T, Vec(N)), xi_sum(N, Vec(N, 0));
            for (int t = 0; t < T; ++t) {
                double s = 0;
                for (int i = 0; i < N; ++i) { gamma[t][i] = alpha[t][i] * beta[t][i]; s += gamma[t][i]; }
                for (int i = 0; i < N; ++i) gamma[t][i] /= s;
            }
            for (int t = 0; t + 1 < T; ++t) {
                double s = 0; Mat xi(N, Vec(N));
                for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) { xi[i][j] = alpha[t][i] * A[i][j] * B[j][o[t + 1]] * beta[t + 1][j]; s += xi[i][j]; }
                for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) xi_sum[i][j] += xi[i][j] / s;
            }
            for (int i = 0; i < N; ++i) {
                pi[i] = gamma[0][i];
                double gs = 0; for (int t = 0; t + 1 < T; ++t) gs += gamma[t][i];
                for (int j = 0; j < N; ++j) A[i][j] = xi_sum[i][j] / gs;
                double gall = gs + gamma[T - 1][i];
                for (int m = 0; m < M; ++m) { double s = 0; for (int t = 0; t < T; ++t) if (o[t] == m) s += gamma[t][i]; B[i][m] = s / gall; }
            }
        }
    }
};

int main() {
    // Estados: 0=Ensolarado, 1=Chuvoso. Observações: 0=caminhar, 1=comprar, 2=limpar
    HMM real{2, 3, {0.6, 0.4}, {{0.7, 0.3}, {0.4, 0.6}}, {{0.6, 0.3, 0.1}, {0.1, 0.4, 0.5}}};
    mt19937 rng(0);
    int T = 2000;
    vector<int> obs(T), estados(T);
    int s = discrete_distribution<int>(real.pi.begin(), real.pi.end())(rng);
    for (int t = 0; t < T; ++t) {
        estados[t] = s;
        obs[t] = discrete_distribution<int>(real.B[s].begin(), real.B[s].end())(rng);
        s = discrete_distribution<int>(real.A[s].begin(), real.A[s].end())(rng);
    }
    // 1) Viterbi com o modelo verdadeiro: quão bem recuperamos os estados ocultos?
    auto path = real.viterbi(obs);
    int ok = 0;
    for (int t = 0; t < T; ++t) ok += path[t] == estados[t];
    printf("Viterbi (modelo verdadeiro): %.1f%% dos estados ocultos recuperados\n", 100.0 * ok / T);

    // 2) Baum–Welch: aprende parâmetros só com as observações, partindo de um chute
    HMM m{2, 3, {0.5, 0.5}, {{0.6, 0.4}, {0.5, 0.5}}, {{0.4, 0.3, 0.3}, {0.3, 0.3, 0.4}}};
    Mat al; Vec c;
    printf("log-verossimilhanca inicial: %.1f\n", m.forward(obs, al, c));
    m.baum_welch(obs, 200);
    printf("log-verossimilhanca apos EM: %.1f  (modelo verdadeiro: %.1f)\n", m.forward(obs, al, c), real.forward(obs, al, c));
    printf("A aprendida = [[%.2f %.2f],[%.2f %.2f]]  (verdadeira [[.7 .3],[.4 .6]]; estados podem vir permutados)\n", m.A[0][0], m.A[0][1], m.A[1][0], m.A[1][1]);
    printf("B aprendida = [[%.2f %.2f %.2f],[%.2f %.2f %.2f]]\n", m.B[0][0], m.B[0][1], m.B[0][2], m.B[1][0], m.B[1][1], m.B[1][2]);
}
