// Estratégias de decodificação de LLMs do zero: greedy, temperatura, top-k, top-p (nucleus) e beam search.
// Usa um "modelo" de linguagem de brinquedo (distribuição do próximo token dada por uma tabela de logits).
// Build: g++ -std=c++17 -O2 sampling.cpp -o sampling
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;

vector<double> softmax(Vec z, double T = 1.0) {
    double mx = *max_element(z.begin(), z.end()), s = 0;
    for (double& v : z) { v = exp((v - mx) / T); s += v; }
    for (double& v : z) v /= s;
    return z;
}

// Mantém os k tokens mais prováveis (renormaliza)
Vec top_k(Vec p, int k) {
    vector<int> idx(p.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
    Vec q(p.size(), 0.0); double s = 0;
    for (int i = 0; i < k; ++i) { q[idx[i]] = p[idx[i]]; s += p[idx[i]]; }
    for (double& v : q) v /= s;
    return q;
}

// Nucleus: menor conjunto de tokens cuja massa acumulada >= p_lim
Vec top_p(Vec p, double p_lim) {
    vector<int> idx(p.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
    Vec q(p.size(), 0.0); double cum = 0, s = 0;
    for (int i : idx) { q[i] = p[i]; s += p[i]; cum += p[i]; if (cum >= p_lim) break; }
    for (double& v : q) v /= s;
    return q;
}

double entropia(const Vec& p) { double h = 0; for (double x : p) if (x > 0) h -= x * log2(x); return h; }

int main() {
    vector<string> vocab = {"o", "gato", "cachorro", "carro", "dorme", "corre", "voa", "sofa", "rua", "ceu"};
    // logits do próximo token dado o contexto "o gato ..." (brinquedo): verbos plausíveis dominam, cauda longa de tokens ruins
    Vec logits = {0.5, 0.2, 0.1, -1.0, 4.0, 3.2, 1.5, 0.3, -0.5, 0.0};
    mt19937 rng(0);

    printf("token       | p(T=1)  | p(T=0.5) | p(T=2)\n");
    Vec p1 = softmax(logits), p05 = softmax(logits, 0.5), p2 = softmax(logits, 2.0);
    for (size_t i = 0; i < vocab.size(); ++i) printf("%-11s |  %.3f  |  %.3f   |  %.3f\n", vocab[i].c_str(), p1[i], p05[i], p2[i]);
    printf("entropia (bits): T=0.5 -> %.2f | T=1 -> %.2f | T=2 -> %.2f   (temperatura alta = mais aleatorio)\n\n", entropia(p05), entropia(p1), entropia(p2));

    auto amostra = [&](const Vec& p, int n) {
        map<string, int> c; discrete_distribution<int> d(p.begin(), p.end());
        for (int i = 0; i < n; ++i) ++c[vocab[d(rng)]];
        string s;
        for (auto& [w, k] : c) s += w + ":" + to_string(k) + " ";
        return s;
    };
    printf("1000 amostras por estrategia:\n");
    printf("  greedy (argmax)   : %s\n", vocab[max_element(logits.begin(), logits.end()) - logits.begin()].c_str());
    printf("  T=1 puro          : %s\n", amostra(p1, 1000).c_str());
    printf("  T=0.5             : %s\n", amostra(p05, 1000).c_str());
    printf("  top-k (k=3)       : %s\n", amostra(top_k(p1, 3), 1000).c_str());
    printf("  top-p (p=0.9)     : %s\n", amostra(top_p(p1, 0.9), 1000).c_str());
    int nucleo = 0; { Vec q = top_p(p1, 0.9); for (double x : q) nucleo += x > 0; }
    printf("  (nucleo de top-p=0.9 tem %d tokens: o tamanho se ADAPTA a incerteza; top-k tem tamanho fixo)\n\n", nucleo);

    // Beam search: maximiza a probabilidade da SEQUÊNCIA (não passo a passo). Modelo de brinquedo: P(w_t | w_{t-1}).
    // Cria um caso em que o guloso erra: 'A' parece melhor no 1o passo mas leva a um beco sem boas continuacoes.
    vector<string> tk = {"A", "B", "fim1", "fim2"};
    map<string, vector<double>> P = {{"<s>", {0.55, 0.45, 0, 0}}, {"A", {0, 0, 0.5, 0.5}}, {"B", {0, 0, 0.95, 0.05}}};
    auto idx = [&](const string& s) { return find(tk.begin(), tk.end(), s) - tk.begin(); };
    // guloso
    string g1 = tk[max_element(P["<s>"].begin(), P["<s>"].end()) - P["<s>"].begin()];
    double pg = P["<s>"][idx(g1)] * *max_element(P[g1].begin(), P[g1].end());
    // beam com largura 2 (enumera as duas melhores hipóteses após o 1o passo)
    double melhor = 0; string bs;
    for (string a : {"A", "B"}) for (string b : {"fim1", "fim2"}) { double p = P["<s>"][idx(a)] * P[a][idx(b)]; if (p > melhor) { melhor = p; bs = a + " " + b; } }
    printf("Beam search: guloso escolhe '%s' -> prob. da sequencia = %.3f | beam (largura 2) acha '%s' com prob. = %.3f\n", g1.c_str(), pg, bs.c_str(), melhor);
    printf("(beam favorece sequencias provaveis mas pode gerar texto repetitivo/generico; LLMs de chat usam amostragem com T/top-p)\n");
}
