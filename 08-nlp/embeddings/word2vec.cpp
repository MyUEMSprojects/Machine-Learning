// word2vec (Skip-gram com Negative Sampling) do zero, num corpus sintético com 3 categorias semânticas.
// Palavras que aparecem em contextos parecidos (gato/cachorro, pao/queijo...) ganham vetores próximos.
// Build: g++ -std=c++17 -O2 word2vec.cpp -o word2vec
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
using namespace std;

using Vec = vector<double>;

int main() {
    mt19937 rng(0);
    vector<string> animais = {"gato", "cachorro", "cavalo", "vaca", "ovelha"};
    vector<string> comidas = {"pao", "queijo", "maca", "arroz", "bolo"};
    vector<string> veiculos = {"carro", "onibus", "trem", "aviao", "barco"};
    auto pick = [&](const vector<string>& v) { return v[uniform_int_distribution<int>(0, v.size() - 1)(rng)]; };

    // corpus: cada categoria aparece com verbos/lugares próprios
    vector<vector<string>> sents;
    for (int i = 0; i < 6000; ++i) {
        sents.push_back({"o", pick(animais), pick({"corre", "dorme", "late", "pasta"}), "no", pick({"campo", "quintal", "pasto"})});
        sents.push_back({"eu", "como", pick(comidas), "no", pick({"cafe", "almoco", "jantar"})});
        sents.push_back({"o", pick(veiculos), pick({"anda", "para", "passa"}), "na", pick({"rua", "estrada", "pista"})});
    }
    map<string, int> id; vector<string> words;
    for (auto& s : sents) for (auto& w : s) if (!id.count(w)) { id[w] = words.size(); words.push_back(w); }
    int V = words.size(), D = 16, WIN = 2, NEG = 5;
    Vec freq(V, 0);
    for (auto& s : sents) for (auto& w : s) freq[id[w]] += 1;
    Vec noise(V);
    for (int i = 0; i < V; ++i) noise[i] = pow(freq[i], 0.75);  // distribuição de ruído ~ f^0.75
    discrete_distribution<int> noise_dist(noise.begin(), noise.end());

    vector<Vec> In(V, Vec(D)), Out(V, Vec(D, 0.0));  // vetores de "palavra" e de "contexto"
    uniform_real_distribution<double> U(-0.5 / D, 0.5 / D);
    for (auto& v : In) for (double& x : v) x = U(rng);
    auto sig = [](double x) { return 1 / (1 + exp(-x)); };

    double lr = 0.05;
    for (int ep = 1; ep <= 5; ++ep) {
        double loss = 0; long n = 0;
        shuffle(sents.begin(), sents.end(), rng);
        for (auto& s : sents)
            for (int i = 0; i < (int)s.size(); ++i)
                for (int j = max(0, i - WIN); j <= min((int)s.size() - 1, i + WIN); ++j) {
                    if (j == i) continue;
                    int w = id[s[i]], c = id[s[j]];
                    Vec grad_in(D, 0.0);
                    for (int k = 0; k <= NEG; ++k) {  // k=0: par positivo; k>0: negativos amostrados
                        int t = k == 0 ? c : noise_dist(rng), label = k == 0;
                        if (k > 0 && t == c) continue;
                        double dot = 0;
                        for (int d = 0; d < D; ++d) dot += In[w][d] * Out[t][d];
                        double p = sig(dot), g = lr * (label - p);   // gradiente da log-verossimilhança logística
                        loss += -(label ? log(p + 1e-12) : log(1 - p + 1e-12)); ++n;
                        for (int d = 0; d < D; ++d) { grad_in[d] += g * Out[t][d]; Out[t][d] += g * In[w][d]; }
                    }
                    for (int d = 0; d < D; ++d) In[w][d] += grad_in[d];
                }
        printf("epoca %d  loss=%.4f\n", ep, loss / n);
    }
    auto cosine = [&](int a, int b) {
        double d = 0, na = 0, nb = 0;
        for (int k = 0; k < D; ++k) { d += In[a][k] * In[b][k]; na += In[a][k] * In[a][k]; nb += In[b][k] * In[b][k]; }
        return d / (sqrt(na * nb) + 1e-12);
    };
    printf("\nVizinhos mais proximos (cosseno):\n");
    for (string q : {"gato", "pao", "carro", "campo"}) {
        vector<pair<double, string>> r;
        for (auto& w : words) if (w != q) r.push_back({cosine(id[q], id[w]), w});
        sort(r.rbegin(), r.rend());
        printf("  %-8s -> %s(%.2f) %s(%.2f) %s(%.2f)\n", q.c_str(), r[0].second.c_str(), r[0].first, r[1].second.c_str(), r[1].first, r[2].second.c_str(), r[2].first);
    }
    // similaridade média dentro vs entre categorias
    auto media = [&](const vector<string>& A, const vector<string>& B) {
        double s = 0; int n = 0;
        for (auto& a : A) for (auto& b : B) if (a != b) { s += cosine(id[a], id[b]); ++n; }
        return s / n;
    };
    printf("\ncos medio: animais-animais=%.2f  comidas-comidas=%.2f  animais-comidas=%.2f  animais-veiculos=%.2f\n",
           media(animais, animais), media(comidas, comidas), media(animais, comidas), media(animais, veiculos));
}
