// Modelo de linguagem n-gram (bigrama/trigrama) do zero: contagens, suavização (add-k e interpolação),
// perplexidade em dados de teste e geração de texto. Corpus sintético com dependências verbo->objeto.
// Build: g++ -std=c++17 -O2 ngram.cpp -o ngram
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
using namespace std;

using Sent = vector<string>;

int main() {
    mt19937 rng(0);
    // gramática: sujeito -> verbo -> objeto (o objeto depende do VERBO; o verbo, do SUJEITO)
    map<string, vector<string>> verbo_de = {{"gato", {"caca", "dorme"}}, {"cachorro", {"morde", "caca"}}, {"menino", {"chuta", "le"}}, {"menina", {"le", "pinta"}}};
    map<string, vector<string>> obj_de = {{"caca", {"rato", "passaro"}}, {"dorme", {"sofa", "cama"}}, {"morde", {"osso", "bola"}},
                                          {"chuta", {"bola", "pedra"}}, {"le", {"livro", "jornal"}}, {"pinta", {"quadro", "parede"}}};
    vector<string> sujeitos;
    for (auto& [s, _] : verbo_de) sujeitos.push_back(s);
    auto pick = [&](const vector<string>& v) { return v[uniform_int_distribution<int>(0, v.size() - 1)(rng)]; };
    auto gera = [&]() {
        string s = pick(sujeitos), v = pick(verbo_de[s]), o = pick(obj_de[v]);
        return Sent{"<s>", "<s>", "o", s, v, "o", o, "</s>"};  // dois <s> p/ contexto de trigrama
    };
    vector<Sent> treino, teste;
    for (int i = 0; i < 40; ++i) treino.push_back(gera());   // treino pequeno => muitos n-gramas do teste nunca foram vistos
    for (int i = 0; i < 500; ++i) teste.push_back(gera());

    map<string, map<string, int>> bi;                 // contagem bigrama: prev -> w
    map<pair<string, string>, map<string, int>> tri;  // contagem trigrama: (prev2, prev1) -> w
    map<string, int> uni; long total = 0;
    set<string> vocab;
    for (auto& s : treino)
        for (size_t i = 2; i < s.size(); ++i) {
            ++uni[s[i]]; ++total; vocab.insert(s[i]);
            ++bi[s[i - 1]][s[i]]; ++tri[{s[i - 2], s[i - 1]}][s[i]];
        }
    double V = vocab.size();
    auto count_of = [](auto& m) { int c = 0; for (auto& [_, v] : m) c += v; return c; };

    auto p_uni = [&](const string& w) { return (uni.count(w) ? uni[w] : 0) / (double)total; };
    auto p_bi = [&](const string& a, const string& w, double k) {           // add-k
        int c = bi.count(a) ? (bi[a].count(w) ? bi[a][w] : 0) : 0, n = bi.count(a) ? count_of(bi[a]) : 0;
        return (c + k) / (n + k * V);
    };
    auto p_tri = [&](const string& a, const string& b, const string& w, double k) {
        auto key = make_pair(a, b);
        int c = tri.count(key) ? (tri[key].count(w) ? tri[key][w] : 0) : 0, n = tri.count(key) ? count_of(tri[key]) : 0;
        return (c + k) / (n + k * V);
    };
    // interpolação linear: P = l3 P_tri + l2 P_bi + l1 P_uni (combina o melhor de cada ordem)
    auto p_interp = [&](const string& a, const string& b, const string& w) {
        double l3 = 0.6, l2 = 0.3, l1 = 0.1;
        return l3 * p_tri(a, b, w, 0) + l2 * p_bi(b, w, 0) + l1 * p_uni(w);
    };

    auto perplexidade = [&](auto&& prob) {  // PP = exp(-1/N sum log P(w_i | contexto))
        double ll = 0; long n = 0;
        for (auto& s : teste) for (size_t i = 2; i < s.size(); ++i) { ll += log(max(prob(s[i - 2], s[i - 1], s[i]), 1e-12)); ++n; }
        return exp(-ll / n);
    };
    printf("vocabulario=%d palavras. Perplexidade no teste (menor = melhor; ~ 'fator de ramificacao efetivo'):\n", (int)V);
    printf("  unigrama                   : %.2f\n", perplexidade([&](auto&, auto&, auto& w) { return p_uni(w); }));
    printf("  bigrama  (add-0.01)        : %.2f\n", perplexidade([&](auto&, auto& b, auto& w) { return p_bi(b, w, 0.01); }));
    printf("  trigrama (add-0.01)        : %.2f\n", perplexidade([&](auto& a, auto& b, auto& w) { return p_tri(a, b, w, 0.01); }));
    printf("  trigrama MLE (sem suaviz.) : %.2f  <- n-gramas nao vistos recebem P=0 (truncada em 1e-12): arriscado com dados esparsos\n",
           perplexidade([&](auto& a, auto& b, auto& w) { return p_tri(a, b, w, 0); }));
    printf("  interpolacao 0.6/0.3/0.1   : %.2f\n", perplexidade([&](auto& a, auto& b, auto& w) { return p_interp(a, b, w); }));

    // geração amostrando do modelo trigrama
    printf("\nTexto gerado pelo trigrama:\n");
    for (int k = 0; k < 5; ++k) {
        string a = "<s>", b = "<s>", out;
        for (int t = 0; t < 12; ++t) {
            auto it = tri.find({a, b});
            if (it == tri.end()) break;
            vector<string> ws; vector<double> ps;
            for (auto& [w, c] : it->second) { ws.push_back(w); ps.push_back(c); }
            string w = ws[discrete_distribution<int>(ps.begin(), ps.end())(rng)];
            if (w == "</s>") break;
            out += w + " "; a = b; b = w;
        }
        printf("  %s\n", out.c_str());
    }
}
