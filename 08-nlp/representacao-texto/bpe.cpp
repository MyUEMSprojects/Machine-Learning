// Byte-Pair Encoding (BPE) do zero: aprende "merges" do corpus e tokeniza palavras novas em subpalavras.
// É o princípio dos tokenizadores de GPT/Llama (que operam sobre bytes).
// Build: g++ -std=c++17 -O2 bpe.cpp -o bpe
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
using namespace std;

using Word = vector<string>;  // símbolos da palavra

int main() {
    string corpus = "low low low low low lower lower newest newest newest newest newest newest "
                    "widest widest widest lowest lowest";
    // frequência das palavras; cada palavra vira uma sequência de caracteres + marcador de fim "</w>"
    map<string, int> freq;
    { istringstream is(corpus); string w; while (is >> w) ++freq[w]; }
    map<Word, int> vocab;
    for (auto& [w, f] : freq) { Word s; for (char c : w) s.push_back(string(1, c)); s.push_back("</w>"); vocab[s] = f; }

    vector<pair<string, string>> merges;
    int num_merges = 10;
    printf("Treinando BPE (%d merges):\n", num_merges);
    for (int m = 0; m < num_merges; ++m) {
        map<pair<string, string>, int> pares;  // conta pares adjacentes ponderados pela frequência da palavra
        for (auto& [s, f] : vocab) for (size_t i = 0; i + 1 < s.size(); ++i) pares[{s[i], s[i + 1]}] += f;
        if (pares.empty()) break;
        auto melhor = pares.begin();
        for (auto it = pares.begin(); it != pares.end(); ++it) if (it->second > melhor->second) melhor = it;
        auto [a, b] = melhor->first;
        merges.push_back({a, b});
        printf("  merge %2d: (%s, %s) -> %s%s   [freq %d]\n", m + 1, a.c_str(), b.c_str(), a.c_str(), b.c_str(), melhor->second);
        map<Word, int> novo;
        for (auto& [s, f] : vocab) {  // aplica o merge em todas as palavras
            Word t;
            for (size_t i = 0; i < s.size(); ++i) {
                if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) { t.push_back(a + b); ++i; }
                else t.push_back(s[i]);
            }
            novo[t] += f;
        }
        vocab = novo;
    }

    // Codificação: aplica os merges NA ORDEM em que foram aprendidos
    auto encode = [&](const string& w) {
        Word s; for (char c : w) s.push_back(string(1, c)); s.push_back("</w>");
        for (auto& [a, b] : merges) {
            Word t;
            for (size_t i = 0; i < s.size(); ++i) {
                if (i + 1 < s.size() && s[i] == a && s[i + 1] == b) { t.push_back(a + b); ++i; }
                else t.push_back(s[i]);
            }
            s = t;
        }
        return s;
    };
    printf("\nTokenizacao com os merges aprendidos:\n");
    for (string w : {"lowest", "newer", "wider", "slowest", "xyz"}) {
        printf("  %-8s ->", w.c_str());
        for (auto& t : encode(w)) printf(" [%s]", t.c_str());
        printf("\n");
    }
    printf("\nPalavras nao vistas ('slowest') se decompoem em subpalavras conhecidas; caracteres raros viram simbolos individuais.\n"
           "Nao ha token <UNK>: no pior caso, cai para caracteres/bytes.\n");
}
