// Regras de associação com Apriori: itemsets frequentes -> regras com suporte, confiança e lift.
// Build: g++ -std=c++17 -O2 apriori.cpp -o apriori
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
using namespace std;

using Itemset = set<string>;

string str(const Itemset& s) {
    string r = "{";
    for (auto& x : s) r += (r.size() > 1 ? "," : "") + x;
    return r + "}";
}

int main() {
    vector<Itemset> T = {
        {"pao", "leite"}, {"pao", "fralda", "cerveja", "ovos"}, {"leite", "fralda", "cerveja", "coca"},
        {"pao", "leite", "fralda", "cerveja"}, {"pao", "leite", "fralda", "coca"}, {"pao", "leite", "manteiga"},
        {"pao", "manteiga"}, {"leite", "fralda", "cerveja"}, {"pao", "leite", "fralda", "cerveja", "manteiga"},
        {"pao", "ovos"}};
    double min_sup = 0.3, min_conf = 0.7;
    int n = T.size();

    auto support = [&](const Itemset& s) {
        int c = 0;
        for (auto& t : T) c += includes(t.begin(), t.end(), s.begin(), s.end());
        return (double)c / n;
    };

    // Apriori: se um itemset é frequente, TODOS os seus subconjuntos também são (propriedade anti-monotônica).
    // Logo só combinamos itemsets frequentes de tamanho k para gerar candidatos de tamanho k+1.
    map<Itemset, double> freq;
    set<Itemset> nivel;
    for (auto& t : T) for (auto& i : t) if (support({i}) >= min_sup) nivel.insert({i});
    while (!nivel.empty()) {
        for (auto& s : nivel) freq[s] = support(s);
        set<Itemset> cand;
        for (auto& a : nivel) for (auto& b : nivel) {
            Itemset u = a; u.insert(b.begin(), b.end());
            if (u.size() != a.size() + 1) continue;
            bool ok = true;  // poda: todo subconjunto de tamanho k deve ser frequente
            for (auto& x : u) { Itemset sub = u; sub.erase(x); if (!nivel.count(sub)) { ok = false; break; } }
            if (ok && support(u) >= min_sup) cand.insert(u);
        }
        nivel = cand;
    }

    cout << "Itemsets frequentes (suporte >= " << min_sup << "):\n";
    for (auto& [s, sup] : freq) if (s.size() > 1) printf("  %-28s suporte=%.2f\n", str(s).c_str(), sup);

    cout << "\nRegras A -> B (confianca >= " << min_conf << "):\n";
    for (auto& [s, sup] : freq) {
        if (s.size() < 2) continue;
        vector<string> items(s.begin(), s.end());
        int m = items.size();
        for (int mask = 1; mask < (1 << m) - 1; ++mask) {  // todas as partições antecedente/consequente
            Itemset A, B;
            for (int i = 0; i < m; ++i) (mask >> i & 1 ? A : B).insert(items[i]);
            double conf = sup / support(A), lift = conf / support(B);
            if (conf >= min_conf) printf("  %-18s -> %-14s suporte=%.2f confianca=%.2f lift=%.2f\n", str(A).c_str(), str(B).c_str(), sup, conf, lift);
        }
    }
    cout << "\nlift > 1: A e B aparecem juntos mais do que o acaso; lift = 1: independentes.\n";
}
