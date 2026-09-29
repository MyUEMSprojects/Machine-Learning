// Detecção de anomalias: (1) distância de Mahalanobis (estatística) e (2) Isolation Forest do zero.
// Build: g++ -std=c++17 -O2 deteccao_anomalias.cpp -o deteccao_anomalias
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

// ---------- Mahalanobis (2D): d^2 = (x-mu)^T Sigma^-1 (x-mu) ----------
struct Mahalanobis {
    double m0 = 0, m1 = 0, i00, i01, i11;
    void fit(const Mat& X) {
        int n = X.size();
        for (auto& r : X) { m0 += r[0] / n; m1 += r[1] / n; }
        double c00 = 0, c01 = 0, c11 = 0;
        for (auto& r : X) { c00 += pow(r[0] - m0, 2); c01 += (r[0] - m0) * (r[1] - m1); c11 += pow(r[1] - m1, 2); }
        c00 /= n - 1; c01 /= n - 1; c11 /= n - 1;
        double det = c00 * c11 - c01 * c01;
        i00 = c11 / det; i01 = -c01 / det; i11 = c00 / det;  // inversa 2x2
    }
    double score(const Vec& x) const {
        double a = x[0] - m0, b = x[1] - m1;
        return a * a * i00 + 2 * a * b * i01 + b * b * i11;
    }
};

// ---------- Isolation Forest ----------
// Ideia: anomalias são "poucas e diferentes" => isoladas com poucos cortes aleatórios (caminho curto).
struct ITree {
    struct N { int feat = -1; double thr = 0; int left = -1, right = -1, size = 0; };
    vector<N> nodes;
    int build(const Mat& X, vector<int> idx, int depth, int limit, mt19937& rng) {
        int id = nodes.size(); nodes.push_back({});
        nodes[id].size = idx.size();
        if (depth >= limit || idx.size() <= 1) return id;
        int f = uniform_int_distribution<int>(0, X[0].size() - 1)(rng);
        double lo = 1e18, hi = -1e18;
        for (int i : idx) { lo = min(lo, X[i][f]); hi = max(hi, X[i][f]); }
        if (lo == hi) return id;
        double thr = uniform_real_distribution<double>(lo, hi)(rng);  // corte totalmente aleatório
        vector<int> l, r;
        for (int i : idx) (X[i][f] < thr ? l : r).push_back(i);
        nodes[id].feat = f; nodes[id].thr = thr;
        int a = build(X, l, depth + 1, limit, rng), b = build(X, r, depth + 1, limit, rng);
        nodes[id].left = a; nodes[id].right = b;
        return id;
    }
    static double c(double n) {  // comprimento médio de caminho de busca numa BST com n nós
        if (n <= 1) return 0;
        return 2 * (log(n - 1) + 0.5772156649) - 2 * (n - 1) / n;
    }
    double path(const Vec& x) const {
        int n = 0; double depth = 0;
        while (nodes[n].feat >= 0) { n = x[nodes[n].feat] < nodes[n].thr ? nodes[n].left : nodes[n].right; ++depth; }
        return depth + c(nodes[n].size);
    }
};

struct IsolationForest {
    vector<ITree> trees; int psi;
    void fit(const Mat& X, int n_trees, int psi_, mt19937& rng) {
        psi = min<int>(psi_, X.size());
        int limit = ceil(log2(psi));
        vector<int> all(X.size());
        iota(all.begin(), all.end(), 0);
        for (int t = 0; t < n_trees; ++t) {
            shuffle(all.begin(), all.end(), rng);
            ITree tr;
            tr.build(X, vector<int>(all.begin(), all.begin() + psi), 0, limit, rng);
            trees.push_back(tr);
        }
    }
    // score em (0,1): próximo de 1 => anomalia; ~0.5 => normal
    double score(const Vec& x) const {
        double e = 0;
        for (auto& t : trees) e += t.path(x);
        e /= trees.size();
        return pow(2.0, -e / ITree::c(psi));
    }
};

int main() {
    mt19937 rng(0);
    normal_distribution<double> N(0, 1);
    Mat X;
    for (int i = 0; i < 500; ++i) { double a = N(rng); X.push_back({a, 0.8 * a + 0.4 * N(rng)}); }  // correlacionados
    // pontos de teste: normal, anomalia "óbvia", e anomalia que só Mahalanobis nota (contra a correlação)
    vector<pair<string, Vec>> teste = {{"normal (0.5, 0.4)", {0.5, 0.4}}, {"anomalia longe (5, 5)", {5, 5}},
                                        {"contra correlacao (1.5, -1.5)", {1.5, -1.5}}};
    Mahalanobis mh; mh.fit(X);
    IsolationForest iforest; iforest.fit(X, 100, 256, rng);
    printf("%-32s | Mahalanobis d^2 | IsolationForest score\n", "ponto");
    for (auto& [nome, p] : teste) printf("%-32s |     %7.2f     |       %.3f\n", nome.c_str(), mh.score(p), iforest.score(p));
    cout << "(chi2 com 2 g.l.: d^2 > 9.2 => anomalia a 99%; IsolationForest > ~0.6 costuma indicar anomalia)\n";
}
