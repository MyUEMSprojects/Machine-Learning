// Bagging + Random Forest do zero: árvores CART em amostras bootstrap com sorteio de features por nó,
// votação majoritária e estimativa OOB (out-of-bag).
// Build: g++ -std=c++17 -O2 random_forest.cpp -o random_forest
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

struct Node { int feat = -1; double thr = 0; int label = 0, left = -1, right = -1; };

struct Tree {
    vector<Node> nodes;
    int max_depth, K, max_features;
    const Mat* X; const vector<int>* y;
    mt19937* rng;

    static double gini(const vector<int>& c, int n) {
        double g = 1;
        for (int v : c) g -= pow((double)v / n, 2);
        return g;
    }
    void fit(const Mat& X_, const vector<int>& y_, vector<int> idx, int K_, int md, int mf, mt19937& r) {
        X = &X_; y = &y_; K = K_; max_depth = md; max_features = mf; rng = &r;
        nodes.clear();
        build(idx, 0);
    }
    int build(vector<int>& idx, int depth) {
        int id = nodes.size(); nodes.push_back({});
        vector<int> cnt(K, 0);
        for (int i : idx) ++cnt[(*y)[i]];
        nodes[id].label = max_element(cnt.begin(), cnt.end()) - cnt.begin();
        int n = idx.size();
        double parent = gini(cnt, n);
        if (parent == 0 || depth >= max_depth || n < 2) return id;

        int d = (*X)[0].size();
        vector<int> feats(d);
        iota(feats.begin(), feats.end(), 0);
        shuffle(feats.begin(), feats.end(), *rng);
        feats.resize(min(max_features, d));  // <- diferença chave da Random Forest

        double best = parent - 1e-12; int bf = -1; double bt = 0;
        for (int f : feats) {
            sort(idx.begin(), idx.end(), [&](int a, int b) { return (*X)[a][f] < (*X)[b][f]; });
            vector<int> L(K, 0), R = cnt;
            for (int s = 0; s + 1 < n; ++s) {
                int c = (*y)[idx[s]]; ++L[c]; --R[c];
                double a = (*X)[idx[s]][f], b = (*X)[idx[s + 1]][f];
                if (a == b) continue;
                double g = ((s + 1) * gini(L, s + 1) + (n - s - 1) * gini(R, n - s - 1)) / n;
                if (g < best) { best = g; bf = f; bt = (a + b) / 2; }
            }
        }
        if (bf < 0) return id;
        vector<int> li, ri;
        for (int i : idx) ((*X)[i][bf] <= bt ? li : ri).push_back(i);
        nodes[id].feat = bf; nodes[id].thr = bt;
        int l = build(li, depth + 1), r = build(ri, depth + 1);
        nodes[id].left = l; nodes[id].right = r;
        return id;
    }
    int predict(const Vec& x) const {
        int n = 0;
        while (nodes[n].feat >= 0) n = x[nodes[n].feat] <= nodes[n].thr ? nodes[n].left : nodes[n].right;
        return nodes[n].label;
    }
};

struct RandomForest {
    vector<Tree> trees;
    vector<vector<char>> in_bag;  // in_bag[t][i] = amostra i foi sorteada para a árvore t
    int K;
    void fit(const Mat& X, const vector<int>& y, int K_, int n_trees, int max_features, mt19937& rng) {
        K = K_;
        int n = X.size();
        trees.assign(n_trees, {});
        in_bag.assign(n_trees, vector<char>(n, 0));
        uniform_int_distribution<int> pick(0, n - 1);
        for (int t = 0; t < n_trees; ++t) {
            vector<int> idx(n);
            for (int& i : idx) { i = pick(rng); in_bag[t][i] = 1; }  // bootstrap: com reposição
            trees[t].fit(X, y, idx, K, 1000, max_features, rng);
        }
    }
    int predict(const Vec& x) const {
        vector<int> votos(K, 0);
        for (auto& t : trees) ++votos[t.predict(x)];
        return max_element(votos.begin(), votos.end()) - votos.begin();
    }
    // Acurácia OOB: cada amostra é votada só pelas árvores que NÃO a viram (validação "de graça").
    double oob_score(const Mat& X, const vector<int>& y) const {
        int ok = 0, tot = 0;
        for (size_t i = 0; i < X.size(); ++i) {
            vector<int> votos(K, 0); int v = 0;
            for (size_t t = 0; t < trees.size(); ++t) if (!in_bag[t][i]) { ++votos[trees[t].predict(X[i])]; ++v; }
            if (!v) continue;
            ++tot;
            ok += (max_element(votos.begin(), votos.end()) - votos.begin()) == y[i];
        }
        return (double)ok / tot;
    }
};

int main() {
    mt19937 rng(5);
    normal_distribution<double> N(0, 1);
    // 2 features informativas + 8 de ruído, com sobreposição de classes
    auto gera = [&](int n, Mat& X, vector<int>& y) {
        for (int i = 0; i < n; ++i) {
            int c = i % 2; Vec x(10);
            x[0] = (c ? 1.0 : -1.0) + 1.3 * N(rng);
            x[1] = (c ? 1.0 : -1.0) + 1.3 * N(rng);
            for (int j = 2; j < 10; ++j) x[j] = N(rng);
            X.push_back(x); y.push_back(c);
        }
    };
    Mat Xtr, Xte; vector<int> ytr, yte;
    gera(300, Xtr, ytr); gera(500, Xte, yte);
    auto acc = [&](auto&& f) { int ok = 0; for (size_t i = 0; i < Xte.size(); ++i) ok += f(Xte[i]) == yte[i]; return (double)ok / Xte.size(); };

    Tree single; vector<int> all(Xtr.size()); iota(all.begin(), all.end(), 0);
    single.fit(Xtr, ytr, all, 2, 1000, 10, rng);
    printf("Uma arvore completa    : acc teste=%.3f\n", acc([&](const Vec& x) { return single.predict(x); }));

    for (int nt : {1, 10, 50, 200}) {
        RandomForest rf; rf.fit(Xtr, ytr, 2, nt, 3, rng);
        printf("Random Forest (%3d arv): acc teste=%.3f  OOB=%.3f\n", nt, acc([&](const Vec& x) { return rf.predict(x); }), rf.oob_score(Xtr, ytr));
    }
}
