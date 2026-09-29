// Árvore de decisão CART (classificação, impureza de Gini) do zero, com impressão da árvore.
// Build: g++ -std=c++17 -O2 arvore_decisao.cpp -o arvore_decisao
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

struct Node {
    int feat = -1;      // -1 => folha
    double thr = 0;
    int label = 0;
    int left = -1, right = -1;
};

struct DecisionTree {
    vector<Node> nodes;
    int max_depth, min_samples, K;
    const Mat* X; const vector<int>* y;

    static double gini(const vector<int>& cnt, int n) {
        double g = 1;
        for (int c : cnt) g -= pow((double)c / n, 2);
        return g;
    }

    void fit(const Mat& X_, const vector<int>& y_, int K_, int max_depth_ = 1000, int min_samples_ = 2) {
        X = &X_; y = &y_; K = K_; max_depth = max_depth_; min_samples = min_samples_;
        nodes.clear();
        vector<int> idx(X_.size());
        iota(idx.begin(), idx.end(), 0);
        build(idx, 0);
    }

    int build(vector<int>& idx, int depth) {
        int id = nodes.size();
        nodes.push_back({});
        vector<int> cnt(K, 0);
        for (int i : idx) ++cnt[(*y)[i]];
        nodes[id].label = max_element(cnt.begin(), cnt.end()) - cnt.begin();
        int n = idx.size();
        double parent = gini(cnt, n);
        if (parent == 0 || depth >= max_depth || n < min_samples) return id;

        double best = parent - 1e-12; int bf = -1; double bt = 0;
        int d = (*X)[0].size();
        for (int f = 0; f < d; ++f) {
            sort(idx.begin(), idx.end(), [&](int a, int b) { return (*X)[a][f] < (*X)[b][f]; });
            vector<int> L(K, 0), R = cnt;
            for (int s = 0; s + 1 < n; ++s) {
                int c = (*y)[idx[s]];
                ++L[c]; --R[c];
                double a = (*X)[idx[s]][f], b = (*X)[idx[s + 1]][f];
                if (a == b) continue;  // só corta entre valores distintos
                int nl = s + 1, nr = n - nl;
                double g = (nl * gini(L, nl) + nr * gini(R, nr)) / n;  // Gini ponderado dos filhos
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

    void print(int n = 0, int ind = 0) const {
        string pad(ind * 2, ' ');
        if (nodes[n].feat < 0) { cout << pad << "-> classe " << nodes[n].label << "\n"; return; }
        cout << pad << "se x[" << nodes[n].feat << "] <= " << nodes[n].thr << ":\n";
        print(nodes[n].left, ind + 1);
        cout << pad << "senao:\n";
        print(nodes[n].right, ind + 1);
    }
};

int main() {
    mt19937 rng(2);
    uniform_real_distribution<double> U(-1, 1);
    auto gera = [&](int n, Mat& X, vector<int>& y) {  // região retangular com 10% de rótulos ruidosos
        for (int i = 0; i < n; ++i) {
            double a = U(rng), b = U(rng);
            int lab = (a > 0.1) && (b > -0.2);
            if (U(rng) > 0.9) lab = 1 - lab;
            X.push_back({a, b}); y.push_back(lab);
        }
    };
    Mat Xtr, Xte; vector<int> ytr, yte;
    gera(300, Xtr, ytr); gera(300, Xte, yte);

    cout << "profundidade | acc treino | acc teste\n";
    for (int md : {1, 2, 3, 5, 8, 1000}) {
        DecisionTree t; t.fit(Xtr, ytr, 2, md);
        int a = 0, b = 0;
        for (size_t i = 0; i < Xtr.size(); ++i) a += t.predict(Xtr[i]) == ytr[i];
        for (size_t i = 0; i < Xte.size(); ++i) b += t.predict(Xte[i]) == yte[i];
        printf("   %4d      |   %.3f    |  %.3f\n", md, (double)a / Xtr.size(), (double)b / Xte.size());
    }
    cout << "\nArvore de profundidade 2 (recupera a regiao x0>0.1 e x1>-0.2):\n";
    DecisionTree t; t.fit(Xtr, ytr, 2, 2);
    t.print();
}
