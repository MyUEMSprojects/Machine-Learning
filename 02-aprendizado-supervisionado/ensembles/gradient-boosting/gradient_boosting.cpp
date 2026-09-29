// Gradient Boosting para regressão (perda quadrática) com árvores de regressão rasas do zero.
// Cada árvore ajusta o RESÍDUO (= -gradiente da perda) do ensemble atual, com shrinkage.
// Build: g++ -std=c++17 -O2 gradient_boosting.cpp -o gradient_boosting
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

using Vec = vector<double>;
using Mat = vector<Vec>;

struct RNode { int feat = -1; double thr = 0, value = 0; int left = -1, right = -1; };

struct RegTree {
    vector<RNode> nodes;
    const Mat* X; const Vec* r; int max_depth;

    void fit(const Mat& X_, const Vec& r_, int md) {
        X = &X_; r = &r_; max_depth = md; nodes.clear();
        vector<int> idx(X_.size());
        iota(idx.begin(), idx.end(), 0);
        build(idx, 0);
    }
    int build(vector<int>& idx, int depth) {
        int id = nodes.size(); nodes.push_back({});
        int n = idx.size();
        double sum = 0;
        for (int i : idx) sum += (*r)[i];
        nodes[id].value = sum / n;
        if (depth >= max_depth || n < 4) return id;

        // melhor corte = maximiza  sumL^2/nL + sumR^2/nR  (equivale a minimizar o SSE)
        double best = sum * sum / n + 1e-12; int bf = -1; double bt = 0;
        int d = (*X)[0].size();
        for (int f = 0; f < d; ++f) {
            sort(idx.begin(), idx.end(), [&](int a, int b) { return (*X)[a][f] < (*X)[b][f]; });
            double sl = 0;
            for (int s = 0; s + 1 < n; ++s) {
                sl += (*r)[idx[s]];
                double a = (*X)[idx[s]][f], b = (*X)[idx[s + 1]][f];
                if (a == b) continue;
                int nl = s + 1, nr = n - nl;
                double sr = sum - sl, score = sl * sl / nl + sr * sr / nr;
                if (score > best) { best = score; bf = f; bt = (a + b) / 2; }
            }
        }
        if (bf < 0) return id;
        vector<int> li, ri;
        for (int i : idx) ((*X)[i][bf] <= bt ? li : ri).push_back(i);
        nodes[id].feat = bf; nodes[id].thr = bt;
        int l = build(li, depth + 1), rr = build(ri, depth + 1);
        nodes[id].left = l; nodes[id].right = rr;
        return id;
    }
    double predict(const Vec& x) const {
        int n = 0;
        while (nodes[n].feat >= 0) n = x[nodes[n].feat] <= nodes[n].thr ? nodes[n].left : nodes[n].right;
        return nodes[n].value;
    }
};

struct GradientBoosting {
    double f0 = 0, lr; vector<RegTree> trees;
    void fit(const Mat& X, const Vec& y, int T, double lr_, int depth, const Mat* Xv = nullptr, const Vec* yv = nullptr) {
        lr = lr_;
        int n = X.size();
        f0 = accumulate(y.begin(), y.end(), 0.0) / n;  // F_0 = média
        Vec F(n, f0);
        trees.clear();
        for (int t = 1; t <= T; ++t) {
            Vec res(n);
            for (int i = 0; i < n; ++i) res[i] = y[i] - F[i];  // -dL/dF para L = (y-F)^2/2
            RegTree tr; tr.fit(X, res, depth);
            for (int i = 0; i < n; ++i) F[i] += lr * tr.predict(X[i]);
            trees.push_back(tr);
            if (Xv && (t == 1 || t % 25 == 0))
                printf("  arvores=%3d  MSE treino=%.4f  MSE teste=%.4f\n", t, mse(X, y), mse(*Xv, *yv));
        }
    }
    double predict(const Vec& x) const {
        double s = f0;
        for (auto& t : trees) s += lr * t.predict(x);
        return s;
    }
    double mse(const Mat& X, const Vec& y) const {
        double s = 0;
        for (size_t i = 0; i < X.size(); ++i) s += pow(predict(X[i]) - y[i], 2);
        return s / X.size();
    }
};

int main() {
    mt19937 rng(0);
    uniform_real_distribution<double> U(0, 2 * M_PI);
    normal_distribution<double> ruido(0, 0.25);
    auto gera = [&](int n, Mat& X, Vec& y) {
        for (int i = 0; i < n; ++i) { double x = U(rng); X.push_back({x}); y.push_back(sin(x) + 0.3 * x + ruido(rng)); }
    };
    Mat Xtr, Xte; Vec ytr, yte;
    gera(300, Xtr, ytr); gera(300, Xte, yte);
    cout << "Gradient Boosting (arvores de profundidade 2, lr=0.1); ruido^2 = 0.0625\n";
    GradientBoosting gb;
    gb.fit(Xtr, ytr, 200, 0.1, 2, &Xte, &yte);
}
